# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license


from ...utils import perf_dict
from .utils import EBMUtils
from .internal import NativeEBM
from .postprocessing import multiclass_postprocess
from ...utils import unify_data, autogen_schema
from ...api.base import ExplainerMixin
from ...api.templates import FeatureValueExplanation
from ...utils import JobLibProvider
from ...utils import gen_name_from_class, gen_global_selector, gen_local_selector

import numpy as np
from warnings import warn

from sklearn.base import is_classifier, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import roc_auc_score, mean_squared_error
from collections import Counter

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassifierMixin,
    RegressorMixin,
)
from sklearn.model_selection import train_test_split
from contextlib import closing
from itertools import combinations

import logging

log = logging.getLogger(__name__)


class EBMExplanation(FeatureValueExplanation):
    """ Visualizes specifically for EBM.
    """

    explanation_type = None

    def __init__(
        self,
        explanation_type,
        internal_obj,
        feature_names=None,
        feature_types=None,
        name=None,
        selector=None,
    ):

        super(EBMExplanation, self).__init__(
            explanation_type,
            internal_obj,
            feature_names=feature_names,
            feature_types=feature_types,
            name=name,
            selector=selector,
        )

    def visualize(self, key=None):
        from ...visual.plot import plot_continuous_bar, plot_horizontal_bar, sort_take

        data_dict = self.data(key)
        if data_dict is None:
            return None

        # Overall graph
        # TODO: Fix for multiclass classification
        if self.explanation_type == "global" and key is None:
            data_dict = sort_take(
                data_dict, sort_fn=lambda x: -abs(x), top_n=15, reverse_results=True
            )
            figure = plot_horizontal_bar(
                data_dict,
                title="Overall Importance:<br>Mean Absolute Score",
                start_zero=True,
            )

            return figure

        # Continuous feature graph
        if (
            self.explanation_type == "global"
            and self.feature_types[key] == "continuous"
        ):
            title = self.feature_names[key]
            if (
                isinstance(data_dict["scores"], np.ndarray)
                and data_dict["scores"].ndim == 2
            ):
                figure = plot_continuous_bar(
                    data_dict, multiclass=True, show_error=False, title=title
                )
            else:
                figure = plot_continuous_bar(data_dict, title=title)

            return figure

        return super().visualize(key)


# TODO: More documentation in binning process to be explicit.
# TODO: Consider stripping this down to the bare minimum.
class EBMPreprocessor(BaseEstimator, TransformerMixin):
    """ Transformer that preprocesses data to be ready before EBM. """

    def __init__(
        self,
        schema=None,
        max_n_bins=255,
        missing_constant=0,
        unknown_constant=0,
        feature_names=None,
        binning_strategy="uniform",
    ):
        """ Initializes EBM preprocessor.

        Args:
            schema: A dictionary that encapsulates column information,
                    such as type and domain.
            max_n_bins: Max number of bins to process numeric features.
            missing_constant: Missing encoded as this constant.
            unknown_constant: Unknown encoded as this constant.
            feature_names: Feature names as list.
            binning_strategy: Strategy to compute bins according to density if "quantile" or equidistant if "uniform". 
        """
        self.schema = schema
        self.max_n_bins = max_n_bins
        self.missing_constant = missing_constant
        self.unknown_constant = unknown_constant
        self.feature_names = feature_names
        self.binning_strategy = binning_strategy

    def fit(self, X):
        """ Fits transformer to provided instances.

        Args:
            X: Numpy array for training instances.

        Returns:
            Itself.
        """
        # self.col_bin_counts_ = {}
        self.col_bin_edges_ = {}

        self.hist_counts_ = {}
        self.hist_edges_ = {}

        self.col_mapping_ = {}
        self.col_mapping_counts_ = {}

        self.col_n_bins_ = {}

        self.col_names_ = []
        self.col_types_ = []
        self.has_fitted_ = False

        self.schema_ = (
            self.schema
            if self.schema is not None
            else autogen_schema(X, feature_names=self.feature_names)
        )
        schema = self.schema_

        for col_idx in range(X.shape[1]):
            col_name = list(schema.keys())[col_idx]
            self.col_names_.append(col_name)

            col_info = schema[col_name]
            assert col_info["column_number"] == col_idx
            col_data = X[:, col_idx]

            self.col_types_.append(col_info["type"])
            if col_info["type"] == "continuous":
                col_data = col_data.astype(float)

                uniq_vals = set(col_data[~np.isnan(col_data)])
                if len(uniq_vals) < self.max_n_bins:
                    bins = list(sorted(uniq_vals))
                else:
                    if self.binning_strategy == "uniform":
                        bins = self.max_n_bins
                    elif self.binning_strategy == "quantile":
                        bins = np.unique(
                            np.quantile(
                                col_data, q=np.linspace(0, 1, self.max_n_bins + 1)
                            )
                        )
                    else:
                        raise ValueError(
                            "Unknown binning_strategy: '{}'.".format(
                                self.binning_strategy
                            )
                        )

                _, bin_edges = np.histogram(col_data, bins=bins)

                hist_counts, hist_edges = np.histogram(col_data, bins="doane")
                self.col_bin_edges_[col_idx] = bin_edges

                self.hist_edges_[col_idx] = hist_edges
                self.hist_counts_[col_idx] = hist_counts
                self.col_n_bins_[col_idx] = len(bin_edges)
            elif col_info["type"] == "ordinal":
                mapping = {val: indx for indx, val in enumerate(col_info["order"])}
                self.col_mapping_[col_idx] = mapping
                self.col_n_bins_[col_idx] = len(col_info["order"])
            elif col_info["type"] == "categorical":
                uniq_vals, counts = np.unique(col_data, return_counts=True)

                non_nan_index = ~np.isnan(counts)
                uniq_vals = uniq_vals[non_nan_index]
                counts = counts[non_nan_index]

                mapping = {val: indx for indx, val in enumerate(uniq_vals)}
                self.col_mapping_counts_[col_idx] = counts
                self.col_mapping_[col_idx] = mapping

                # TODO: Review NA as we don't support it yet.
                self.col_n_bins_[col_idx] = len(uniq_vals)

        self.has_fitted_ = True
        return self

    def transform(self, X):
        """ Transform on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Transformed numpy array.
        """
        check_is_fitted(self, "has_fitted_")

        schema = self.schema
        X_new = np.copy(X)
        for col_idx in range(X.shape[1]):
            col_info = schema[list(schema.keys())[col_idx]]
            assert col_info["column_number"] == col_idx
            col_data = X[:, col_idx]

            X_new[:, col_idx] = self.transform_one_column(col_info, col_data)

        return X_new.astype(np.int64)

    def transform_one_column(self, col_info, col_data):
        col_idx = col_info["column_number"]

        if col_info["type"] == "continuous":
            col_data = col_data.astype(float)
            bin_edges = self.col_bin_edges_[col_idx].copy()

            digitized = np.digitize(col_data, bin_edges, right=False)
            digitized[digitized == 0] = 1
            digitized -= 1

            # NOTE: NA handling done later.
            # digitized[np.isnan(col_data)] = self.missing_constant
            return digitized
        elif col_info["type"] == "ordinal":
            mapping = self.col_mapping_[col_idx]
            mapping[np.nan] = self.missing_constant
            vec_map = np.vectorize(
                lambda x: mapping[x] if x in mapping else self.unknown_constant
            )
            return vec_map(col_data)
        elif col_info["type"] == "categorical":
            mapping = self.col_mapping_[col_idx]
            mapping[np.nan] = self.missing_constant
            vec_map = np.vectorize(
                lambda x: mapping[x] if x in mapping else self.unknown_constant
            )
            return vec_map(col_data)

        return col_data

    def get_hist_counts(self, attribute_index):
        col_type = self.col_types_[attribute_index]
        if col_type == "continuous":
            return list(self.hist_counts_[attribute_index])
        elif col_type == "categorical":
            return list(self.col_mapping_counts_[attribute_index])
        else:  # pragma: no cover
            raise Exception("Cannot get counts for type: {0}".format(col_type))

    def get_hist_edges(self, attribute_index):
        col_type = self.col_types_[attribute_index]
        if col_type == "continuous":
            return list(self.hist_edges_[attribute_index])
        elif col_type == "categorical":
            map = self.col_mapping_[attribute_index]
            return list(map.keys())
        else:  # pragma: no cover
            raise Exception("Cannot get counts for type: {0}".format(col_type))

    # def get_bin_counts(self, attribute_index):
    #     col_type = self.col_types_[attribute_index]
    #     if col_type == 'continuous':
    #         return list(self.col_bin_counts_[attribute_index])
    #     elif col_type == 'categorical':
    #         return list(self.col_mapping_counts_[attribute_index])
    #     else:
    #         raise Exception("Cannot get counts for type: {0}".format(col_type))

    def get_bin_labels(self, attribute_index):
        """ Returns bin labels for a given attribute index.

        Args:
            attribute_index: An integer for attribute index.

        Returns:
            List of labels for bins.
        """

        col_type = self.col_types_[attribute_index]
        if col_type == "continuous":
            return list(self.col_bin_edges_[attribute_index])
        elif col_type == "ordinal":
            map = self.col_mapping_[attribute_index]
            return list(map.keys())
        elif col_type == "categorical":
            map = self.col_mapping_[attribute_index]
            return list(map.keys())
        else:  # pragma: no cover
            raise Exception("Unknown column type")


# TODO: Clean up
class BaseCoreEBM(BaseEstimator):
    """Internal use EBM."""

    def __init__(
        self,
        # Data
        col_types=None,
        col_n_bins=None,
        # Core
        interactions=0,
        holdout_split=0.15,
        data_n_episodes=2000,
        early_stopping_tolerance=1e-5,
        early_stopping_run_length=50,
        feature_fit_scheme='round_robin',
        # Native
        feature_step_n_inner_bags=0,
        learning_rate=0.01,
        training_step_episodes=1,
        max_tree_splits=2,
        min_cases_for_splits=2,
        # Overall
        random_state=42,
    ):

        # Arguments for data
        self.col_types = col_types
        self.col_n_bins = col_n_bins

        # Arguments for EBM beyond training a feature-step.
        self.interactions = interactions
        self.holdout_split = holdout_split
        self.data_n_episodes = data_n_episodes
        self.early_stopping_tolerance = early_stopping_tolerance
        self.early_stopping_run_length = early_stopping_run_length
        self.feature_fit_scheme = feature_fit_scheme

        # Arguments for internal EBM.
        self.feature_step_n_inner_bags = feature_step_n_inner_bags
        self.learning_rate = learning_rate
        self.training_step_episodes = training_step_episodes
        self.max_tree_splits = max_tree_splits
        self.min_cases_for_splits = min_cases_for_splits

        # Arguments for overall
        self.random_state = random_state

    def fit(self, X, y):
        if is_classifier(self):
            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
        else:
            self.n_classes_ = -1

        # Split data into train/val

        if self.holdout_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.holdout_split,
                random_state=self.random_state,
                stratify=y if is_classifier(self) else None,
            )
        elif self.holdout_split == 0:
            X_train = X
            y_train = y
            X_val = np.empty(shape=(0, 0)).astype(np.int64)
            y_val = np.empty(shape=(0,)).astype(np.int64)
        else:  # pragma: no cover
            raise Exception("Holdout_split must be between 0 and 1.")
        # Define attributes
        self.attributes_ = EBMUtils.gen_attributes(self.col_types, self.col_n_bins)
        # Build EBM allocation code
        if is_classifier(self):
            model_type = "classification"
        else:
            model_type = "regression"

        # For multiclass, need an intercept term per class
        if self.n_classes_ > 2:
            self.intercept_ = [0] * self.n_classes_
        else:
            self.intercept_ = 0

        self.attribute_sets_ = []
        self.attribute_set_models_ = []

        main_attr_indices = [[x] for x in range(len(self.attributes_))]
        main_attr_sets = EBMUtils.gen_attribute_sets(main_attr_indices)
        with closing(
            NativeEBM(
                self.attributes_,
                main_attr_sets,
                X_train,
                y_train,
                X_val,
                y_val,
                num_inner_bags=self.feature_step_n_inner_bags,
                num_classification_states=self.n_classes_,
                model_type=model_type,
                training_scores=None,
                validation_scores=None,
            )
        ) as native_ebm:
            # Train main effects
            self._fit_main(native_ebm, main_attr_sets)

            # Build interaction terms
            self.inter_indices_ = self._build_interactions(native_ebm)

        self.staged_fit_interactions(X, y, self.inter_indices_)

        return self

    def _build_interactions(self, native_ebm):
        if isinstance(self.interactions, int) and self.interactions != 0:
            log.info("Estimating with FAST")
            interaction_scores = []
            interaction_indices = [
                x for x in combinations(range(len(self.col_types)), 2)
            ]
            for pair in interaction_indices:
                score = native_ebm.fast_interaction_score(pair)
                interaction_scores.append((pair, score))

            ranked_scores = list(
                sorted(interaction_scores, key=lambda x: x[1], reverse=True)
            )
            n_interactions = min(len(ranked_scores), self.interactions)

            inter_indices_ = [x[0] for x in ranked_scores[0:n_interactions]]
        elif isinstance(self.interactions, int) and self.interactions == 0:
            inter_indices_ = []
        elif isinstance(self.interactions, list):
            inter_indices_ = self.interactions
        else:  # pragma: no cover
            raise RuntimeError("Argument 'interaction' has invalid value")

        return inter_indices_

    def _fit_main(self, native_ebm, main_attr_sets):
        log.info("Train main effects")
        self.current_metric_, self.main_episode_idx_ = self._cyclic_gradient_boost(
            native_ebm, main_attr_sets, "Main"
        )
        log.debug("Main Metric: {0}".format(self.current_metric_))
        for index, attr_set in enumerate(main_attr_sets):
            attribute_set_model = native_ebm.get_best_model(index)
            self.attribute_set_models_.append(attribute_set_model)
            self.attribute_sets_.append(attr_set)

        self.has_fitted_ = True

        return self

    def staged_fit_interactions(self, X, y, inter_indices=[]):
        check_is_fitted(self, "has_fitted_")

        self.inter_episode_idx_ = 0
        if len(inter_indices) == 0:
            log.info("No interactions to train")
            return self

        log.info("Training interactions")

        # Split data into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.holdout_split,
            random_state=self.random_state,
            stratify=y if is_classifier(self) else None,
        )
        if is_classifier(self):
            model_type = "classification"
        else:
            model_type = "regression"

        # Discard initial interactions
        new_attribute_set_models = []
        new_attribute_sets = []
        for i, attribute_set in enumerate(self.attribute_sets_):
            if attribute_set["n_attributes"] != 1:
                continue
            new_attribute_set_models.append(self.attribute_set_models_[i])
            new_attribute_sets.append(self.attribute_sets_[i])
        self.attribute_set_models_ = new_attribute_set_models
        self.attribute_sets_ = new_attribute_sets

        # Fix main, train interactions
        training_scores = self.decision_function(X_train)
        validation_scores = self.decision_function(X_val)
        inter_attr_sets = EBMUtils.gen_attribute_sets(inter_indices)
        with closing(
            NativeEBM(
                self.attributes_,
                inter_attr_sets,
                X_train,
                y_train,
                X_val,
                y_val,
                num_inner_bags=self.feature_step_n_inner_bags,
                num_classification_states=self.n_classes_,
                model_type=model_type,
                training_scores=training_scores,
                validation_scores=validation_scores,
                random_state=self.random_state,
            )
        ) as native_ebm:
            log.info("Train interactions")
            self.current_metric_, self.inter_episode_idx_ = self._cyclic_gradient_boost(
                native_ebm, inter_attr_sets, "Pair"
            )
            log.debug("Interaction Metric: {0}".format(self.current_metric_))

            for index, attr_set in enumerate(inter_attr_sets):
                self.attribute_set_models_.append(native_ebm.get_best_model(index))
                self.attribute_sets_.append(attr_set)

        return self

    def decision_function(self, X):
        check_is_fitted(self, "has_fitted_")

        return EBMUtils.decision_function(
            X, self.attribute_sets_, self.attribute_set_models_, 0
        )

    def _cyclic_gradient_boost(self, native_ebm, attribute_sets, name=None):

        no_change_run_length = 0
        curr_metric = np.inf
        min_metric = np.inf
        bp_metric = np.inf
        log.info("Start boosting {0}".format(name))
        curr_episode_index = 0
        for data_episode_index in range(self.data_n_episodes):
            curr_episode_index = data_episode_index

            if data_episode_index % 10 == 0:
                log.debug("Sweep Index for {0}: {1}".format(name, data_episode_index))
                log.debug("Metric: {0}".format(curr_metric))

            if len(attribute_sets) == 0:
                log.debug("No sets to boost for {0}".format(name))

            if self.feature_fit_scheme == 'round_robin':
                for index, attribute_set in enumerate(attribute_sets):
                    curr_metric = native_ebm.training_step(
                        index,
                        training_step_episodes=self.training_step_episodes,
                        learning_rate=self.learning_rate,
                        max_tree_splits=self.max_tree_splits,
                        min_cases_for_split=self.min_cases_for_splits,
                        training_weights=0,
                        validation_weights=0,
                    )
            elif self.feature_fit_scheme == 'best_first':
                min_gain = np.inf
                min_index = None
                for index, attribute_set in enumerate(attribute_sets):
                    gain = native_ebm.training_peek(
                        index,
                        training_step_episodes=self.training_step_episodes,
                        learning_rate=self.learning_rate,
                        max_tree_splits=self.max_tree_splits,
                        min_cases_for_split=self.min_cases_for_splits,
                        training_weights=0,
                        validation_weights=0,
                    )

                    if gain < min_gain:
                        min_gain = gain
                        min_index = index

                log.debug("min gain: {0} with feature index {1}".format(min_gain, min_index))

                curr_metric = native_ebm.training_step(
                    min_index,
                    training_step_episodes=self.training_step_episodes,
                    learning_rate=self.learning_rate,
                    max_tree_splits=self.max_tree_splits,
                    min_cases_for_split=self.min_cases_for_splits,
                    training_weights=0,
                    validation_weights=0,
                )
            else:
                raise RuntimeError("Argument 'feature_fit_scheme' has invalid value")

            # NOTE: Out of per-feature boosting on purpose.
            min_metric = min(curr_metric, min_metric)

            if no_change_run_length == 0:
                bp_metric = min_metric
            if curr_metric + self.early_stopping_tolerance < bp_metric:
                no_change_run_length = 0
            else:
                no_change_run_length += 1

            if (
                self.early_stopping_run_length >= 0
                and no_change_run_length >= self.early_stopping_run_length
            ):
                log.info("Early break {0}: {1}".format(name, data_episode_index))
                break
        log.info("End boosting {0}".format(name))

        return curr_metric, curr_episode_index


class CoreEBMClassifier(BaseCoreEBM, ClassifierMixin):
    def __init__(
        self,
        # Data
        col_types=None,
        col_n_bins=None,
        # Core
        interactions=0,
        holdout_split=0.15,
        data_n_episodes=2000,
        early_stopping_tolerance=1e-5,
        early_stopping_run_length=50,
        feature_fit_scheme='round_robin',
        # Native
        feature_step_n_inner_bags=0,
        learning_rate=0.01,
        training_step_episodes=1,
        max_tree_splits=2,
        min_cases_for_splits=2,
        # Overall
        random_state=42,
    ):
        super(CoreEBMClassifier, self).__init__(
            # Data
            col_types=col_types,
            col_n_bins=col_n_bins,
            # Core
            interactions=interactions,
            holdout_split=holdout_split,
            data_n_episodes=data_n_episodes,
            early_stopping_tolerance=early_stopping_tolerance,
            early_stopping_run_length=early_stopping_run_length,
            feature_fit_scheme=feature_fit_scheme,
            # Native
            feature_step_n_inner_bags=feature_step_n_inner_bags,
            learning_rate=learning_rate,
            training_step_episodes=training_step_episodes,
            max_tree_splits=max_tree_splits,
            min_cases_for_splits=min_cases_for_splits,
            # Overall
            random_state=random_state,
        )

    def predict_proba(self, X):
        check_is_fitted(self, "has_fitted_")
        prob = EBMUtils.classifier_predict_proba(X, self)
        return prob

    def predict(self, X):
        check_is_fitted(self, "has_fitted_")
        return EBMUtils.classifier_predict(X, self)


class CoreEBMRegressor(BaseCoreEBM, RegressorMixin):
    def __init__(
        self,
        # Data
        col_types=None,
        col_n_bins=None,
        # Core
        interactions=0,
        holdout_split=0.15,
        data_n_episodes=2000,
        early_stopping_tolerance=1e-5,
        early_stopping_run_length=50,
        feature_fit_scheme='round_robin',
        # Native
        feature_step_n_inner_bags=0,
        learning_rate=0.01,
        training_step_episodes=1,
        max_tree_splits=2,
        min_cases_for_splits=2,
        # Overall
        random_state=42,
    ):
        super(CoreEBMRegressor, self).__init__(
            # Data
            col_types=col_types,
            col_n_bins=col_n_bins,
            # Core
            interactions=interactions,
            holdout_split=holdout_split,
            data_n_episodes=data_n_episodes,
            early_stopping_tolerance=early_stopping_tolerance,
            early_stopping_run_length=early_stopping_run_length,
            feature_fit_scheme=feature_fit_scheme,
            # Native
            feature_step_n_inner_bags=feature_step_n_inner_bags,
            learning_rate=learning_rate,
            training_step_episodes=training_step_episodes,
            max_tree_splits=max_tree_splits,
            min_cases_for_splits=min_cases_for_splits,
            # Overall
            random_state=random_state,
        )

    def predict(self, X):
        check_is_fitted(self, "has_fitted_")
        return EBMUtils.regressor_predict(X, self)


class BaseEBM(BaseEstimator):
    """Client facing SK EBM."""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Data
        schema=None,
        # Ensemble
        n_estimators=16,
        holdout_size=0.15,
        scoring=None,
        # Core
        interactions=0,
        holdout_split=0.15,
        data_n_episodes=2000,
        early_stopping_tolerance=1e-5,
        early_stopping_run_length=50,
        feature_fit_scheme='round_robin',
        # Native
        feature_step_n_inner_bags=0,
        learning_rate=0.01,
        training_step_episodes=1,
        max_tree_splits=2,
        min_cases_for_splits=2,
        # Overall
        n_jobs=-2,
        random_state=42,
        # Preprocessor
        binning_strategy="uniform",
    ):

        # Arguments for explainer
        self.feature_names = feature_names
        self.feature_types = feature_types

        # Arguments for data
        self.schema = schema

        # Arguments for ensemble
        self.n_estimators = n_estimators
        self.holdout_size = holdout_size
        self.scoring = scoring

        # Arguments for EBM beyond training a feature-step.
        self.interactions = interactions
        self.holdout_split = holdout_split
        self.data_n_episodes = data_n_episodes
        self.early_stopping_tolerance = early_stopping_tolerance
        self.early_stopping_run_length = early_stopping_run_length
        self.feature_fit_scheme = feature_fit_scheme

        # Arguments for internal EBM.
        self.feature_step_n_inner_bags = feature_step_n_inner_bags
        self.learning_rate = learning_rate
        self.training_step_episodes = training_step_episodes
        self.max_tree_splits = max_tree_splits
        self.min_cases_for_splits = min_cases_for_splits

        # Arguments for overall
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Arguments for preprocessor
        self.binning_strategy = binning_strategy

    def fit(self, X, y):
        X, y, self.feature_names, _ = unify_data(
            X, y, self.feature_names, self.feature_types
        )

        # Build preprocessor
        self.schema_ = self.schema
        if self.schema_ is None:
            self.schema_ = autogen_schema(
                X, feature_names=self.feature_names, feature_types=self.feature_types
            )

        self.preprocessor_ = EBMPreprocessor(
            schema=self.schema_, binning_strategy=self.binning_strategy
        )
        self.preprocessor_.fit(X)

        if is_classifier(self):
            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
            if self.n_classes_ > 2:
                warn("Multiclass is still experimental. Subject to change per release.")
            if self.n_classes_ > 2 and self.interactions != 0:
                raise RuntimeError(
                    "Multiclass with interactions currently not supported."
                )

            proto_estimator = CoreEBMClassifier(
                # Data
                col_types=self.preprocessor_.col_types_,
                col_n_bins=self.preprocessor_.col_n_bins_,
                # Core
                interactions=self.interactions,
                holdout_split=self.holdout_split,
                data_n_episodes=self.data_n_episodes,
                early_stopping_tolerance=self.early_stopping_tolerance,
                early_stopping_run_length=self.early_stopping_run_length,
                feature_fit_scheme=self.feature_fit_scheme,
                # Native
                feature_step_n_inner_bags=self.feature_step_n_inner_bags,
                learning_rate=self.learning_rate,
                training_step_episodes=self.training_step_episodes,
                max_tree_splits=self.max_tree_splits,
                min_cases_for_splits=self.min_cases_for_splits,
                # Overall
                random_state=self.random_state,
            )
        else:
            self.n_classes_ = -1
            proto_estimator = CoreEBMRegressor(
                # Data
                col_types=self.preprocessor_.col_types_,
                col_n_bins=self.preprocessor_.col_n_bins_,
                # Core
                interactions=self.interactions,
                holdout_split=self.holdout_split,
                data_n_episodes=self.data_n_episodes,
                early_stopping_tolerance=self.early_stopping_tolerance,
                early_stopping_run_length=self.early_stopping_run_length,
                feature_fit_scheme=self.feature_fit_scheme,
                # Native
                feature_step_n_inner_bags=self.feature_step_n_inner_bags,
                learning_rate=self.learning_rate,
                training_step_episodes=self.training_step_episodes,
                max_tree_splits=self.max_tree_splits,
                min_cases_for_splits=self.min_cases_for_splits,
                # Overall
                random_state=self.random_state,
            )

        # Train base models for main effects, pair detection.

        # Intercept needs to be a list for multiclass
        if self.n_classes_ > 2:
            self.intercept_ = [0] * self.n_classes_
        else:
            self.intercept_ = 0
        X_orig = X
        X = self.preprocessor_.transform(X)
        estimators = []
        for i in range(self.n_estimators):
            estimator = clone(proto_estimator)
            estimator.set_params(random_state=self.random_state + i)
            estimators.append(estimator)

        provider = JobLibProvider(n_jobs=self.n_jobs)

        def train_model(estimator, X, y):
            return estimator.fit(X, y)

        train_model_args_iter = (
            (estimators[i], X, y) for i in range(self.n_estimators)
        )

        estimators = provider.parallel(train_model, train_model_args_iter)

        if isinstance(self.interactions, int) and self.interactions > 0:
            # Select merged pairs
            pair_indices = self._select_merged_pairs(estimators, X, y)

            # Retrain interactions for base models
            def staged_fit_fn(estimator, X, y, inter_indices=[]):
                return estimator.staged_fit_interactions(X, y, inter_indices)

            staged_fit_args_iter = (
                (estimators[i], X, y, pair_indices) for i in range(self.n_estimators)
            )

            estimators = provider.parallel(staged_fit_fn, staged_fit_args_iter)
        elif isinstance(self.interactions, int) and self.interactions == 0:
            pair_indices = []
        elif isinstance(self.interactions, list):
            pair_indices = self.interactions
        else:  # pragma: no cover
            raise RuntimeError("Argument 'interaction' has invalid value")

        # Average base models into one.
        self.attributes_ = EBMUtils.gen_attributes(
            self.preprocessor_.col_types_, self.preprocessor_.col_n_bins_
        )
        main_indices = [[x] for x in range(len(self.attributes_))]
        self.attribute_sets_ = EBMUtils.gen_attribute_sets(main_indices)
        self.attribute_sets_.extend(EBMUtils.gen_attribute_sets(pair_indices))

        # Merge estimators into one.
        self.attribute_set_models_ = []
        self.model_errors_ = []
        for index, _ in enumerate(self.attribute_sets_):
            log_odds_tensors = []
            for estimator in estimators:
                log_odds_tensors.append(estimator.attribute_set_models_[index])

            averaged_model = np.average(np.array(log_odds_tensors), axis=0)
            model_errors = np.std(np.array(log_odds_tensors), axis=0)

            self.attribute_set_models_.append(averaged_model)
            self.model_errors_.append(model_errors)

        # Get episode indexes for base estimators.
        self.main_episode_idxs_ = []
        self.inter_episode_idxs_ = []
        for estimator in estimators:
            self.main_episode_idxs_.append(estimator.main_episode_idx_)
            self.inter_episode_idxs_.append(estimator.inter_episode_idx_)

        # Extract feature names and feature types.
        self.feature_names = []
        self.feature_types = []
        for index, attribute_set in enumerate(self.attribute_sets_):
            feature_name = EBMUtils.gen_feature_name(
                attribute_set["attributes"], self.preprocessor_.col_names_
            )
            feature_type = EBMUtils.gen_feature_type(
                attribute_set["attributes"], self.preprocessor_.col_types_
            )
            self.feature_types.append(feature_type)
            self.feature_names.append(feature_name)

        # Mean center graphs - only for binary classification and regression
        if self.n_classes_ <= 2:
            scores_gen = EBMUtils.scores_by_attrib_set(
                X, self.attribute_sets_, self.attribute_set_models_, []
            )
            self._attrib_set_model_means_ = []

            # TODO: Clean this up before release.
            for set_idx, attribute_set, scores in scores_gen:
                score_mean = np.mean(scores)

                self.attribute_set_models_[set_idx] = (
                    self.attribute_set_models_[set_idx] - score_mean
                )

                # Add mean center adjustment back to intercept
                self.intercept_ = self.intercept_ + score_mean
                self._attrib_set_model_means_.append(score_mean)

        # Postprocess model graphs for multiclass
        if self.n_classes_ > 2:
            binned_predict_proba = lambda x: EBMUtils.classifier_predict_proba(x, self)

            postprocessed = multiclass_postprocess(
                X, self.attribute_set_models_, binned_predict_proba, self.feature_types
            )
            self.attribute_set_models_ = postprocessed["feature_graphs"]
            self.intercept_ = postprocessed["intercepts"]

        # Generate overall importance
        scores_gen = EBMUtils.scores_by_attrib_set(
            X, self.attribute_sets_, self.attribute_set_models_, []
        )
        self.mean_abs_scores_ = []
        for set_idx, attribute_set, scores in scores_gen:
            mean_abs_score = np.mean(np.abs(scores))
            self.mean_abs_scores_.append(mean_abs_score)

        # Generate selector
        self.global_selector = gen_global_selector(
            X_orig, self.feature_names, self.feature_types, None
        )

        self.has_fitted_ = True
        return self

    def _select_merged_pairs(self, estimators, X, y):
        # Select pairs from base models
        def score_fn(est, X, y, drop_indices):
            if is_classifier(est):
                prob = EBMUtils.classifier_predict_proba(X, estimator, drop_indices)
                return -1.0 * roc_auc_score(y, prob[:, 1])
            else:
                pred = EBMUtils.regressor_predict(X, estimator, drop_indices)
                return mean_squared_error(y, pred)

        pair_cum_rank = Counter()
        pair_freq = Counter()
        for index, estimator in enumerate(estimators):
            backward_impacts = []
            forward_impacts = []

            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.holdout_split,
                random_state=estimator.random_state,
                stratify=y if is_classifier(self) else None,
            )
            base_forward_score = score_fn(
                estimator, X_val, y_val, estimator.inter_indices_
            )
            base_backward_score = score_fn(estimator, X_val, y_val, [])
            for pair_idx, pair in enumerate(estimator.inter_indices_):
                pair_freq[pair] += 1
                backward_score = score_fn(
                    estimator, X_val, y_val, estimator.inter_indices_[pair_idx]
                )
                forward_score = score_fn(
                    estimator,
                    X_val,
                    y_val,
                    estimator.inter_indices_[:pair_idx]
                    + estimator.inter_indices_[pair_idx + 1 :],
                )
                backward_impact = backward_score - base_backward_score
                forward_impact = base_forward_score - forward_score

                backward_impacts.append(backward_impact)
                forward_impacts.append(forward_impact)

            # Average ranks
            backward_ranks = np.argsort(backward_impacts[::-1])
            forward_ranks = np.argsort(forward_impacts[::-1])
            pair_ranks = np.mean(np.array([backward_ranks, forward_ranks]), axis=0)

            # Add to cumulative rank for a pair across all models
            for pair_idx, pair in enumerate(estimator.inter_indices_):
                pair_cum_rank[pair] += pair_ranks[pair_idx]

        # Calculate pair importance ranks
        pair_weighted_ranks = pair_cum_rank.copy()
        for pair, freq in pair_freq.items():
            # Calculate average rank
            pair_weighted_ranks[pair] /= freq
            # Reweight by frequency
            pair_weighted_ranks[pair] /= np.sqrt(freq)
        pair_weighted_ranks = sorted(pair_weighted_ranks.items(), key=lambda x: x[1])

        # Retrieve top K pairs
        pair_indices = [x[0] for x in pair_weighted_ranks[: self.interactions]]

        return pair_indices

    def decision_function(self, X):
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)

        decision_scores = EBMUtils.decision_function(
            X, self.attribute_sets_, self.attribute_set_models_, self.intercept_
        )

        return decision_scores

    def explain_global(self, name=None):
        if name is None:
            name = gen_name_from_class(self)

        # Obtain min/max for model scores
        lower_bound = np.inf
        upper_bound = -np.inf
        for attribute_set_index, attribute_set in enumerate(self.attribute_sets_):
            errors = self.model_errors_[attribute_set_index]
            scores = self.attribute_set_models_[attribute_set_index]

            lower_bound = min(lower_bound, np.min(scores - errors))
            upper_bound = max(upper_bound, np.max(scores + errors))

        bounds = (lower_bound, upper_bound)

        # Add per feature graph
        data_dicts = []
        feature_list = []
        density_list = []
        for attribute_set_index, attribute_set in enumerate(self.attribute_sets_):
            model_graph = self.attribute_set_models_[attribute_set_index]

            # NOTE: This uses stddev. for bounds, consider issue warnings.
            errors = self.model_errors_[attribute_set_index]
            attribute_indexes = self.attribute_sets_[attribute_set_index]["attributes"]

            if len(attribute_indexes) == 1:
                bin_labels = self.preprocessor_.get_bin_labels(attribute_indexes[0])
                # bin_counts = self.preprocessor_.get_bin_counts(
                #     attribute_indexes[0]
                # )
                scores = list(model_graph)
                upper_bounds = list(model_graph + errors)
                lower_bounds = list(model_graph - errors)
                density_dict = {
                    "names": self.preprocessor_.get_hist_edges(attribute_indexes[0]),
                    "scores": self.preprocessor_.get_hist_counts(attribute_indexes[0]),
                }

                feature_dict = {
                    "type": "univariate",
                    "names": bin_labels,
                    "scores": scores,
                    "scores_range": bounds,
                    "upper_bounds": upper_bounds,
                    "lower_bounds": lower_bounds,
                }
                feature_list.append(feature_dict)
                density_list.append(density_dict)

                data_dict = {
                    "type": "univariate",
                    "names": bin_labels,
                    "scores": model_graph,
                    "scores_range": bounds,
                    "upper_bounds": model_graph + errors,
                    "lower_bounds": model_graph - errors,
                    "density": {
                        "names": self.preprocessor_.get_hist_edges(
                            attribute_indexes[0]
                        ),
                        "scores": self.preprocessor_.get_hist_counts(
                            attribute_indexes[0]
                        ),
                    },
                }
                data_dicts.append(data_dict)
            elif len(attribute_indexes) == 2:
                bin_labels_left = self.preprocessor_.get_bin_labels(
                    attribute_indexes[0]
                )
                bin_labels_right = self.preprocessor_.get_bin_labels(
                    attribute_indexes[1]
                )

                feature_dict = {
                    "type": "pairwise",
                    "left_names": bin_labels_left,
                    "right_names": bin_labels_right,
                    "scores": model_graph,
                    "scores_range": bounds,
                }
                feature_list.append(feature_dict)
                density_list.append({})

                data_dict = {
                    "type": "pairwise",
                    "left_names": bin_labels_left,
                    "right_names": bin_labels_right,
                    "scores": model_graph,
                    "scores_range": bounds,
                }
                data_dicts.append(data_dict)
            else:  # pragma: no cover
                raise Exception("Interactions greater than 2 not supported.")

        overall_dict = {
            "type": "univariate",
            "names": self.feature_names,
            "scores": self.mean_abs_scores_,
        }
        internal_obj = {
            "overall": overall_dict,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_global",
                    "value": {"feature_list": feature_list},
                },
                {"explanation_type": "density", "value": {"density": density_list}},
            ],
        }

        return EBMExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=self.global_selector,
        )

    def explain_local(self, X, y=None, name=None):
        # Produce feature value pairs for each instance.
        # Values are the model graph score per respective attribute set.
        if name is None:
            name = gen_name_from_class(self)

        X, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types)
        instances = self.preprocessor_.transform(X)
        scores_gen = EBMUtils.scores_by_attrib_set(
            instances, self.attribute_sets_, self.attribute_set_models_
        )

        n_rows = instances.shape[0]
        data_dicts = []
        for _ in range(n_rows):
            data_dict = {
                "type": "univariate",
                "names": [],
                "scores": [],
                "values": [],
                "extra": {
                    "names": ["Intercept"],
                    "scores": [self.intercept_],
                    "values": [1],
                },
            }
            data_dicts.append(data_dict)

        for set_idx, attribute_set, scores in scores_gen:
            for row_idx in range(n_rows):
                feature_name = self.feature_names[set_idx]
                data_dicts[row_idx]["names"].append(feature_name)
                data_dicts[row_idx]["scores"].append(scores[row_idx])
                if attribute_set["n_attributes"] == 1:
                    data_dicts[row_idx]["values"].append(
                        X[row_idx, attribute_set["attributes"][0]]
                    )
                else:
                    data_dicts[row_idx]["values"].append("")

        if is_classifier(self):
            scores = EBMUtils.classifier_predict_proba(instances, self)[:, 1]
        else:
            scores = EBMUtils.regressor_predict(instances, self)

        perf_list = []
        for row_idx in range(n_rows):
            perf = perf_dict(y, scores, row_idx)
            perf_list.append(perf)
            data_dicts[row_idx]["perf"] = perf

        selector = gen_local_selector(instances, y, scores)

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_local",
                    "value": {
                        "scores": self.attribute_set_models_,
                        "intercept": self.intercept_,
                        "perf": perf_list,
                    },
                }
            ],
        }
        internal_obj["mli"].append(
            {
                "explanation_type": "evaluation_dataset",
                "value": {"dataset_x": X, "dataset_y": y},
            }
        )

        return EBMExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )


class ExplainableBoostingClassifier(BaseEBM, ClassifierMixin, ExplainerMixin):
    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing EBM classifier."""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Data
        schema=None,
        # Ensemble
        n_estimators=16,
        holdout_size=0.15,
        scoring=None,
        # Core
        interactions=0,
        holdout_split=0.15,
        data_n_episodes=2000,
        early_stopping_tolerance=1e-5,
        early_stopping_run_length=50,
        feature_fit_scheme='round_robin',
        # Native
        feature_step_n_inner_bags=0,
        learning_rate=0.01,
        training_step_episodes=1,
        max_tree_splits=2,
        min_cases_for_splits=2,
        # Overall
        n_jobs=-2,
        random_state=42,
        # Preprocessor
        binning_strategy="uniform",
    ):

        super(ExplainableBoostingClassifier, self).__init__(
            # Explainer
            feature_names=feature_names,
            feature_types=feature_types,
            # Data
            schema=schema,
            # Ensemble
            n_estimators=n_estimators,
            holdout_size=holdout_size,
            scoring=scoring,
            # Core
            interactions=interactions,
            holdout_split=holdout_split,
            data_n_episodes=data_n_episodes,
            early_stopping_tolerance=early_stopping_tolerance,
            early_stopping_run_length=early_stopping_run_length,
            feature_fit_scheme=feature_fit_scheme,
            # Native
            feature_step_n_inner_bags=feature_step_n_inner_bags,
            learning_rate=learning_rate,
            training_step_episodes=training_step_episodes,
            max_tree_splits=max_tree_splits,
            min_cases_for_splits=min_cases_for_splits,
            # Overall
            n_jobs=n_jobs,
            random_state=random_state,
            # Preprocessor
            binning_strategy=binning_strategy,
        )

    # TODO: Throw ValueError like scikit for 1d instead of 2d arrays
    def predict_proba(self, X):
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)
        prob = EBMUtils.classifier_predict_proba(X, self)
        return prob

    def predict(self, X):
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)
        return EBMUtils.classifier_predict(X, self)


class ExplainableBoostingRegressor(BaseEBM, RegressorMixin, ExplainerMixin):
    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing EBM regressor."""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Data
        schema=None,
        # Ensemble
        n_estimators=16,
        holdout_size=0.15,
        scoring=None,
        # Core
        interactions=0,
        holdout_split=0.15,
        data_n_episodes=2000,
        early_stopping_tolerance=1e-5,
        early_stopping_run_length=50,
        feature_fit_scheme='round_robin',
        # Native
        feature_step_n_inner_bags=0,
        learning_rate=0.01,
        training_step_episodes=1,
        max_tree_splits=2,
        min_cases_for_splits=2,
        # Overall
        n_jobs=-2,
        random_state=42,
        # Preprocessor
        binning_strategy="uniform",
    ):

        super(ExplainableBoostingRegressor, self).__init__(
            # Explainer
            feature_names=feature_names,
            feature_types=feature_types,
            # Data
            schema=schema,
            # Ensemble
            n_estimators=n_estimators,
            holdout_size=holdout_size,
            scoring=scoring,
            # Core
            interactions=interactions,
            holdout_split=holdout_split,
            data_n_episodes=data_n_episodes,
            early_stopping_tolerance=early_stopping_tolerance,
            early_stopping_run_length=early_stopping_run_length,
            feature_fit_scheme=feature_fit_scheme,
            # Native
            feature_step_n_inner_bags=feature_step_n_inner_bags,
            learning_rate=learning_rate,
            training_step_episodes=training_step_episodes,
            max_tree_splits=max_tree_splits,
            min_cases_for_splits=min_cases_for_splits,
            # Overall
            n_jobs=n_jobs,
            random_state=random_state,
            # Preprocessor
            binning_strategy=binning_strategy,
        )

    def predict(self, X):
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)
        return EBMUtils.regressor_predict(X, self)
