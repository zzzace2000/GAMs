import numpy as np
import pandas as pd

from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from .EncodingBase import LabelEncodingRegressorMixin, LabelEncodingClassifierMixin, OnehotEncodingRegressorMixin, OnehotEncodingClassifierMixin
from .base import MyCommonBase, MyFitMixin


class MyExplainableBoostingMixin(MyCommonBase):
    def __init__(self, random_state=1377, data_n_episodes=20000, n_estimators=1, feature_step_n_inner_bags=0,
        n_jobs=-1, learning_rate=None, min_cases_for_splits=2, binning_strategy='uniform',
        feature_fit_scheme='round_robin', holdout_split=0.176, **kwargs):
        if learning_rate is None:
            learning_rate = 0.01 if isinstance(self, ExplainableBoostingClassifier) else 0.1
        if kwargs.get('feature_fit_scheme', 'round_robin') == 'best_first':
            learning_rate *= 10 # Default set to 0.1

        super(MyExplainableBoostingMixin, self).__init__(
            random_state=random_state, data_n_episodes=data_n_episodes, n_estimators=n_estimators, n_jobs=n_jobs,
            learning_rate=learning_rate, feature_step_n_inner_bags=feature_step_n_inner_bags, min_cases_for_splits=min_cases_for_splits,
            binning_strategy=binning_strategy, feature_fit_scheme=feature_fit_scheme, holdout_split=holdout_split, **kwargs)

    def fit(self, X, y):
        result = super().fit(X, y)

        # Fix the feature_name inconsistencies in EBM model
        if 'feature_0' in self.feature_names:
            self.feature_names = ['f%d' % idx for idx in range(len(self.feature_names))]
        return result

    def get_GAM_plot_dataframe(self, x_values_lookup=None):
        ebm_global = self.explain_global()
        overall_importance = ebm_global.data()['scores']

        results = [{
            'feat_name': 'offset',
            'feat_idx': -1,
            'x': None,
            'y': np.full(1, self.intercept_),
            'importance': -1,
        }]

        for feat_idx, feat_name in enumerate(self.feature_names):
            tmp = ebm_global.data(feat_idx)
            if tmp['type'] == 'pairwise':
                break

            x, y = tmp['names'], tmp['scores']
            x = np.array(x)[~np.isnan(x)]
            if y is None:
                y = np.zeros(len(x))
            y = np.array(y)

            y_std = None
            if 'lower_bounds' in tmp and tmp['lower_bounds'] is not None:
                y_std = y - np.array(tmp['lower_bounds'])

            # interpolate, since sometimes each split would not have the same unique value of x
            if x_values_lookup is not None:
                x_val = x_values_lookup[feat_name]
                if len(x_val) != len(x) or np.any(x_val != x):
                    # transform into integer, then take out the y value from y
                    col_info = self.preprocessor_.schema[list(self.preprocessor_.schema.keys())[feat_idx]]
                    x_idxes = self.preprocessor_.transform_one_column(col_info, x_val)
                    x, y = x_val, y[x_idxes]
                    if y_std is not None:
                        y_std = y_std[x_idxes]


            results.append(dict(
                feat_name=feat_name,
                feat_idx=feat_idx,
                x=x,
                y=y,
                importance=overall_importance[feat_idx],
                **{k: v for k, v in [('y_std', y_std)] if v is not None},
            ))

        return pd.DataFrame(results)

    @property
    def param_distributions(self):
        if self.feature_fit_scheme == 'best_first':
            return {
                'learning_rate': [0.5, 0.1, 0.05, 0.01],
                'min_cases_for_splits': [2, 5, 10, 50, 100, 200],
            }

        return {
            'learning_rate': [0.05, 0.01, 0.005, 0.001],
            'min_cases_for_splits': [2, 5, 10, 50, 100, 200],
        }

class MyExplainableBoostingClassifier(LabelEncodingClassifierMixin, MyExplainableBoostingMixin, ExplainableBoostingClassifier):
    pass


class MyExplainableBoostingRegressor(LabelEncodingRegressorMixin, MyExplainableBoostingMixin, ExplainableBoostingRegressor):
    pass


class MyOnehotExplainableBoostingClassifier(OnehotEncodingClassifierMixin, MyFitMixin, MyExplainableBoostingMixin, ExplainableBoostingClassifier):
    pass


class MyOnehotExplainableBoostingRegressor(OnehotEncodingRegressorMixin, MyFitMixin, MyExplainableBoostingMixin, ExplainableBoostingRegressor):
    pass