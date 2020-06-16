import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split

from .EncodingBase import LabelEncodingRegressorMixin, LabelEncodingClassifierMixin, OnehotEncodingClassifierMixin, OnehotEncodingRegressorMixin
from .base import MyGAMPlotMixinBase
from .utils import sigmoid
import copy


class MyXGBMixin(object):
    def __init__(
        self,
        max_depth=1,
        random_state=1377,
        learning_rate=None,
        n_estimators=30000,
        min_child_weight=1,
        tree_method='exact',
        reg_lambda=0,
        n_jobs=-1,
        objective=None,
        missing=None,
        colsample_bytree=1.,
        subsample=1.,
        # My own parameter
        holdout_split=0.176, # 85% * 0.176 = 15%
        early_stopping_rounds=50,
        **kwargs,
    ):
        if objective is None:
            objective = 'binary:logistic' if isinstance(self, XGBClassifier) else 'reg:squarederror'

        if learning_rate is None:
            learning_rate = 0.1 if isinstance(self, XGBClassifier) else 1.0

        super(MyXGBMixin, self).__init__(max_depth=max_depth, random_state=random_state, learning_rate=learning_rate,
            n_estimators=n_estimators, min_child_weight=min_child_weight, tree_method=tree_method, reg_lambda=reg_lambda,
            n_jobs=n_jobs, objective=objective, missing=missing, colsample_bytree=colsample_bytree,
            subsample=subsample, **kwargs)

        self.holdout_split = holdout_split
        self.early_stopping_rounds = early_stopping_rounds
        self.onehot_columns = None
        self.clean_feat_names = None

    def fit(self, X, y, verbose=False, **kwargs):
        self.clean_feat_names = []
        for feat in self.feature_names:
            if '<' in feat:
                feat = feat.replace('<', 'under')

            self.clean_feat_names.append(feat)

        if isinstance(X, pd.DataFrame): # XGB can not accept feature names having "<" or ","
            X.columns = self.clean_feat_names

        stratify = None if isinstance(self, XGBRegressor) else y
        the_X_train, the_X_val, the_y_train, the_y_val = train_test_split(
            X, y,
            random_state=self.random_state,
            test_size=self.holdout_split,
            stratify=stratify)

        eval_metric = 'logloss' if isinstance(self, XGBClassifier) else 'rmse'

        return super(MyXGBMixin, self).fit(the_X_train, the_y_train, eval_set=[(the_X_val, the_y_val)], eval_metric=eval_metric,
            early_stopping_rounds=self.early_stopping_rounds, verbose=verbose, **kwargs)

    @property
    def param_distributions(self):
        if isinstance(self, XGBClassifier):
            return {
                'learning_rate': [0.2, 0.1, 0.05],
                'subsample': [1., 0.9, 0.8, 0.6],
                'min_child_weight': [0, 1, 2, 5, 10, 20, 50],
                'colsample_bytree': [1., 0.9, 0.8, 0.6],
            }
        else:
            return {
                'learning_rate': [1, 0.5, 0.1],
                'subsample': [1., 0.9, 0.8, 0.6],
                'min_child_weight': [0, 1, 2, 5, 10, 20, 50],
                'colsample_bytree': [1., 0.9, 0.8, 0.6],
            }



class MyXGBClassifierBase(MyGAMPlotMixinBase, MyXGBMixin, XGBClassifier):
    @property
    def is_GAM(self):
        return self.max_depth == 1

    def predict_proba(self, data, ntree_limit=None, validate_features=True):
        # It can not accept feature names having "<" or ","
        if isinstance(data, pd.DataFrame) and hasattr(self, 'clean_feat_names') and self.clean_feat_names is not None:
            data.columns = self.clean_feat_names
        return super().predict_proba(data, ntree_limit=ntree_limit, validate_features=False)


class MyXGBRegressorBase(MyGAMPlotMixinBase, MyXGBMixin, XGBRegressor):
    @property
    def is_GAM(self):
        return self.max_depth == 1

    def predict(self, data, output_margin=False, ntree_limit=None, validate_features=True):
        if isinstance(data, pd.DataFrame) and hasattr(self, 'clean_feat_names') and self.clean_feat_names is not None:
            data.columns = self.clean_feat_names
        return super().predict(data, output_margin=output_margin, ntree_limit=ntree_limit, validate_features=False)


class MyXGBClassifier(OnehotEncodingClassifierMixin, MyXGBClassifierBase):
    def get_params(self, deep=True):
        ''' 
        A hack to make it work through the XGB code. They use the base class 0 to retrieve the parameters.
        Since I overwrite the base_class[0] as OnehotEncodingClassifierMixin, now I do a hack to temporarily
        assign the base class as the next one (XGB class).
        '''
        orig_bases = copy.deepcopy(self.__class__.__bases__)
        self.__class__.__bases__ = (XGBClassifier,)
        self.__class__ = XGBClassifier

        params = XGBClassifier.get_params(self, deep=deep)
        self.__class__ = MyXGBClassifier
        self.__class__.__bases__ = orig_bases
        return params


class MyXGBRegressor(OnehotEncodingRegressorMixin, MyXGBRegressorBase):
    def get_params(self, deep=True):
        ''' 
        A hack to make it work through the XGB code. They use the base class 0 to retrieve the parameters.
        Since I overwrite the base_class[0] as OnehotEncodingClassifierMixin, now I do a hack to temporarily
        assign the base class as the next one (XGB class).
        '''
        orig_bases = copy.deepcopy(self.__class__.__bases__)
        self.__class__.__bases__ = (XGBRegressor,)
        self.__class__ = XGBRegressor

        params = XGBRegressor.get_params(self, deep=deep)
        self.__class__ = MyXGBRegressor
        self.__class__.__bases__ = orig_bases
        return params

class MyXGBLabelEncodingClassifier(LabelEncodingClassifierMixin, MyXGBClassifierBase):
    def get_params(self, deep=True):
        ''' 
        A hack to make it work through the XGB code. They use the base class 0 to retrieve the parameters.
        Since I overwrite the base_class[0] as OnehotEncodingClassifierMixin, now I do a hack to temporarily
        assign the base class as the next one (XGB class).
        '''
        orig_bases = copy.deepcopy(self.__class__.__bases__)
        self.__class__.__bases__ = (XGBClassifier,)
        self.__class__ = XGBClassifier

        params = XGBClassifier.get_params(self, deep=deep)
        self.__class__ = MyXGBClassifier
        self.__class__.__bases__ = orig_bases
        return params

class MyXGBLabelEncodingRegressor(LabelEncodingRegressorMixin, MyXGBRegressorBase):
    def get_params(self, deep=True):
        ''' 
        A hack to make it work through the XGB code. They use the base class 0 to retrieve the parameters.
        Since I overwrite the base_class[0] as OnehotEncodingClassifierMixin, now I do a hack to temporarily
        assign the base class as the next one (XGB class).
        '''
        orig_bases = copy.deepcopy(self.__class__.__bases__)
        self.__class__.__bases__ = (XGBRegressor,)
        self.__class__ = XGBRegressor

        params = XGBRegressor.get_params(self, deep=deep)
        self.__class__ = MyXGBRegressor
        self.__class__.__bases__ = orig_bases
        return params
