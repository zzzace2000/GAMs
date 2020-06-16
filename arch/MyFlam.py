import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .utils import Timer, my_interpolate, sigmoid, predict_score, get_x_values_lookup, predict_score_by_df
from .base import MyFitMixin, MyCommonBase
from .EncodingBase import OnehotEncodingClassifierMixin, OnehotEncodingRegressorMixin


class MyFLAMBase(MyCommonBase):
    def __init__(self, family, holdout_split=0.15, random_state=1377, n_lambda=100,
        lambda_min_ratio=1e-4, **kwargs):
        
        import rpy2.robjects as ro
        self.family = family
        self.n_lambda = n_lambda
        self.holdout_split = holdout_split
        self.random_state = random_state
        self.lambda_min_ratio = lambda_min_ratio

        ro.r('set.seed(%d)' % random_state)
        ro.r.library('flam')

    def fit(self, X, y, **kwargs):
        import rpy2.robjects as ro
        import rpy2.robjects.numpy2ri as n2r
        
        # Make everything as numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
            y = y.values

        min_idx, all_lambdas = self._select_lam_by_val(X, y)

        # Create final model by rerunning the whole dataset
        with Timer('Fitting the final model'):
            n2r.activate()
            r_result = ro.r['flam'](X, y, family=self.family, alpha = 1.,
                **{'lambda.seq': all_lambdas})
            n2r.deactivate()

        scores = np.asanyarray(r_result.rx2('theta.hat.list')[min_idx])
        intercept = r_result.rx2('beta0.hat.vec')[min_idx]

        self.GAM_plot_dataframe = self._create_df_from_r_result(X, scores, intercept)
        # TODO: remove this return. Just to debug
        # return r_result
        ro.r('rm(list = ls())') # Remove vars

    def _select_lam_by_val(self, X, y):
        import rpy2.robjects as ro
        import rpy2.robjects.numpy2ri as n2r

        stratify = y if isinstance(self, MyFLAMClassifier) else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            random_state=self.random_state,
            test_size=self.holdout_split,
            stratify=stratify)

        with Timer('Fitting the model'):
            n2r.activate()
            r_result = ro.r['flam'](X_train, y_train, family=self.family, alpha = 1.,
                **{'n.lambda': self.n_lambda, 'lambda.min.ratio': self.lambda_min_ratio})
            n2r.deactivate()

        with Timer('Select model by the validation set'):
            # If use the original R predict function, would be sth like:
            # r_y_prob = ro.r['predict'](r_result, **{'new.x': X_val, 'lambda': lam, 'alpha': 1})
            # But <1> it's slow <2> It uses linear interpolation to handle new value, which we should use step func
            all_lambdas = np.asanyarray(r_result.rx2('all.lambda'))
            min_error, min_idx, min_df = np.inf, None, None
            for lam_idx, lam in enumerate(all_lambdas):
                scores = np.asanyarray(r_result.rx2('theta.hat.list')[lam_idx])
                intercept = r_result.rx2('beta0.hat.vec')[lam_idx]

                self.GAM_plot_dataframe = self._create_df_from_r_result(X_train, scores, intercept)

                error = self.error(X_val, y_val)
                # print(lam_idx, lam, error)
                if error < min_error:
                    min_error, min_idx, min_df = error, lam_idx, self.GAM_plot_dataframe

            print('Min:', min_idx, all_lambdas[min_idx], min_error)
        
        # Clean up the memory in R to avoid crash
        ro.r('rm(list=ls())')
        ro.r('gc()')

        return min_idx, all_lambdas

    def _create_df_from_r_result(self, X, scores, intercept):
        # Create a DF
        results = [{
            'feat_name': 'offset',
            'feat_idx': -1,
            'x': None,
            'y': np.full(1, intercept),
            'importance': -1,
        }]

        for feat_idx, feat_name in enumerate(self.feature_names):
            x, index, counts = np.unique(X[:, feat_idx], return_index=True, return_counts=True)
            y = scores[index, feat_idx]
            importance = np.average(np.abs(y), weights=counts)

            results.append({
                'feat_name': feat_name,
                'feat_idx': feat_idx,
                'x': x,
                'y': y,
                'importance': importance,
            })

        return pd.DataFrame(results)

    def error(self, X, y):
        ''' Return the prediction loss for these X '''
        raise NotImplementedError

    def get_GAM_plot_dataframe(self, x_values_lookup=None):
        ''' To avoid going through the encoder again '''
        return self._get_GAM_plot_dataframe(x_values_lookup)
    
    def _get_GAM_plot_dataframe(self, x_values_lookup=None):
        if x_values_lookup is None:
            return self.GAM_plot_dataframe

        # Interpolate / Extrapolate
        df = self.GAM_plot_dataframe.copy().set_index('feat_name')

        for feat_idx, feat_name in enumerate(self.feature_names):
            record = df.loc[feat_name]
            x, y = record.x, record.y

            # interpolate, since sometimes each split would not have the same unique value of x
            x_val = x_values_lookup[feat_name]
            if len(x_val) != len(x) or np.any(x_val != x):
                # Extrapolate
                x, y = x_val, my_interpolate(x, y, x_val)

            df.at[feat_name, 'x'] = x
            df.at[feat_name, 'y'] = y

        return df.reset_index()


class MyFLAMClassifierBase(MyFitMixin, MyFLAMBase):
    def __init__(self, n_lambda=100, **kwargs):
        super().__init__(family='binomial', n_lambda=n_lambda, **kwargs)

    def predict_proba(self, X):
        x_values_lookup = get_x_values_lookup(X, self.feature_names)
        df = self._get_GAM_plot_dataframe(x_values_lookup)
        logodds = predict_score_by_df(df, X)

        prob = sigmoid(logodds)
        return np.vstack([1. - prob, prob]).T

    def error(self, X, y):
        ''' Return the prediction loss. In this case BCE loss '''
        x_values_lookup = get_x_values_lookup(X, self.feature_names)
        df = self._get_GAM_plot_dataframe(x_values_lookup)
        logodds = predict_score_by_df(df, X)

        y_prob = sigmoid(logodds)
        error = -np.mean(y * np.log(y_prob) + (1. - y) * np.log(1. - y_prob))
        return error

class MyFLAMClassifier(OnehotEncodingClassifierMixin, MyFLAMClassifierBase):
    pass


class MyFLAMRegressorBase(MyFitMixin, MyFLAMBase):
    def __init__(self, n_lambda=100, lambda_min_ratio=1e-4, **kwargs):
        super().__init__(family='gaussian', n_lambda=n_lambda, lambda_min_ratio=lambda_min_ratio, **kwargs)

    def predict(self, X):
        x_values_lookup = get_x_values_lookup(X, self.feature_names)
        df = self._get_GAM_plot_dataframe(x_values_lookup)
        return predict_score_by_df(df, X)

    def error(self, X, y):
        ''' Return the prediction loss. In this case MSE loss '''
        x_values_lookup = get_x_values_lookup(X, self.feature_names)
        df = self._get_GAM_plot_dataframe(x_values_lookup)
        y_pred = predict_score_by_df(df, X)

        error = np.mean((y - y_pred) ** 2)
        return error

class MyFLAMRegressor(OnehotEncodingRegressorMixin, MyFLAMRegressorBase):
    pass
