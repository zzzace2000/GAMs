import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .utils import Timer, my_interpolate, sigmoid, predict_score_by_df, get_x_values_lookup
from .base import MyFitMixin, MyExtractLogOddsMixin
from .EncodingBase import OnehotEncodingFitMixin

class MyRSplineBase(MyExtractLogOddsMixin, OnehotEncodingFitMixin):
    def __init__(self, family='binomial', random_state=1377, maxk=100, nthreads=30, 
        model_to_use='bam', discrete=True, select=False, **kwargs):
        assert model_to_use in ['gam', 'bam']
        assert not (model_to_use == 'gam' and discrete is True), 'Not supported for discrete GAM'
        
        import rpy2.robjects as ro
        self.family = family
        self.random_state = random_state
        self.maxk = maxk
        self.nthreads = nthreads
        self.model_to_use = model_to_use
        self.discrete = discrete
        self.select = select

        self.GAM_plot_dataframe = None
        self.R_model = None
        self.clean_feature_names = None
        self.onehot_prefix = None

        ro.r('set.seed(%d)' % random_state)
        ro.r.library('mgcv')

    def fit(self, X, y, **kwargs):
        # Do one-hot encoding
        self.cat_columns = X.columns[X.dtypes == object].values.tolist()
        X = pd.get_dummies(X)

        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, Formula

        assert not self.is_fit(), 'Call fit() twice!'

        if self.clean_feature_names is None: # R can not accept wierd symbols as names
            self.clean_feature_names = []
            for name in self.feature_names:
                the_name = name.replace('-', '_').replace(' ', '_').replace('$', '').replace('/', '_')\
                    .replace('>', '_big_').replace('(', '_lq_').replace(')', '_rq_').replace('?', '_ques_')\
                    .replace('.', '_dot_').replace('&', '_and_')
                if the_name.startswith('_'):
                    the_name = 's_%s' % (the_name[1:])

                self.clean_feature_names.append(the_name)                

        # Create the fitting string e.g. 'y~s(age)+s(BUN_level)+gender'
        formula_terms = []
        for feat_name, clean_feat_name in zip(self.feature_names, self.clean_feature_names):
            num_unique_x = len(self.X_values_counts[feat_name])
            if num_unique_x < 2:
                continue
            
            term_str = "%s" % clean_feat_name if num_unique_x == 2 \
                else "s(%s, bs='cr', k=%d)" % (clean_feat_name, min(self.maxk, int(num_unique_x*2/3)))

            formula_terms.append(term_str)

        formula_str = 'y~%s' % ('+'.join(formula_terms))
        print('formula_str:', formula_str)
        formula = Formula(formula_str)

        pandas2ri.activate()

        env = formula.environment
        env['y'] = y
        for feat_name, clean_feat_name in zip(self.feature_names, self.clean_feature_names):
            env[clean_feat_name] = X[feat_name]

        # with Timer('Fitting the R mgcv model'):
        self.R_model = ro.r[self.model_to_use](formula, family=self.family, 
            nthreads=self.nthreads, discrete=self.discrete, select=self.select)
        
        pandas2ri.deactivate()

    def get_GAM_plot_dataframe(self, x_values_lookup=None):
        assert self.is_fit(), 'The fit() has not been called!'
        assert self.GAM_plot_dataframe is not None, 'Call create_df_from_R_model() first!'

        if x_values_lookup is None:
            return self.GAM_plot_dataframe

        x_values_lookup = self.convert_x_values_lookup(x_values_lookup)
        
        # Just take a subset of value to get the feature value
        df = self.GAM_plot_dataframe.copy().set_index('feat_name')
        for feat_idx, feat_name in enumerate(self.feature_names):
            model_feat_val = df.loc[feat_name]
            
            model_xs, passed_xs = model_feat_val.x, np.array(x_values_lookup[feat_name])
            assert len(model_xs) >= len(passed_xs), \
                'The model has %d and passed_in x value has %d num. Could be out of range.' \
                    % (len(model_xs), len(passed_xs))

            if len(model_xs) != len(passed_xs) or np.any(model_xs != passed_xs):
                # The passed value is different!
                y_lookup = pd.Series(model_feat_val.y, model_feat_val.x)
                new_y = y_lookup[passed_xs].values

                df.at[feat_name, 'x'] = passed_xs
                df.at[feat_name, 'y'] = new_y
        
        df = df.reset_index()

        return self.revert_dataframe(df)

    def _my_predict_logodds(self, X):
        ''' Used in the base class MyExtractLogOddsMixin '''
        return self.predict_by_R(X)

    def create_df_from_R_model(self, X):
        x_values_lookup = get_x_values_lookup(X)
        x_values_lookup = self.convert_x_values_lookup(x_values_lookup)

        self.GAM_plot_dataframe = self.revert_dataframe(super().get_GAM_plot_dataframe(x_values_lookup))

        # Clean up R
        import rpy2.robjects as ro
        self.R_model = None
        ro.r('rm(list = ls())') # Remove vars
        ro.r('gc()')
        return

    def is_fit(self):
        return self.GAM_plot_dataframe is not None or self.R_model is not None

    def predict_by_R(self, X):
        ''' Return the score (logodds) for the prediction '''
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.clean_feature_names)
        
        X.columns = self.clean_feature_names

        pandas2ri.activate()
        score = ro.r['predict'](self.R_model, newdata=X)
        pandas2ri.deactivate()

        X.columns = self.feature_names
        return score


class MyRSplineClassifier(MyFitMixin, MyRSplineBase):
    def __init__(self, **kwargs):
        super().__init__(family='binomial', **kwargs)
    
    def predict_proba(self, X):
        assert self.is_fit(), 'The fit() has not been called!'

        if self.GAM_plot_dataframe is None: # Use R lib to predict
            X = self.transform_X_to_fit_model_feats(X)
            logodds = self.predict_by_R(X)
        else:
            logodds = predict_score_by_df(self.GAM_plot_dataframe, X)
        prob = sigmoid(logodds)

        return np.vstack([1. - prob, prob]).T


class MyRSplineRegressor(MyFitMixin, MyRSplineBase):
    def __init__(self, **kwargs):
        super().__init__(family='gaussian', **kwargs)
    
    def predict(self, X):
        assert self.is_fit(), 'The fit() has not been called!'

        if self.GAM_plot_dataframe is None: # Use R lib to predict
            X = self.transform_X_to_fit_model_feats(X)

            score = self.predict_by_R(X)
            return score

        return predict_score_by_df(self.GAM_plot_dataframe, X)
