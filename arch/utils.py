import time
import numpy as np
import pandas as pd
from scipy import interpolate


def get_GAM_plot_dataframe_by_models(models, x_values_lookup=None, aggregate=True):
    models = iter(models)

    first_model = next(models)

    first_df = first_model.get_GAM_plot_dataframe(x_values_lookup)

    is_x_values_lookup_none = x_values_lookup is None
    if is_x_values_lookup_none:
        x_values_lookup = first_df[['feat_name', 'x']].set_index('feat_name').x.to_dict()
    
    all_dfs = [first_df]
    for model in models:
        the_df = model.get_GAM_plot_dataframe(x_values_lookup)
        all_dfs.append(the_df)
    
    if not aggregate:
        return all_dfs
    
    if len(all_dfs) == 1:
        return first_df

    all_ys = [np.concatenate(df.y) for df in all_dfs]

    split_pts = first_df.y.apply(lambda x: len(x)).cumsum()[:-1]
    first_df['y'] = np.split(np.mean(all_ys, axis=0), split_pts)
    first_df['y_std'] = np.split(np.std(all_ys, axis=0), split_pts)
    return first_df


def predict_score(model, X):
    result = predict_score_with_each_feature(model, X)
    return result.values.sum(axis=1)

def predict_score_by_df(GAM_plot_df, X):
    result = predict_score_with_each_feature_by_df(GAM_plot_df, X)
    return result.values.sum(axis=1)

def predict_score_with_each_feature(model, X):
    x_values_lookup = get_x_values_lookup(X, model.feature_names)
    GAM_plot_df = model.get_GAM_plot_dataframe(x_values_lookup)
    return predict_score_with_each_feature_by_df(GAM_plot_df, X)

def predict_score_with_each_feature_by_df(GAM_plot_df, X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=GAM_plot_df.feat_name.iloc[1:].values.tolist())

    df_lookup = GAM_plot_df.set_index('feat_idx')

    offset = 0. if -1 not in df_lookup.index else df_lookup.loc[-1].y
    # scores = np.full(X.shape[0], offset)
    scores = np.empty((X.shape[0], GAM_plot_df.feat_idx.max() + 2))
    scores[:, 0] = offset
    names = ['offset']

    for f_idx in range(X.shape[1]):
        attrs = df_lookup.loc[f_idx]

        score_lookup = pd.Series(attrs.y, index=attrs.x)

        truncated_X = X.iloc[:, f_idx]
        if truncated_X.dtype == object:
            truncated_X = truncated_X.astype('str')

        scores[:, (f_idx+1)] = score_lookup[truncated_X].values
        names.append(attrs.feat_name)

    return pd.DataFrame(scores, columns=names)


def sigmoid(x):
    "Numerically stable sigmoid function."
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def get_X_values_counts(X, feature_names=None):
    if feature_names is None:
        feature_names = ['f%d' % i for i in range(X.shape[1])] if isinstance(X, np.ndarray) else X.columns
    
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)
        # return {'f%d' % idx: dict(zip(*np.unique(X[:, idx], return_counts=True))) for idx in range(X.shape[1])}
        
    return X.apply(lambda x: x.value_counts().sort_index().to_dict(), axis=0)

def get_x_values_lookup(X, feature_names=None):
    if isinstance(X, np.ndarray):
        if feature_names is None:
            feature_names = ['f%d' for idx in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
    else:
        feature_names = X.columns

    return {
        feat_name : np.unique(X.iloc[:, feat_idx]).astype(X.dtypes[feat_idx])
        for feat_idx, feat_name in enumerate(feature_names)
    }

def my_interpolate(x, y, new_x):
    ''' Handle edge cases for interpolation '''
    assert len(x) == len(y)

    if len(x) == 1:
        y = np.full(len(new_x), y[0])
    else:
        f = interpolate.interp1d(x, y, fill_value='extrapolate', kind='nearest')
        y = f(new_x.astype(float))
    return y


class Timer:
    def __init__(self, name, remove_start_msg=True):
        self.name = name
        self.remove_start_msg = remove_start_msg

    def __enter__(self):
        self.start_time = time.time()
        print('Run "%s".........' % self.name, end='\r' if self.remove_start_msg else '\n')

    def __exit__(self, exc_type, exc_val, exc_tb):
        time_diff = float(time.time() - self.start_time)
        time_str = '{:.1f}s'.format(time_diff) if time_diff >= 1 else '{:.0f}ms'.format(time_diff * 1000)

        print('Finish "{}" in {}'.format(self.name, time_str))
