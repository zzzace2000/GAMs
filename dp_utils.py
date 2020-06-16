import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd


def is_way_smaller(x1, x2, epsilon=1e-12):
    return (x1 + epsilon) < x2


def get_se(the_feat, the_y, the_sw, mimic_fn=LinearRegression):
    if the_sw is not None and the_sw.sum() == 0.:
        # Don't choose this path during dynamic programming
        return np.inf, None

    if the_feat.ndim == 1:
        the_feat = the_feat.reshape(-1, 1)
    reg = mimic_fn().fit(the_feat, the_y, sample_weight=the_sw)
    y_pred = reg.predict(the_feat)

    if the_sw is None:
        the_sw = 1.
    se = (((the_y - y_pred) ** 2) * the_sw).sum()
    return se, reg


def run_dynamic_programming(feat, bin_feat, target, w, k=None, mimic_fn_type='linear'):
    '''
    feat: all the features of x
    bin_feat: the binnarized feature of x (in EBM, at most 256 bins)
    target: the corresponding value of y
    w: sample weighted. Has the same size as (L)
    '''

    L = len(bin_feat)
    if k is None:
        k = len(bin_feat) + 1

    mimic_fn = LinearRegression if mimic_fn_type == 'linear' else DummyRegressor

    ## dynamic programming algorithm
    score_mat = -np.ones((k, L))
    back_mat = -np.ones((k, L), dtype=int)
    pred_mat = np.empty((k, L), dtype=object)

    # Pre-caculate all the possible fitting error: totally l*(l-1) / 2 operations
    # Only use the top half
    pre_cached_error = {}
    for i in range(L):
        pre_cached_error[i] = {}
        for j in range(i+1, L):

            val_s, val_e = bin_feat[i], bin_feat[j]

            criteria = ((val_s < feat) & (feat <= val_e))
            the_feat = feat[criteria]
            the_target = target[criteria[:, 0]]
            the_sw = None if w is None else w[criteria[:, 0]]

            the_se, _ = get_se(the_feat, the_target, the_sw, mimic_fn)

            pre_cached_error[i][j] = the_se


    # initalitargete k=0 for all the entry
    for i in range(L):
        criteria = (feat <= bin_feat[i])
        the_feat = feat[criteria]
        the_target = target[criteria[:, 0]]
        the_sw = None if w is None else w[criteria[:, 0]]

        score_mat[0, i], model = get_se(the_feat, the_target, the_sw, mimic_fn)
        pred_mat[0, i] = None if model is None else model.predict(np.array(bin_feat[:(i+1)]).reshape(-1, 1))

    # intialize when n=0, the error is the same
    score_mat[:, 0], back_mat[:, 0] = score_mat[0, 0], 0
    for j in range(k):
        pred_mat[j, 0] = pred_mat[0, 0]

    # DP
    for idx_k in range(1, k):
        for idx_n in range(idx_k, L):
            # initialitargete as no cut
            the_min_val, the_min_idx, the_min_pred = score_mat[idx_k-1, idx_n], idx_n, None

            for prev_row_idx in range(idx_k-1, idx_n):
                # val_s, val_e = bin_feat[prev_row_idx], bin_feat[idx_n]

                # criteria = ((val_s < feat) & (feat <= val_e))
                # the_feat = feat[criteria]
                # the_target = target[criteria[:, 0]]
                # the_sw = None if w is None else w[criteria[:, 0]]

                # the_se, model = get_se(the_feat, the_target, the_sw, mimic_fn)

                the_se = pre_cached_error[prev_row_idx][idx_n]
                the_se += score_mat[idx_k-1, prev_row_idx]

                if is_way_smaller(the_se, the_min_val):
                    the_min_val, the_min_idx = the_se, prev_row_idx
                    # the_min_pred = None if model is None else model.predict(np.array(bin_feat[prev_row_idx+1:idx_n+1]).reshape(-1, 1))

            score_mat[idx_k, idx_n] = the_min_val
            back_mat[idx_k, idx_n] = the_min_idx

            # TODO: FIX the model prediction problem....
            # Fit actual model and get the prediction

            the_min_pred = None if model is None else model.predict(np.array(bin_feat[prev_row_idx+1:idx_n+1]).reshape(-1, 1))
            pred_mat[idx_k, idx_n] = the_min_pred

        # Stop when there is no more benefit to have more cuts
        if back_mat[idx_k, L-1] == L-1:
            break

    return score_mat[:idx_k], back_mat[:idx_k], pred_mat[:idx_k]


def backtrace(score_mat, back_mat, pred_mat, num_cut_point):
    '''
    Return the seperated indexes and the squared error
    '''
    assert num_cut_point >= 0 and num_cut_point < back_mat.shape[0]

    cut_idxes = []

    idx_k, idx_n = num_cut_point, back_mat.shape[1]-1
    all_preds = []
    while idx_k > 0:
        the_pred = pred_mat[idx_k, idx_n]

#         print(idx_k, idx_n, the_pred.shape)
        if the_pred is not None:
            all_preds.append(the_pred)

        idx_n = back_mat[idx_k, idx_n]
        assert idx_n != -1, str(idx_n)
        cut_idxes.append(idx_n)

        idx_k = idx_k - 1

#     print(idx_k, idx_n, pred_mat[idx_k, idx_n].shape)
    the_pred = pred_mat[idx_k, idx_n]
    if the_pred is not None:
        all_preds.append(the_pred)

    # all_idxes = [0] + [val+1 for val in cut_idxes[::-1]] + [back_mat.shape[1]]
    se = score_mat[num_cut_point, -1]
    return cut_idxes[::-1], se, np.hstack(all_preds[::-1])


def simplify_curve(x, target_y, sample_weight=None, mimic_fn_type='linear'):
    '''
    Simply the target curve.

    Args:
        x: the binned feature values with numpy array of shape (P, 1). E.g. [[18], [19], [20]]
        target_y: the log-odds risk of the x. Numpy array of shape (P) E.g. [0.1, 0.5, 1.0]
        sample_weight: the sample weight for each x value. None means no sample weighting.
        mimic_fn_type: only supports "const" or "linear".
    '''
    assert isinstance(x, np.ndarray) and x.ndim == 2, 'x is not in the right format'
    assert len(x) == len(target_y), 'x and y should have the same length'
    assert mimic_fn_type in ['const', 'linear'], 'mimic fn type should be linear or const. But get %s' % mimic_fn_type

    score_mat, back_mat, pred_mat = run_dynamic_programming(x, x, target_y, sample_weight, k=None, mimic_fn_type=mimic_fn_type)

    results = []

    K = len(score_mat)
    for k in range(0, K):
        cut_points, se, simplified = backtrace(score_mat, back_mat, pred_mat, k)

        result = {}
        result['se'] = se
        result['cut_points'] = cut_points
        result['simplified'] = simplified
        result['k'] = k
        results.append(result)

    return pd.DataFrame(results)
