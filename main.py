import sys
sys.path.insert(0, './my_interpret/python')

import numpy as np
import argparse
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit

from loaddata_utils import load_data
import models_utils
from sklearn.metrics import roc_auc_score, average_precision_score
import time, os
from collections import OrderedDict
import pickle
from general_utils import output_csv
import pandas as pd

# import traceback, warnings
# _old_warn = warnings.warn
# def warn(*args, **kwargs):
#     tb = traceback.extract_stack()
#     _old_warn(*args, **kwargs)
#     print("".join(traceback.format_list(tb)[:-1]))

# warnings.warn = warn


def eval_perf(model, X, y):
    result = {}
    if hasattr(model, 'predict_proba'):
        result['y_pred'] = model.predict_proba(X)[:, 1]
        result['auc'] = roc_auc_score(y, result['y_pred'])
        result['aupr'] = average_precision_score(y, result['y_pred'])
        result['acc'] = np.mean(np.round(result['y_pred']) == y)

        p = np.clip(result['y_pred'], 1e-12, 1. - 1e-12)
        result['logloss'] = np.mean(y * -np.log(p) + (1. - y) * (-np.log(1. - p)))
    else:
        result['y_pred'] = model.predict(X)
        result['auc'] = 0.
        result['aupr'] = 0.
        result['acc'] = 0.
        result['logloss'] = 0.

    result['mse'] = ((y - result['y_pred']) ** 2).mean()
    return result


def get_model(X_train, y_train, problem, model_name, **kwargs):
    model = models_utils.get_model(X_train, y_train, problem, model_name, random_state=args.random_state, **kwargs)
    
    ## Special case to handle the Rspline model case:
    # To pickle this model, since R object is not picklable, a workaround is to extract the 
    # GAM marginal plot and save those value. To avoid extrapolation myself, I need to fit
    # the whole dataset and save all the response value in the pandas df to pickle the model.
    
    if 'rspline' in model_name:
        model.create_df_from_R_model(dataset['full']['X'])

    return model

def load_or_train_model(model_path, model_fn):
    if not args.overwrite and os.path.exists(model_path):
        print('Exists this model! Load from %s' % model_path)
        try:
            model = models_utils.mypickle_load(model_path)
        except (pickle.UnpicklingError, EOFError) as e:
            print(e)
            print('Ignore this model file.... train the model')
            model = model_fn()
    else:
        model = model_fn()
        # Store the model
        pickle.dump(model, open(model_path, 'wb'))
    return model


def get_bias_var(X_train, y_train, X_test, y_test, problem, d_name, model_name, split_idx, **kwargs):
    all_sub_records = []
    subsample_ss = args.split_cls(n_splits=args.n_subsamples, train_size=args.subsample_ratio,
        test_size=(1. - args.subsample_ratio), random_state=args.random_state)

    args.lam = None # Reset the spline grid search for every splits

    for sub_idx, (subsample_idxes, _) in enumerate(subsample_ss.split(X_train, y_train)):
        print(d_name, model_name, split_idx, sub_idx, end='\r')
        X, y = X_train.iloc[subsample_idxes], y_train.iloc[subsample_idxes]

        model_path = os.path.join(args.output_dir, args.identifier,
            '%s_%s_r%d_%d_%d_%d_%d_%.1f.pkl' % (d_name, model_name, args.random_state, split_idx,
                args.n_splits, sub_idx, args.n_subsamples, args.subsample_ratio))
        model = load_or_train_model(model_path, model_fn=lambda: get_model(X, y, problem, model_name, **kwargs))

        sub_record = OrderedDict()
        test_perf = eval_perf(model, X_test, y_test)
        for k in ['auc', 'aupr', 'mse', 'acc', 'logloss', 'y_pred']:
            sub_record['test_%s' % k] = test_perf[k]

        all_sub_records.append(sub_record)

    sub_df = pd.DataFrame(all_sub_records)

    variance = np.mean(np.var(np.array(sub_df.test_y_pred), axis=0))
    avg_test_y_pred = np.mean(np.array(sub_df.test_y_pred), axis=0)
    bias = np.mean((avg_test_y_pred - y_test.values) ** 2)

    error = np.mean(sub_df.test_mse)
    assert np.isclose(bias + variance, error), 'bias: %f, var: %f, error: %f' % (bias, variance, error)

    record = OrderedDict()
    record['bias'] = bias
    record['variance'] = variance
    record['error_test_mse'] = error
    record['avg_test_auc'] = np.mean(sub_df.test_auc)
    record['avg_test_aupr'] = np.mean(sub_df.test_aupr)
    record['avg_test_acc'] = np.mean(sub_df.test_acc)
    record['avg_test_logloss'] = np.mean(sub_df.test_logloss)
    record['n_subsamples'] = args.n_subsamples
    record['subsample_ratio'] = args.subsample_ratio
    return record


def get_training(X_train, y_train, X_test, y_test, problem, d_name, model_name, split_idx, **kwargs):
    model_path = os.path.join(args.output_dir, args.identifier,
        '%s_%s_r%d_%d_%d.pkl' % (d_name, model_name, args.random_state, split_idx, args.n_splits))
    model = load_or_train_model(model_path, model_fn=lambda: get_model(X_train, y_train, problem, model_name, **kwargs))

    record = OrderedDict()

    test_perf = eval_perf(model, X_test, y_test)
    for k in ['auc', 'aupr', 'mse', 'acc', 'logloss']:
        record['test_%s' % k] = test_perf[k]

    train_perf = eval_perf(model, X_train, y_train)
    for k in ['auc', 'aupr', 'mse', 'acc', 'logloss']:
        record['train_%s' % k] = train_perf[k]

    # Store the model
    record['model_path'] = model_path
    return record


def main():
    if not os.path.exists(os.path.join(args.output_dir, args.identifier)):
        os.mkdir(os.path.join(args.output_dir, args.identifier))

    csv_path = os.path.join('./results/', '%s.csv' % args.identifier)
    if not os.path.exists('./results'):
        os.mkdir('./results/')

    curr_content_lookup = None
    if os.path.exists(csv_path):
        curr_content_lookup = pd.read_csv(csv_path).set_index(['d_name', 'model_name', 'split_idx']).sort_index()

    for d_name in args.d_name:
        global dataset # to make it accessible in the function get_model()
        dataset = load_data(d_name)
        print(d_name)

        # Handle the spline lam parameters. Reset for every datasets
        args.lam = None

        X, y, problem = dataset['full']['X'], dataset['full']['y'], dataset['problem']
        test_size = args.test_size

        args.split_cls = StratifiedShuffleSplit if problem == 'classification' else ShuffleSplit
        train_test_ss = args.split_cls(n_splits=args.n_splits, test_size=test_size, random_state=args.random_state)

        idxes_generator = train_test_ss.split(X, y)
        for split_idx, (train_idx, test_idx) in enumerate(idxes_generator):
            if split_idx < args.start_split:
                continue

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            # print('y_train mean:', np.mean(y_train), 'y_test mean:', np.mean(y_test))
            # print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)

            for model_name in args.model_name:
                if args.check_in_records and curr_content_lookup is not None \
                    and (d_name, model_name, split_idx) in curr_content_lookup.index:
                    print('Found in the record. Skip! "%s %s %d"' % (d_name, model_name, split_idx))
                    continue

                print('Start running "%s %s %d"' % (d_name, model_name, split_idx))
                start_time = time.time()

                additional_model_args = {}
                # Set the range of hyperparameter to search for each dataset to save time
                if model_name.startswith('spline') and 'search_lam' in dataset:
                    additional_model_args['search_lam'] = dataset['search_lam']
                if model_name.startswith('spline') and 'n_splines' in dataset:
                    additional_model_args['n_splines'] = dataset['n_splines']
                if model_name.startswith('rspline') and 'discrete' in dataset:
                    additional_model_args['discrete'] = dataset['discrete']
                if model_name.startswith('rspline') and 'maxk' in dataset:
                    additional_model_args['maxk'] = dataset['maxk']

                exp_mode_fn = eval('get_%s' % args.exp_mode)
                experiment_result = exp_mode_fn(
                    X_train, y_train, X_test, y_test, problem, d_name, model_name, 
                    split_idx, **additional_model_args)
                if experiment_result is None:
                    continue

                record = OrderedDict()
                record['d_name'] = d_name
                record['model_name'] = model_name
                record['split_idx'] = split_idx
                record['n_splits'] = args.n_splits
                record['random_state'] = args.random_state
                record['fit_time'] = float(time.time() - start_time)
                record['test_size'] = test_size

                record.update(experiment_result)

                # Follow the column order
                output_csv(csv_path, record)

                print('finish %s %s %d/%d and %s with %.1fs' % (args.exp_mode, d_name, 
                    split_idx, args.n_splits, model_name, float(time.time() - start_time)))
                import gc; gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description='Training a classifier serving as reward for RL')

    ## General
    parser.add_argument('--identifier', type=str, default='0604_datasets', 
        help='The unique identifier for the model')
    parser.add_argument('--random_state', type=int, default=1377, help='random seed')
    parser.add_argument('--output_dir', type=str, default='models/', 
        help='Model saved directory. Default set as models/')
    parser.add_argument('--overwrite', type=int, default=1, 
        help='if set as 1, then it would remove the previous models with same identifier.'\
             'If 0, then use the stored model to make test prediction.')
    parser.add_argument('--check_in_records', type=int, default=1,
        help='If set as 1, then check if the output csv already has the result. If so, skip.')
    parser.add_argument('--start_split', type=int, default=0)
    parser.add_argument('--n_splits', type=int, default=5, 
        help='Rerun the experiment for this number of splits')
    parser.add_argument('--test_size', type=float, default=0.15, 
        help='How many data to be set as test set')

    ## Exp Mode
    parser.add_argument('--exp_mode', type=str, default='training', 
        choices=['training', 'bias_var'], 
        help='If set as bias_var, then run the bias and varaince experiment.')
    parser.add_argument('--n_subsamples', type=int, default=8, 
        help='For bias and variance experiment only. It sets how many runs for the experiment.')
    parser.add_argument('--subsample_ratio', type=float, default=0.5,
        help='Ratio of the dataset to be used as one run of bias and variance experiment.')

    ## Which model and dataset to run
    parser.add_argument('--model_name', nargs='+', type=str, default=['ebm'])
    parser.add_argument('--d_name', nargs='+', type=str, default=['support2cls2'],
        choices=['adult', 'breast', 'churn', 'compass', 'credit', 'heart', 'pneumonia', 'mimicii', 
            'mimiciii', 'support2cls2'])

    args = parser.parse_args()

    np.random.seed(args.random_state)
    return args


if __name__ == "__main__":
    args = parse_args()

    main()
