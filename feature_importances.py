import sys
sys.path.insert(0, './my_interpret/python')

import numpy as np
import argparse
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit

from loaddata_utils import load_data, load_train_test_data
from sklearn.metrics import roc_auc_score, average_precision_score
import time, os
from collections import OrderedDict
from general_utils import output_csv, Timer, vector_in
import pandas as pd
from arch.utils import predict_score_with_each_feature, sigmoid
from models_utils import mypickle_load


class RemovalExp(object):
    def __init__(self, X_selection, y_selection, problem, model, metric, n_features_limit=None, X_report=None, y_report=None):
        super().__init__()
        self.problem = problem
        self.model = model
        self.metric = metric
        self.feature_names = X_selection.columns

        self.n_features_limit = n_features_limit
        if self.n_features_limit is None:
            self.n_features_limit = X_selection.shape[1]

        # Process X to get the score per feature [N, D]
        self.selection_score_per_feat = predict_score_with_each_feature(self.model, X_selection).values
        self.y_selection = y_selection

        self.report_score_per_feat = self.selection_score_per_feat
        self.y_report = y_selection
        if X_report is not None:
            self.report_score_per_feat = predict_score_with_each_feature(self.model, X_report).values
            self.y_report = y_report

    def run_exp(self):
        # Selection phase
        cur_features = list(range(len(self.feature_names)))
        selected_feat_idxes = []

        cur_scores = self.initialize_scores(self.selection_score_per_feat)
        for _ in range(self.n_features_limit):
            cur_scores, _, best_feat = self.select_feat(cur_scores, cur_features)
            selected_feat_idxes.append(best_feat)

            cur_features.remove(best_feat)
        
        # Metric phase
        cur_scores = self.initialize_scores(self.report_score_per_feat)
        initial_perf = self.eval_perf(cur_scores, self.y_report)
        selected_perf = []

        for f_idx in selected_feat_idxes:
            cur_scores = self.process_feature(cur_scores, self.report_score_per_feat, f_idx)
            the_perf = self.eval_perf(cur_scores, self.y_report)
            selected_perf.append(the_perf)

        the_result = OrderedDict()
        the_result['initial_perf'] = initial_perf
        the_result['feat_idxes'] = selected_feat_idxes
        the_result['feat_perf'] = selected_perf
        the_result['feat_names'] = [self.feature_names[f_idx] for f_idx in selected_feat_idxes]
        return the_result

    def select_feat(self, cur_scores, cur_features):
        '''
        Do a forward selection of the features.
        '''
        assert len(cur_features) > 0

        best_score, best_perf, best_feat = None, None, None
        for f_idx in cur_features:
            the_score = self.process_feature(cur_scores, self.selection_score_per_feat, f_idx)
            the_perf = self.eval_perf(the_score, self.y_selection)

            if best_perf is None:
                best_perf, best_score, best_feat = the_perf, the_score, f_idx
                continue

            if self.better(the_perf, best_perf):
                best_perf, best_score, best_feat = the_perf, the_score, f_idx
        return best_score, best_perf, best_feat

    def get_scores_per_feat(self):
        if self.data_mode not in self.cache_scores_per_feat:
            the_X = getattr(self, 'X_%s' % self.data_mode)
            self.cache_scores_per_feat[self.data_mode] = predict_score_with_each_feature(self.model, the_X).values

        return self.cache_scores_per_feat[self.data_mode]

    def eval_perf(self, pred_score, y):
        if self.problem == 'classification':
            pred_score = sigmoid(pred_score)

        if self.metric == 'auc':
            return roc_auc_score(y, pred_score)
        elif self.metric == 'aupr':
            return average_precision_score(y, pred_score)
        elif self.metric == 'acc':
            return np.mean(np.round(pred_score) == y)
        elif self.metric == 'logloss':
            eps = np.finfo(pred_score.dtype).eps
            p = np.clip(pred_score, eps, 1. - eps)
            return np.mean(y * -np.log(p) + (1. - y) * (-np.log(1. - p)))
        elif self.metric == 'mse':
            return ((y - pred_score) ** 2).mean()
        else:
            raise RuntimeError('Metric is not defined. %s' % self.metric)

    def better(self, left, right):
        ''' In removal experiment, reducing the auc the most (small better) or increases the MSE the most (big better) '''
        is_higher_better = (self.metric in ['auc', 'aupr', 'acc'])
        if is_higher_better:
            return left < right
        else:
            return left > right

    @staticmethod
    def initialize_scores(score_per_feat):
        ''' Sum of the score for each person '''
        return score_per_feat.sum(axis=1)

    @staticmethod
    def process_feature(cur_scores, scores_per_feat, f_idx):
        return cur_scores - scores_per_feat[:, f_idx+1]


class AddExp(RemovalExp):
    @staticmethod
    def initialize_scores(scores_per_feat):
        return scores_per_feat[:, 0]

    @staticmethod
    def process_feature(cur_scores, scores_per_feat, f_idx):
        return cur_scores + scores_per_feat[:, f_idx+1]

    def better(self, left, right):
        ''' In add experiment, increasing the auc the most or decreases the MSE the most '''
        is_higher_better = (self.metric in ['auc', 'aupr', 'acc'])
        if is_higher_better:
            return left > right
        else:
            return left < right


def main():
    if not os.path.exists(args.data_path):
        exit('Exit! Not existing this file %s' % args.data_path)

    # Read into the inputs
    records_df = pd.read_csv(args.data_path)
    if args.model_name is not None:
        records_df = records_df.loc[vector_in(records_df.model_name, args.model_name)]
    if args.d_name is not None:
        records_df = records_df.loc[vector_in(records_df.d_name, args.d_name)]
    if args.end_splits is not None:
        records_df = records_df.loc[records_df.split_idx < args.end_splits]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    data_path_filename = args.data_path.split('/')[-1].split('.')[0]
    output_path = os.path.join(args.output_dir, '%s-fimp-%s-%s.tsv' % (args.identifier, args.exp_mode, data_path_filename))

    # Check the record if overwrite flag is 1!
    if args.overwrite and os.path.exists(output_path):
        if args.model_name is None and args.d_name is None:
            os.remove(output_path)
        else:
            records_df_index = records_df.set_index(['d_name', 'model_name']).index
            df = pd.read_csv(output_path, sep='\t')

            new_df = pd.DataFrame([
                row for r_idx, row in df.iterrows()
                    if (row.d_name, row.model_name) not in records_df_index
            ])
            if len(new_df) == 0:
                os.remove(output_path)
            else:
                new_df.to_csv(output_path, sep='\t', index=None)

    curr_content_lookup = None
    if not args.overwrite and os.path.exists(output_path):
        curr_content_lookup = pd.read_csv(output_path, sep='\t') \
            .set_index(['d_name', 'model_name', 'split_idx', 'metric']).sort_index()

    print('Total df size: ', len(records_df))
    for d_name, df in records_df.groupby('d_name'):
        dataset = load_data(d_name)

        if dataset['problem'] == 'regression' and args.metric != 'mse':
            print('Regression dataset only uses mse as the metric. Skip dataset %s for metric %s.'
                % (d_name, args.metric))
            continue

        for row_idx, (df_idx, record) in enumerate(df.iterrows()):
            if curr_content_lookup is not None and (d_name, record.model_name, record.split_idx, args.metric) in curr_content_lookup.index:
                print('Found in the record. Skip! "%s %s %d %s"' \
                    % (d_name, record.model_name, record.split_idx, args.metric))
                continue

            model = mypickle_load(record.model_path)
            if not hasattr(model, 'is_GAM') or not model.is_GAM: # model is not a GAM
                continue

            with Timer(
                'handling record dataset %s %s %d with idx %d of total %d (%d)' % (
                    d_name, record.model_name, record.split_idx, row_idx, df.shape[0], df_idx)
            ):
                # Reload the train and test set for that record
                X_train, X_test, y_train, y_test = \
                    load_train_test_data(dataset, record.split_idx, 
                        record.n_splits, record.test_size, record.random_state)
                
                # Record the metadata
                the_result = OrderedDict()
                for k in ['d_name', 'model_name', 'model_path', 'split_idx', 'n_splits', 'test_size', 'random_state']:
                    the_result[k] = record[k]
                the_result['metric'] = args.metric

                for mode_name, X_selection, y_selection, X_report, y_report in [
                    # ('train', X_train, y_train, None, None),
                    # ('test', X_test, y_test, None, None),
                    # ('train_test', X_train, y_train, X_test, y_test),
                    ('test_test', X_test.iloc[:int(X_test.shape[0] / 2)], y_test.iloc[:int(X_test.shape[0] / 2)], 
                        X_test.iloc[int(X_test.shape[0] / 2):], y_test.iloc[int(X_test.shape[0] / 2):]),
                ]:
                    exp_obj = args.exp_cls(X_selection, y_selection, dataset['problem'], 
                        model, args.metric, args.n_features_limit, X_report, y_report)

                    exp_result = exp_obj.run_exp()
                    for k in exp_result:
                        the_result['%s_%s' % (mode_name, k)] = exp_result[k]

                output_csv(output_path, the_result, delimiter='\t')


def parse_args():
    parser = argparse.ArgumentParser(description='Training a classifier serving as reward for RL')

    ## General
    parser.add_argument('--identifier', type=str, default='debug3')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--overwrite', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='results/082319_datasets.csv')

    ## Exp Mode
    parser.add_argument('--exp_mode', type=str, default='AddExp', choices=['RemovalExp', 'AddExp'])
    parser.add_argument('--metric', type=str, default='logloss', choices=['mse', 'auc', 'aupr', 'logloss'])
    parser.add_argument('--model_name', nargs='+', type=str, default=None)
    parser.add_argument('--d_name', nargs='+', type=str, default=None)
    parser.add_argument('--n_features_limit', type=int, default=None)
    parser.add_argument('--end_splits', type=int, default=None,
        help='the split index bigger or equal to this will be filtered.'\
        'E.g. if 5, then only split idx 0~4 will be included.'
    )

    ## Which model to run
    args = parser.parse_args()

    args.exp_cls = eval(args.exp_mode)
    return args


if __name__ == "__main__":
    args = parse_args()

    main()
