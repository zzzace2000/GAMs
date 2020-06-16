import sys
sys.path.insert(0, './my_interpret/python')

import argparse
import loaddata_utils
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GroupShuffleSplit
import numpy as np
from models_utils import load_model


def main():
    for d_name in args.d_name:
        dataset = loaddata_utils.load_data(d_name)
        X, y, problem = dataset['full']['X'], dataset['full']['y'], dataset['problem']

        if 'xgb' in args.model_name:
            model = get_xgb_model(args.model_name, problem)


        X, y, problem = dataset['full']['X'], dataset['full']['y'], dataset['problem']

        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.15, random_state=1377)
        # hyperparameteres....
        model = get_ebm_model(model_name, problem, random_state=1377)



        param_grid = [
            {'min_child_weight': [0., 0.5, 1., 2., 3.]},
            {'learning_rate': [0.5, 0.2, 0.1, 0.05]},
            {'reg_lambda': [1.0, 0.1, 0.01, 0.]},
        ]

        # with parallel_backend('threading'):
        cv_model = GridSearchCV(model, param_grid=param_grid, n_jobs=5, scoring='roc_auc', cv=cv, refit=False)
        cv_model.fit(X, y)


def parse_args():
    parser = argparse.ArgumentParser(description='Training a classifier serving as reward for RL')

    ## General
    parser.add_argument('--identifier', type=str, default='debug2')
    parser.add_argument('--random_state', type=int, default=1377)
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--cv_splits', type=int, default=5)
    parser.add_argument('--test_size', type=float, default=0.15)

    ## Which model to run
    parser.add_argument('--model_name', nargs='+', type=str, default=['ebm'])
    parser.add_argument('--d_name', nargs='+', type=str, default=['pneumonia'])

    args = parser.parse_args()

    np.random.seed(args.random_state)
    return args


if __name__ == "__main__":
    args = parse_args()

    main()