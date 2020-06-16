import sys
sys.path.insert(0, './my_interpret/python')

import pandas as pd
import numpy as np
import pickle
import os
from arch.utils import get_GAM_plot_dataframe_by_models
import argparse
from general_utils import vector_in, Timer
from loaddata_utils import load_data
from models_utils import mypickle_load

def model_generator(model_paths):
    for p in model_paths:
        yield mypickle_load(p)


def main():
    if not os.path.exists(args.data_path):
        exit('Exit! Not existing this file %s' % args.data_path)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    data_path_filename = args.data_path.split('/')[-1].split('.')[0]
    output_dir = os.path.join(args.output_dir, '%s-%s-df' % (args.identifier, data_path_filename))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    records_df = pd.read_csv(args.data_path)
    if args.d_name is None:
        args.d_name = records_df.d_name.unique()
    if args.model_name is None:
        args.model_name = ['gnd_truth'] + records_df.model_name.unique().tolist()

    for d_name in args.d_name:
        output_path = os.path.join(output_dir, '%s.pkl' % (d_name))

        result_dict = {} 
        if os.path.exists(output_path):
            with open(output_path, 'rb') as fp:
                result_dict = pickle.load(fp)

        with Timer(d_name, remove_start_msg=False):
            # Handle ss baseline
            df2 = records_df[records_df.d_name == d_name]
            if len(df2) == 0:
                print('No record found for this dataset %s' % (d_name))
                continue

            dset = load_data(d_name)
            default_x_values_lookup, default_x_counts = {}, {}
            for feat_name in dset['full']['X']:
                X_uni, X_counts = np.unique(dset['full']['X'][feat_name], return_counts=True)
            
                default_x_values_lookup[feat_name] = X_uni
                default_x_counts[feat_name] = X_counts

            # To create importance, we cache the map to get the counts for each feature value
            X_map_dict = {}
            for feat_name in dset['full']['X']:
                X_map = pd.Series(X_counts, X_uni) # map unique value to counts
                X_map_dict[feat_name] = X_map

            for model_name in args.model_name:
                # Handle gnd_truth model class in ss experiments
                if model_name == 'gnd_truth' and d_name.startswith('ss'):
                    if 'gnd_truth' not in result_dict or args.overwrite:
                        gnd_truth_models = mypickle_load(dset['models_path'])

                        result_dict['gnd_truth'] = get_GAM_plot_dataframe_by_models(gnd_truth_models, default_x_values_lookup)

                        # X_values_counts = dset['full']['X'].apply(lambda x: x.value_counts().sort_index().to_dict(), axis=0)
                        # result_dict['gnd_truth']['sample_weights'] = result_dict['gnd_truth'].feat_name.apply(
                        #     lambda x: np.array(list(X_values_counts[x].values()), dtype=np.int) if x != 'offset' else None)
                        result_dict['gnd_truth']['sample_weights'] = result_dict['gnd_truth'].apply(
                            lambda row: None
                                if row.feat_name == 'offset' \
                                else default_x_counts[row.feat_name]
                        , axis=1)

                        result_dict['gnd_truth']['importance'] = result_dict['gnd_truth'].apply(
                            lambda row: -1 
                                if row.feat_name == 'offset' \
                                else np.average(np.abs(row.y), weights=row.sample_weights)
                        , axis=1)

                        pickle.dump(result_dict, open(output_path, 'wb'))
                        print('Finish this %s gnd_truth' % (d_name))
                    else:
                        print('Already finish this %s %s' % (d_name, model_name))
                    continue

                df = df2[df2.model_name == model_name]

                if len(df) == 0:
                    print('No record found for this model %s in dataset %s' % (model_name, d_name))
                    continue

                if not args.overwrite and model_name in result_dict:
                    print('Already finish this %s %s' % (d_name, model_name))
                    continue

                if 'rf' in model_name or 'xgb-d3' in model_name or 'skgbt-d3' in model_name:
                    print(model_name, 'is not a GAM. Skip!')
                    continue

                with Timer('loading %s to check if it is a GAM' % model_name):
                    model = mypickle_load(df.model_path.iloc[0])
                    if not hasattr(model, 'is_GAM') or not model.is_GAM:
                        print(model_name, 'is not a GAM. Skip!')
                        continue

                with Timer(d_name + ' ' + model_name):
                    # use a generator to save memory of loading each model to get df
                    models = model_generator(df.model_path.tolist())
                    result_dict[model_name] = get_GAM_plot_dataframe_by_models(models, default_x_values_lookup)
                    result_dict[model_name]['importance'] = result_dict[model_name].apply(
                        lambda row: -1 
                            if row.feat_name == 'offset' \
                            else np.average(np.abs(row.y), weights=default_x_counts[row.feat_name])
                    , axis=1)

                    pickle.dump(result_dict, open(output_path, 'wb'))


def parse_args():
    parser = argparse.ArgumentParser(description='Training a classifier serving as reward for RL')

    ## General
    parser.add_argument('--identifier', type=str, default='debug3')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--overwrite', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='results/082319_datasets.csv')

    ## Exp Mode
    # parser.add_argument('--exp_mode', type=str, default='RemovalExp', choices=['RemovalExp', 'OneAddATimeExp'])
    parser.add_argument('--model_name', nargs='+', type=str, default=None)
    parser.add_argument('--d_name', nargs='+', type=str, default=None)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()

    main()
