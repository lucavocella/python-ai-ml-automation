import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from create_graphics import create_graphics
from test_model import test_model

combinations_dict = {}

def compute_overlap_percentage(dataframe_1, dataframe_2):
    elements_1 = dataframe_1.iloc[:, 5].values
    elements_2 = dataframe_2.iloc[:, 5].values
    return (sum(np.isin(elements_1, elements_2)) + sum(
        np.isin(elements_2, elements_1))) / (len(elements_1) + len(elements_2))


def combine_dataframes(dataframe_1, dataframe_2):
    dataframe_result = pd.concat([dataframe_1, dataframe_2], axis='rows')
    dataframe_result = dataframe_result.sort_values(
        by=[dataframe_1.columns[5]],
        axis='rows'
    )
    dataframe_result = dataframe_result.reset_index(drop=True)
    dataframe_result[0] = range(len(dataframe_result))
    dataframe_result[0] = dataframe_result[0].astype(int)

    for i in range(1, len(dataframe_result)):
        dataframe_result.at[i, 4] = dataframe_result.at[i - 1, 4] + dataframe_result.at[i, 3]

    return dataframe_result

def get_combination_list(element, csv_list):
    subset_containing_element = [subset.copy() for subset in
                                 combinations_dict.values() if
                                 element in subset]

    for subset in subset_containing_element:
        subset.remove(element)
    joined_subset = [element]
    for subset in subset_containing_element:
        joined_subset.extend(subset)
    csv_to_combine_list_i = [csv_file for csv_file in csv_list if
                             csv_file not in joined_subset]
    return csv_to_combine_list_i


def combinate_while_possible(csv_to_combine, list_of_csv_to_combine, out_path):
    if not list_of_csv_to_combine:
        return None
    df_to_combine = pd.read_csv(csv_to_combine, sep=';', header=None)
    overlap_list = list(map(lambda x: compute_overlap_percentage(df_to_combine, pd.read_csv(x, sep=';', header=None)), list_of_csv_to_combine))
    min_idx = np.argmin(overlap_list)
    min_overlap = overlap_list[min_idx]
    if min_overlap < 0.1:
        min_overlap_csv = list_of_csv_to_combine[min_idx]
        min_overlap_df = pd.read_csv(min_overlap_csv, sep=';', header=None)
        combined_df = combine_dataframes(df_to_combine, min_overlap_df)
        if 'combination' in csv_to_combine:
            new_path = csv_to_combine
            combinations_dict[new_path].append(min_overlap_csv)
        else:
            new_path = out_path + f'combination_{len(os.listdir(out_path)) + 1}.csv'
            combinations_dict[new_path] = [csv_to_combine, min_overlap_csv]
        csv_to_combine_list_i = get_combination_list(min_overlap_csv,
                                                     list_of_csv_to_combine)
        combined_df.to_csv(new_path, sep=';', index=False, header=None)
        return combinate_while_possible(new_path, csv_to_combine_list_i, out_path)
    else:
        return None

parser = argparse.ArgumentParser()
parser.add_argument('csv_folder',  type=str, help='Folder containing CSV files')
parser.add_argument('graphs_folder',  help='Folder to save graphs')
parser.add_argument('model_path',  help='Path of model to load')
parser.add_argument('combinations_csv_path',  help='Path of output the '
                                                      'combinations')
parser.add_argument('combinations_png_path',  help='Path of output the png')
parser.add_argument('best_combinations_path',  help='Path of output the best combinations')
parser.add_argument('--incorrectness_threshold', type=float, default=0.01)
parser.add_argument('--overlap_threshold', type=float, default=0.1)
parser.add_argument('--min_length', type=int, default=50)
parser.add_argument('--log_folder', type=str, default='logs/')

if __name__ == "__main__":

    args = parser.parse_args()
    csv_folder = args.csv_folder
    graphs_folder = args.graphs_folder
    model_path = args.model_path
    combinations_csv_path = args.combinations_csv_path
    combinations_png_path = args.combinations_png_path
    best_combinations_path = args.best_combinations_path
    incorrectness_threshold = args.incorrectness_threshold
    overlap_threshold = args.overlap_threshold
    min_length = args.min_length
    log_folder = args.log_folder
    correctness_log = os.path.join(log_folder, 'correctness.csv')
    combinations_log = os.path.join(log_folder, 'combinations.json')
    combinations_correctness_log = os.path.join(log_folder, 'combinations_correctness.csv')
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(combinations_csv_path, exist_ok=True)
    os.makedirs(combinations_png_path, exist_ok=True)
    os.makedirs(best_combinations_path, exist_ok=True)
    csv_list = os.listdir(csv_folder)
    print('Creating graphics...')
    create_graphics(
        csv_list=[os.path.join(csv_folder, element) for element in csv_list],
        x_column=0,
        y_column=3,
        window=10,
        out_folder=graphs_folder
    )
    print('Computing Correctness...')
    test_model(graphs_folder, model_path, correctness_log)

    print('Selecting correct csv...')
    df = pd.read_csv(correctness_log)
    correct_aux = (df[df['incorrect_probability'] < 0.5]['image_name'].
                   str.replace('.png', '.csv').values)
    csv_to_combine_list = []
    for element in tqdm(correct_aux):
        element = os.path.join(str(csv_folder), str(element))
        df_aux = pd.read_csv(
            element,
            sep=';',
            header=None
        )
        if len(df_aux) >= min_length:
            csv_to_combine_list.append(element)
    best_samples = (df[df['incorrect_probability'] < incorrectness_threshold]['image_name'].
                    str.replace('.png', '.csv').values)
    best_samples = [os.path.join(str(csv_folder), str(f)) for f in best_samples if
                    len(pd.read_csv(os.path.join(str(csv_folder), str(f)))) >= min_length]
    print('\nCombining best csv files with other corrects')
    for i, csv_to_combine in tqdm(enumerate(best_samples), total=len(best_samples)):
        # get list of csv to combine that was not already combined
        csv_to_combine_list_i = get_combination_list(csv_to_combine, csv_to_combine_list)
        combinate_while_possible(csv_to_combine, csv_to_combine_list_i, combinations_csv_path)
    json.dump(combinations_dict, open(combinations_log, 'w'), indent=4)
    print('\nCreating graphics...')
    create_graphics(
        csv_list=[os.path.join(combinations_csv_path, file) for file in os.listdir(combinations_csv_path)],
        x_column=0,
        y_column=3,
        window=10,
        out_folder=combinations_png_path,
    )
    print('\nComputing Correctness of combinations')
    test_model(combinations_png_path, model_path, combinations_correctness_log)
    df = pd.read_csv(combinations_correctness_log)
    best_samples = (
        df[df['incorrect_probability'] < incorrectness_threshold]['image_name'].values)
    best_samples = [os.path.join(combinations_png_path, sample) for sample in best_samples]
    os.makedirs(best_combinations_path, exist_ok=True)
    print('\nSaving best combinations')
    for sample in tqdm(best_samples):
        shutil.copy(sample, best_combinations_path)

