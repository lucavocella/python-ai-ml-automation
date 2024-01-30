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

def add_suffix_to_path(original_path, suffix):
    # Remove trailing slash if present
    original_path = os.path.normpath(original_path)
    new_path = original_path + suffix

    return new_path

def combinate_while_possible(csv_to_combine, list_of_csv_to_combine, out_path, overlap_threshold):
    if not list_of_csv_to_combine:
        return None
    df_to_combine = pd.read_csv(csv_to_combine, sep=';', header=None)
    overlap_list = list(map(lambda x: compute_overlap_percentage(df_to_combine, pd.read_csv(x, sep=';', header=None)), list_of_csv_to_combine))
    min_idx = np.argmin(overlap_list)
    min_overlap = overlap_list[min_idx]
    if min_overlap < overlap_threshold:
        min_overlap_csv = list_of_csv_to_combine[min_idx]
        min_overlap_df = pd.read_csv(min_overlap_csv, sep=';', header=None)
        combined_df = combine_dataframes(df_to_combine, min_overlap_df)
        if 'combination' in csv_to_combine:
            new_path = csv_to_combine
            combinations_dict[new_path].append(min_overlap_csv)
        else:
            new_path = os.path.join(out_path, f'combination_{len(os.listdir(out_path)) + 1}.csv')
            combinations_dict[new_path] = [csv_to_combine, min_overlap_csv]
        csv_to_combine_list_i = get_combination_list(min_overlap_csv,
                                                     list_of_csv_to_combine)
        combined_df.to_csv(new_path, sep=';', index=False, header=None)
        return combinate_while_possible(new_path, csv_to_combine_list_i, out_path, overlap_threshold)
    else:
        return None

parser = argparse.ArgumentParser()
parser.add_argument('csv_folder',  type=str, help='Folder containing CSV files')
parser.add_argument('model_path',  help='Path of model to load')
parser.add_argument('--incorrectness_threshold', type=float, default=0.01)
parser.add_argument('--overlap_threshold', type=float, default=0.1)
parser.add_argument('--min_length', type=int, default=50)
parser.add_argument('--skip_graphs_generation', action='store_true')
parser.add_argument('--skip_correctness_computation', action='store_true')

if __name__ == "__main__":

    args = parser.parse_args()
    csv_folder = args.csv_folder
    graphs_folder = add_suffix_to_path(csv_folder, '_graphs')
    model_path = args.model_path
    combinations_csv_path = add_suffix_to_path(csv_folder, '_combinations_csv')
    combinations_png_path = add_suffix_to_path(csv_folder, '_combinations_png')
    best_combinations_path = add_suffix_to_path(csv_folder, '_best_combinations_png')
    incorrectness_threshold = args.incorrectness_threshold
    overlap_threshold = args.overlap_threshold
    min_length = args.min_length
    skip_graphs_generation = args.skip_graphs_generation
    skip_correctness_computation = args.skip_correctness_computation
    log_folder = add_suffix_to_path(csv_folder, '_logs')
    correctness_log = os.path.join(log_folder, 'correctness.csv')
    combinations_log = os.path.join(log_folder, 'combinations.json')
    combinations_correctness_log = os.path.join(log_folder, 'combinations_correctness.csv')
    if os.path.exists(combinations_csv_path):
        shutil.rmtree(combinations_csv_path)
    if os.path.exists(combinations_png_path):
        shutil.rmtree(combinations_png_path)
    if os.path.exists(best_combinations_path):
        shutil.rmtree(best_combinations_path)
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(combinations_csv_path, exist_ok=True)
    os.makedirs(combinations_png_path, exist_ok=True)
    os.makedirs(best_combinations_path, exist_ok=True)
    csv_list = os.listdir(csv_folder)
    csv_list = [element for element in csv_list if element[-4:] == '.csv']
    if not skip_graphs_generation:
        print('Creating graphics...')
        if os.path.exists(graphs_folder):
            shutil.rmtree(graphs_folder)
        create_graphics(
            csv_list=[os.path.join(csv_folder, element) for element in csv_list],
            x_column=0,
            y_column=4,
            window=10,
            out_folder=graphs_folder
        )
    if not skip_correctness_computation:
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
        combinate_while_possible(csv_to_combine, csv_to_combine_list_i, combinations_csv_path, overlap_threshold)
    json.dump(combinations_dict, open(combinations_log, 'w'), indent=4)
    print('\nCreating graphics...')
    create_graphics(
        csv_list=[os.path.join(combinations_csv_path, file) for file in os.listdir(combinations_csv_path)],
        x_column=0,
        y_column=4,
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

