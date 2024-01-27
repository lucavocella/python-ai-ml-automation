import pandas as pd
import os
from tslearn.utils import to_time_series_dataset

import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('csv_folder', type=str)
parser.add_argument('--model_path', type=str, default='model.pkl')
parser.add_argument('--result_csv', type=str, default='result.csv')

args = parser.parse_args()
csv_folder = args.csv_folder
model_path = args.model_path
result_csv = args.result_csv

samples = [
    os.path.join(
        csv_folder,
        file
    )
    for file in os.listdir(csv_folder)
    if file[-4:] == '.csv'
]

X = []
for file in samples:
    sequence = pd.read_csv(file, sep=None,
                           header=None,
                           engine='python')

    data = sequence.iloc[:, len(sequence.columns) - 1].values.tolist()
    X.append(data)

X = to_time_series_dataset(X)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

Y_predicted = model.predict_proba(X)

result_df = pd.DataFrame(Y_predicted,
                         columns=['normal', 'incorrect_probability'])
result_df['csv name'] = samples
result_df['csv name'] = result_df['csv name'].apply(os.path.basename)
result_df[['csv name', 'incorrect_probability']].to_csv(result_csv, index=False)
