import numpy as np
import pandas as pd
import os

from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.svm import TimeSeriesSVC
from tslearn.utils import to_time_series_dataset
from sklearn.pipeline import Pipeline

import pickle
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
parser = argparse.ArgumentParser()
parser.add_argument('csv_folder', type=str)
parser.add_argument('--model_save_path', type=str, default='model.pkl')

args = parser.parse_args()
csv_folder = args.csv_folder
model_save_path = args.model_save_path

correct = [
    os.path.join(
        csv_folder,
        'correct',
        file
    )
    for file in os.listdir(f'{csv_folder}/correct')
    if file[-4:] == '.csv'
]

incorrect = [
    os.path.join(
        csv_folder,
        'incorrect',
        file
    )
    for file in os.listdir(f'{csv_folder}/incorrect')
    if file[-4:] == '.csv'
]

X = []
Y = []
for file in correct:
    sequence = pd.read_csv(file, sep=None,
                           header=None,
                           engine='python')

    data = sequence.iloc[:, len(sequence.columns) - 1].values.tolist()
    X.append(data)
    Y.append(0)

for file in incorrect:
    sequence = pd.read_csv(file, sep=None,
                           header=None,
                           engine='python')
    data = sequence.iloc[:, len(sequence.columns) - 1].values.tolist()
    X.append(data)
    Y.append(1)

X = to_time_series_dataset(X)

n_paa_segments = 10
pipeline = Pipeline(
    [
        ('approximate', PiecewiseAggregateApproximation(
            n_segments=n_paa_segments)
         ),
        ('scaler', TimeSeriesScalerMeanVariance(mu=0., std=1.)),
        ('svc', TimeSeriesSVC(
            kernel="gak", probability=True, class_weight='balanced')
         )

    ]
)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
print(np.shape(X_train))
print(np.shape(Y_train))
pipeline = pipeline.fit(X_train, Y_train)

Y_test_pred = pipeline.predict(X_test)
classification_report = classification_report(Y_test, Y_test_pred)
print(classification_report)
# save_directory = os.path.dirname(model_save_path)
# if save_directory:
#     os.makedirs(save_directory, exist_ok=True)
# with open(model_save_path, 'wb') as f:
#     pickle.dump(pipeline, f)
