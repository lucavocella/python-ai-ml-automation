import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import convolve1d


parser = argparse.ArgumentParser(
    prog='python create_graphics.py',
    description='Create graphics locally', )

parser.add_argument('csv_folder')
parser.add_argument('--x_column', default=0)
parser.add_argument('--y_column', default='last')
parser.add_argument('--window', default=10)
parser.add_argument('--out', required=True)

args = parser.parse_args()

csv_folder = args.csv_folder
csv_list = os.listdir(csv_folder)
csv_list = list(filter(
    lambda element: element.split('.')[1] == 'csv',
    csv_list
))

x_column = int(args.x_column)
y_column = args.y_column
window = int(args.window)
out_folder = args.out

os.makedirs(out_folder, exist_ok=True)

for csv_file in tqdm(csv_list):

    dataframe = pd.read_csv(
        os.path.join(
            csv_folder,
            csv_file
        ),
        sep=None,
        header=None,
        engine='python'
    )
    if y_column == 'last':
        y_column = len(dataframe.columns) - 1
    else:
        y_column = int(y_column)
    columns = [x_column, y_column]
    dataframe = dataframe.iloc[:, columns]

    dataframe[y_column] = convolve1d(dataframe[y_column],
                                     weights=[1/window] * window,
                                     mode='reflect')
    plt.figure()
    plt.axis('off')

    plt.plot(dataframe.iloc[:, 0], dataframe.iloc[:, 1])
    plt.savefig(
        os.path.join(
            out_folder,
            os.path.basename(csv_file).split('.')[0] + '.png'
        ),
        bbox_inches='tight'
    )
    plt.close()
