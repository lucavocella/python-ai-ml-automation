import requests
import os
import pandas as pd
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
    prog='python call_api.py',
    description='Upload csv files to chart api', )

parser.add_argument('csv_folder')
parser.add_argument('--token_file', required=True)
parser.add_argument('--x_column', default=0)
parser.add_argument('--y_column', default='last')

args = parser.parse_args()
token_file = args.token_file
csv_folder = args.csv_folder
x_column = int(args.x_column)
y_column = args.y_column

with open(token_file, 'r') as file:
    token = file.read()

base_url = 'https://api.datawrapper.de/v3'
chart_url = f"{base_url}/charts"

csv_list = os.listdir(csv_folder)

# Create folder
folder_url = f"{base_url}/folders"
folder_name = os.path.dirname(csv_folder)

response = requests.get(
    folder_url,
    headers={
        "accept": "*/*",
        "Authorization": f"Bearer {token}"
    }
)

if response.status_code // 100 != 2:
    print(f"Can't retrieve folder information.")
    print(response.text)
    exit(-1)

folders = json.loads(response.content.decode('utf-8'))['list'][0]['folders']
folders_info = {
    f['name']: f['id'] for f in folders
}

folder_id = folders_info.get(folder_name)

if not folder_id:
    response = requests.post(
        folder_url,
        headers={
            "Authorization": f"Bearer {token}",
            "accept": "*/*",
            "content-type": "application/json"
        },
        json={
            'name': folder_name
        }
    )

    if response.status_code // 100 != 2:
        print(f"Remote folder {folder_name} couldn't be created.")
        print(response.text)

    folder_id = int(json.loads(response.content.decode('utf-8'))['id'])

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
    data_of_interest = dataframe.iloc[:, columns].to_csv(index=False,
                                                         header=False, sep=';')
    chart_info = {
        "title": csv_file.split('.')[0],
        "type": "d3-lines",
        'folderId': folder_id
    }

    response = requests.post(
        chart_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"application/json"
        },
        json=chart_info
    )
    if response.status_code // 100 != 2:
        print(f"Chart for {csv_file} couldn't be created.")
        print(response.text)
        continue
    chart_id = json.loads(response.content.decode('utf-8'))['id']
    data_chart_url = f'{chart_url}/{chart_id}/data'
    response = requests.put(
        data_chart_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"text/csv"
        },
        data=data_of_interest
    )
    if response.status_code // 100 != 2:
        print(f"Data for {csv_file} couldn't be set.")
        print(response.text)
        continue
    response = requests.get(
        data_chart_url,
        headers={
            "Authorization": f"Bearer {token}"
        }
    )
