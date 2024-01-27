import argparse

import torch
import torchvision
from PIL import Image

from model import get_model

import os
import pandas as pd
from tqdm import tqdm


def test_model(image_folder, model_path, result_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_path, device)
    model.eval()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ]
    )
    results = []
    for image_name in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)
        with open(image_path, "rb") as f:
            image = Image.open(f).convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        output = model(image)
        results.append([image_name, f'{torch.sigmoid(output).item():.3f}'])
    dataframe_result = pd.DataFrame(results, columns=[
        'image_name', 'incorrect_probability'])
    dataframe_result.to_csv(result_csv, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_folder', type=str, help='path to image folder')
    parser.add_argument('--model_path', type=str, default='model.pth',
                        help='path to model')
    parser.add_argument('--result_csv',  type=str, default='result.csv', help='path to result csv')
    args = parser.parse_args()
    model_path = args.model_path
    image_folder = args.image_folder
    result_csv = args.result_csv
    test_model(image_folder, model_path, result_csv)
