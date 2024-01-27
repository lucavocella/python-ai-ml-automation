import torchvision
from torch import nn
import torch
import torch.optim as optim
from model import get_model
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device : ', device)
def train(epochs, loader, model, criterion, optimizer):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if loader.dataset.class_to_idx['incorrect'] == 0:
                labels = 1 - labels
            inputs = torch.Tensor(inputs).to(device)
            labels = torch.Tensor(labels).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = torch.squeeze(model(inputs))
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(
            f'[Epoch: {epoch + 1}]\tloss: {running_loss :.3f}')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_folder', type=str, default='images')
    parser.add_argument('--model_save_path', type=str, default='model.pth')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    images_folder = args.images_folder
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    model_save_path = args.model_save_path

    model = get_model(device=device)

    dataset = torchvision.datasets.ImageFolder(
        images_folder,

        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
        ]
        )
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(epochs, train_loader, model, criterion, optimizer)

    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved in {model_save_path}')
