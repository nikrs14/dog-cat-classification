"""
Script for evaluating
a model
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from convnet import ConvNet
from dataset import SpeciesDataset
from meta import batch_size, transform, path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Working on {device}.')

classes = ('dog', 'cat')

test_set =  SpeciesDataset(csv_file = path + '/dataset/data.csv', root_dir = path + '/dataset/set/', transform = transform)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True)

model = ConvNet().to(device)
model.load_state_dict(torch.load(path + '/saves/model01.pth'))

print('\nStarting evaluation...')
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels, names in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct/n_samples
    print(f'\nAccuracy : {acc}%')
