"""
Script for training a model

IMPORTANT: Make sure to change
the save file to another filename
(modelxx.pth) to avoid
loss of previous models.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from convnet import ConvNet
from dataset import SpeciesDataset
from meta import batch_size, transform, path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Working on {device}.')

num_epochs = 50
learning_rate = 0.001

train_set = SpeciesDataset(csv_file = path + '/dataset/data.csv', root_dir = path + '/dataset/set/', transform = transform)
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

n_total_steps = len(train_loader)

print('\nStarting Training...\n')

for epoch in range(num_epochs):
    for i, (images, labels, names) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Step {i+1}/{n_total_steps}, Loss: {loss.item()}')

print('\nFinished Training')

torch.save(model.state_dict(), path + '/saves/model02.pth')
