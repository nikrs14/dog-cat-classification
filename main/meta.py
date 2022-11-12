"""
Two variables that are
used on multiple files
"""

import torch
from torchvision import transforms
import os

path = os.path.dirname(os.getcwd())

batch_size = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
