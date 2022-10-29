"""
Variables that are
used on multiple files
"""

from torchvision import transforms
import os

path = os.path.dirname(os.getcwd())
print(f'{path}')

batch_size = 100

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
