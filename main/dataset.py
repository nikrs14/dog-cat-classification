"""
The custom dataset
"""

import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset

class SpeciesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = self.csv_file[:]['filename']
        self.labels = np.array(self.csv_file.drop(['filename', 'species'], axis = 1))
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, index):
        filename = self.filenames.iloc[index]
        image = cv2.imread(self.root_dir + filename)
        image = self.transform(image)
        targets = 0 if self.csv_file.values[index][1] == 'dog' else 1
        return image, targets
