import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch
from os import listdir, path

class ImageGenerationDataset(Dataset):
    def __init__(self, directory):
        self.X = None
        self.y = None

        for filename in listdir(directory):
            if filename.endswith('.npz'):
                filepath = path.join(directory, filename)
                data = np.load(filepath)

                features = data['features'] # Features is of shape [b, 5, 512] (Corresponding to EEG recording)
                labels = data['labels'] # Labels is of shape [b, 1, 32, 32].


                # Cut last channel of features.
                features = features[:, 0:4, :]
                # Convert labels to shape [b, 32, 32].
                labels = labels[:, 0, :, :]

                if self.X is None and self.y is None:
                    self.X = features
                    self.y = labels
                else:
                    self.X = np.concat((self.X, features), axis=0)
                    self.y = np.concat((self.y, labels), axis=0)
        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]