'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)

File: dataset.py
- Uses torch's Dataset class to import the Secure datasets of ECG
  to be used to train or test the models.

  Info on the dataset files needed:

  SecureECGDataset require a pickle file, e.g. "ecg_data.pickle", which stores 
  a tuple of six numpy arrays. The first three numpy arrays correspond to the 
  anchors, the positives, and the negatives, respectively. The fourth array 
  includes the IDs of the anchors on each triplet. The fifth and sixth arrays 
  correspond to the cancelable keys of each triplet, k1 and k2, respectively.
  See "ecg_prepare_uoftdb.py" for more details.

"Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics"
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso
IEEE Transactions on Biometrics, Behavior, and Identity Science

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import torch
import numpy as np
import pickle as pk
from torch.utils.data import Dataset


class SecureECGDataset(Dataset):
    # Used to load ECG triplet data, including cancellability keys
    # for the SecureTL method.

    def __init__(self, data_path):
        with open(data_path, 'rb') as handle:
            data = pk.load(handle)
        self.xA = data[0]
        self.xP = data[1]
        self.xN = data[2]
        self.k1 = data[4]
        self.k2 = data[5]
        self.sample_shape = (1, 1000)

    def __getitem__(self, index):
        # Load anchor, positive, negative, and the two keys for a given index
        xA = self.xA[index].reshape(self.sample_shape)
        xP = self.xP[index].reshape(self.sample_shape)
        xN = self.xN[index].reshape(self.sample_shape)
        k1 = self.k1[index]
        k2 = self.k2[index]
        return (xA, xP, xN, k1, k2)

    def __len__(self):
        return len(self.xA)
