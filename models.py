'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)

File: models.py
- Defines the secure models (with cancellability keys)
  that can be trained with the proposed Secure Triplet Loss. The ECG model is
  based on our prior work [1] and the face model is based on the Inception 
  ResNet [2].

  References:
  [1] JR Pinto and JS Cardoso, "A end-to-end convolutional neural network for
      ECG based biometric authentication", in BTAS 2019.
  [2] C. Szegedy et al., "Inceptionv4, Inception-ResNet and the Impact of
      Residual Connections on Learning", in AAAI 2017.

"Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics"
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso
IEEE Transactions on Biometrics, Behavior, and Identity Science

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import torch
from torch import nn
from torch.nn import functional as F

class SecureECGNetwork(nn.Module):
    # Defines the ECG secure network that processes a
    # single biometric sample and a key. Is used with
    # SecureModel for training with Secure Triplet Loss.

    def __init__(self, dropout_prob=0.5):
        # Defining the structure of the ECG secure network.
        super(SecureECGNetwork, self).__init__()
        self.convnet = nn.Sequential(nn.Conv1d(1, 16, 5),
                                     nn.ReLU(),
                                     nn.MaxPool1d(3, stride=3),
                                     nn.Conv1d(16, 16, 5),
                                     nn.ReLU(),
                                     nn.MaxPool1d(3, stride=3),
                                     nn.Conv1d(16, 32, 5),
                                     nn.ReLU(),
                                     nn.MaxPool1d(3, stride=3),
                                     nn.Conv1d(32, 32, 5),
                                     nn.ReLU(),
                                     nn.MaxPool1d(3, stride=3) )

        self.dropout = nn.Sequential(nn.Dropout(p=dropout_prob))

        self.fc = nn.Sequential(nn.Linear(420, 100),
                                nn.ReLU(),
                                nn.Linear(100, 100),
                                nn.ReLU() )

    def forward(self, x, k):
        # Network's inference routine.
        h = self.convnet(x)
        h = h.view(h.size()[0], -1)
        h = self.dropout(h)
        h = torch.cat((h, k), dim=1)
        output = self.fc(h)
        return output

    def get_embedding(self, x, k):
        # To get a secure embedding (template).
        return self.forward(x, k)


class SecureModel(nn.Module):
    # Defines the model to be trained with the
    # Secure Triplet Loss.

    def __init__(self, network):
        super(SecureModel, self).__init__()
        self.network = network

    def forward(self, xA, xP, xN, k1, k2):
        # Secure triplet inference routine.
        output_a = self.network(xA, k1)
        output_p1 = self.network(xP, k1)
        output_p2 = self.network(xP, k2)
        output_n1 = self.network(xN, k1)
        output_n2 = self.network(xN, k2)
        return output_a, output_p1, output_p2, output_n1, output_n2

    def get_embedding(self, x, k):
        # To get a secure embedding (template).
        return self.network(x, k)
