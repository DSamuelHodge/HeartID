'''
Secure Triplet Loss Project Repository (adapted for ECG-ID dataset)
File: ecg_prepare_ecgid.py
- Takes the ECG-ID Database and prepares everything to be used in the 
  Secure Triplet Loss training and experiments.

Adapted from the original UofTDB script by:
"Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics"
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso
IEEE Transactions on Biometrics, Behavior, and Identity Science
'''

import pickle
import numpy as np
import aux_functions as af

N_TRAIN = 100000   # Number of triplets for the train set
N_TEST = 10000     # Number of triplets for the test set
ECGID_PATH = '/path/to/ecgiddb/1.0.0'  # Path to ECG-ID database
SAVE_TRAIN = 'ecg_train_data_ecgid.pickle'
SAVE_TEST = 'ecg_test_data_ecgid.pickle'
fs = 500.0  # Sampling frequency of ECG-ID data

# Set a random seed for reproducibility
np.random.seed(42)


# Get list of all subjects
import os
all_subjects = [folder for folder in os.listdir(ECGID_PATH) if folder.startswith('Person_')]


# Dividing subjects for training and for testing (70% train, 30% test)
np.random.shuffle(all_subjects)
split_index = int(0.7 * len(all_subjects))
train_subjects = all_subjects[:split_index]
test_subjects = all_subjects[split_index:]


# Extracting data for training and testing
train_data = af.extract_data_ecgid(ECGID_PATH, train_subjects, fs=fs)
test_data = af.extract_data_ecgid(ECGID_PATH, test_subjects, fs=fs)


# Preparing data for a deep neural network
X_train_a, y_train_a = af.prepare_for_dnn(train_data['X_anchors'], train_data['y_anchors'])
X_train_r, y_train_r = af.prepare_for_dnn(train_data['X_remaining'], train_data['y_remaining'])
X_test_a, y_test_a = af.prepare_for_dnn(test_data['X_anchors'], test_data['y_anchors'])
X_test_r, y_test_r = af.prepare_for_dnn(test_data['X_remaining'], test_data['y_remaining'])


# Generating triplets
train_triplets = af.generate_triplets(X_train_a, y_train_a, X_train_r, y_train_r, N=N_TRAIN)
test_triplets = af.generate_triplets(X_test_a, y_test_a, X_test_r, y_test_r, N=N_TEST)


# Saving the prepared data
with open(SAVE_TRAIN, 'wb') as handle:
    pickle.dump(train_triplets, handle)


with open(SAVE_TEST, 'wb') as handle:
    pickle.dump(test_triplets, handle)


print(f"Training data saved to {SAVE_TRAIN}")
print(f"Testing data saved to {SAVE_TEST}")
