from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import random
import os
from sklearn.utils import shuffle

# returns synthetic and real data, shuffled, alongside labels. Loads up synthetic data,
# but takes in real data according to train, validation, test, split
class Mixed(Dataset):
    def __init__(self, synth_data_dir, real_data, real_labels, transform=None, is_test=False, amount=300):
        self.synth_data_dir = synth_data_dir
        self.real_data = real_data
        self.real_labels = real_labels
        self.transform = transform
        self.is_test = is_test
        self.amount = amount

    def __len__(self):
        return self.amount+len(self.real_labels)

    def __getitem__(self, index):
        synthetic_data = []
        synthetic_label = []
        for file in list(os.listdir(self.synth_data_dir)):
            # print(file)
            img = nib.load(self.synth_data_dir + file)
            data = img.get_fdata()
            synthetic_data.append(data)
            stripped = file.replace('.nii.gz', "")
            label = int(stripped[-1])
            #  print(label)
            synthetic_label.append(label)
            # get labels associated
        # synthetic_data = path + 'tests/'  # load up all .nii.gz and convert to .npy format
        x_sub, y_sub = zip(*random.sample(list(zip(synthetic_data, synthetic_label)), self.amount))
        mixed_data = np.concatenate((x_sub, self.real_data), axis=0)
        mixed_labels = np.concatenate((y_sub, self.real_labels), axis=0)
        shuffled_mixed_data, shuffled_mixed_labels = shuffle(mixed_data, mixed_labels)
        return mixed_data[index], mixed_labels[index]

# for loading just synthetic data
class SyntheticOnly(Dataset):
    def __init__(self, synth_data_dir, transform=None, is_test=False, amount=300):
        self.synth_data_dir = synth_data_dir
        self.transform = transform
        self.is_test = is_test
        self.amount = amount

    def __len__(self):
        return self.amount

    def __getitem__(self, index):
        synthetic_data = []
        synthetic_label = []
        for file in list(os.listdir(self.synth_data_dir)):
            # print(file)
            img = nib.load(self.synth_data_dir + file)
            data = img.get_fdata()
            synthetic_data.append(data)
            stripped = file.replace('.nii.gz', "")
            label = int(stripped[-1])
            synthetic_label.append(label)
        x_sub, y_sub = zip(*random.sample(list(zip(synthetic_data, synthetic_label)), self.amount))
        return x_sub[index], y_sub[index]



class RealTest(Dataset):
    def __init__(self, real_data, real_labels, transform=None, is_test=True):
        self.real_data = real_data
        self.real_labels = real_labels
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.real_labels)

    def __getitem__(self, index):
        return self.real_data[index], self.real_labels[index]



