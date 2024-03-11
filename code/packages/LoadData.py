import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from esm import FastaBatchedDataset
import torch
from torch.nn.utils.rnn import pad_sequence


### DposFinder use dataset###
#############################
#############################
class DposFinderDataset(Dataset):
    def __init__(self, path):
        self.path = path

        self.check_dataset()

    def check_dataset(self):
        tmpdataset = FastaBatchedDataset.from_file(self.path)
        sequence_strs = tmpdataset.sequence_strs
        sequence_labels = tmpdataset.sequence_labels
        self.sequence_strs = np.array(sequence_strs)
        self.sequence_labels = np.array(sequence_labels)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, index):
        labels = self.sequence_labels[index]
        strs = self.sequence_strs[index]
        return labels, strs

### DposFinder train and test dataset###
########################################
########################################
class DepolymeraseDataset(Dataset):

    def __init__(self, path, transform=None, mode='train', kfold=0, fold_num=0, seed=42):
        self.path = path
        self.transform = transform
        self.mode = mode
        self.kfold = kfold
        self.fold_num = fold_num
        self.seed = seed

        self.check_dataset()

    def check_dataset(self):
        tempdataset = FastaBatchedDataset.from_file(self.path)
        sequence_labels = [label[-3:] for label in tempdataset.sequence_labels]
        sequence_strs = tempdataset.sequence_strs

        if self.mode == 'test':
            self.sequence_labels = np.array(sequence_labels)
            self.sequence_strs = np.array(sequence_strs)
        else:
            if self.kfold != 0:
                kf = StratifiedKFold(n_splits=self.kfold,
                                     shuffle=True, random_state=self.seed)
                train_idx, valid_idx = list(kf.split(sequence_strs, sequence_labels))[
                    self.fold_num]
                if self.mode == 'train':
                    self.sequence_labels = np.array(sequence_labels)[train_idx]
                    self.sequence_strs = np.array(sequence_strs)[train_idx]
                else:
                    self.sequence_labels = np.array(sequence_labels)[valid_idx]
                    self.sequence_strs = np.array(sequence_strs)[valid_idx]
            else:
                train_strs, val_strs, train_labels, val_labels = train_test_split(
                    sequence_strs, sequence_labels, test_size=0.25, stratify=sequence_labels, random_state=self.seed)

                if self.mode == 'train':
                    self.sequence_labels = np.array(train_labels)
                    self.sequence_strs = np.array(train_strs)
                else:
                    self.sequence_labels = np.array(val_labels)
                    self.sequence_strs = np.array(val_strs)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, index):
        labels = self.sequence_labels[index]
        strs = self.sequence_strs[index]
        if self.transform is not None:
            labels = self.transform(labels)
        return labels, strs

class BertDataset(Dataset):
    def __init__(self, path, transform=None, mode='train', kfold=0, fold_num=0, seed=42):
        self.path = path
        self.transform = transform
        self.mode = mode
        self.kfold = kfold
        self.fold_num = fold_num
        self.seed = seed

        self.check_dataset()

    def check_dataset(self):
        dict = torch.load(self.path)
        strs = np.array([key for key in dict.keys()])
        sequence_labels = [label[-3:] for label in strs]
        toks = [value for value in dict.values()]

        if self.mode == 'test':
            self.sequence_labels = np.array(sequence_labels)
            self.sequence_strs = strs
            self.toks = toks
        else:
            if self.kfold != 0:
                kf = StratifiedKFold(n_splits=self.kfold,
                                     shuffle=True, random_state=self.seed)
                train_idx, valid_idx = list(kf.split(strs, sequence_labels, toks))[
                    self.fold_num]
                if self.mode == 'train':
                    self.sequence_labels = np.array(sequence_labels)[train_idx]
                    self.sequence_strs = np.array(strs)[train_idx]
                    self.toks = [toks[i] for i in train_idx]
                else:
                    self.sequence_labels = np.array(sequence_labels)[valid_idx]
                    self.sequence_strs = np.array(strs)[valid_idx]
                    self.toks = [toks[i] for i in valid_idx]
            else:
                train_strs, val_strs, train_labels, val_labels = train_test_split(
                    strs, sequence_labels, test_size=0.25, stratify=sequence_labels, random_state=self.seed)

                if self.mode == 'train':
                    self.sequence_labels = np.array(train_labels)
                    self.sequence_strs = np.array(train_strs)
                    self.toks = toks
                else:
                    self.sequence_labels = np.array(val_labels)
                    self.sequence_strs = np.array(val_strs)
                    self.toks = toks

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, index):
        labels = self.sequence_labels[index]
        strs = self.sequence_strs[index]
        if self.transform is not None:
            labels = self.transform(labels)
        toks = self.toks[index]
        return labels, strs, toks
    
def padtoks(batch):
    labels, strs, toks = zip(*batch)

    # Pad the sequences
    toks = pad_sequence(toks, batch_first=True, padding_value=0)

    return labels, strs, toks