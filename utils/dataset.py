# -*- coding: utf-8 -*-
#!/usr/bin/python
#****************************************************************#
# ScriptName: utils/data_loader.py
# Author: hq_weng@zju.edu.cn
# Create Date: 2018-10-16 17:39
# Modify Author: hq_weng@zju.edu.cn
# Modify Date: 2018-10-16 17:39
# Function: User defined dataset to load mini batch training data. 
#***************************************************************#
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset, labels, lengths, transform=None):
        self.dataset = dataset
        self.labels = labels
        self.lengths = lengths
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence = self.dataset[idx, :]
        label = self.labels[idx, :]
        length = self.lengths[idx] 
        return sentence, label, [length]
