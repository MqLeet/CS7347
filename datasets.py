#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# TNLI dataset
class SJTUDataSet(Dataset):
    def __init__(self,path_data, is_train=0):

        self.path_data = path_data
        self.is_train = is_train
        self.train_file = os.path.join(self.path_data, "train.tsv")
        self.valid_file = os.path.join(self.path_data, "valid.tsv")
        self.test_file = os.path.join(self.path_data, "test.tsv")

        self.name2Num = {"entailment":0,"neutral":1,"contradiction":2}

        if is_train == 0:
            self.df = pd.read_csv(self.train_file,sep='\t', usecols=['id','sentence1', 'sentence2', 'label'], quoting=csv.QUOTE_NONE)
        elif is_train == 1:
            self.df = pd.read_csv(self.valid_file,sep='\t', usecols=['id','sentence1', 'sentence2', 'label'], quoting=csv.QUOTE_NONE)
        else:
            self.df = pd.read_csv(self.test_file, sep='\t', usecols=['id', 'sentence1', 'sentence2'], quoting=csv.QUOTE_NONE)

        self.df = self.df[self.df["sentence1"].notnull()] #过滤sentence1中的空数据
        self.df = self.df[self.df["sentence2"].notnull()] #过滤sentence2中的空数据
        self.pairs = self.df.values.tolist()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        id, text1, text2 = self.pairs[idx][0], self.pairs[idx][1], self.pairs[idx][2]
        if self.is_train == 0 or self.is_train == 1:
            label = self.name2Num[self.pairs[idx][3]]
            return id, text1, text2, label
        else:
            return id, text1, text2


if __name__ == "__main__":
    path_data = os.path.join(os.getcwd(), "data")
    train_data = SJTUDataSet(path_data=path_data, is_train=True)
    train_loader = DataLoader(dataset=train_data)


    for idx, (id, text1, text2, label) in enumerate(train_loader):
        print(id, text1, text2, label)

