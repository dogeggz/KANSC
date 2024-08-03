# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: EurDataset.py
@Time: 2021/3/31 23:20
"""

# imports for file load, read, write
import os
import pickle

# improts for data manipulation
import numpy as np
import torch
from torch.utils.data import Dataset


# class for dataset
class EurDataset(Dataset):
    # constructor
    def __init__(self, split="train"):
        data_dir = "data/"
        # load data from pickle file
        with open(data_dir + "europarl/{}_data.pkl".format(split), "rb") as f:
            self.data = pickle.load(f)

    # get data from index
    def __getitem__(self, index):
        sents = self.data[index]
        return sents

    # length of the dataset
    def __len__(self):
        return len(self.data)


# function to collate data
# basically, we are padding the datat with zeros and store it in a matrixli ke tensor in torch
def collate_data(batch):
    # get the lenght of the batch
    batch_size = len(batch)
    # get the max length of the batch
    max_len = max(map(lambda x: len(x), batch))  # map data to data length in teh batchy
    sents = np.zeros((batch_size, max_len), dtype=np.int64)
    # srothe batch by length in descendign order
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

    # iterate throgh the batch and put the sorted insents matrix
    # padding the data with zeros if the length is less than matrix_len
    for i, sent in enumerate(sort_by_len):
        length = len(sent)
        sents[i, :length] = sent  # padding the questions

    # return the setns matrix as tensor in torch
    return torch.from_numpy(sents)

