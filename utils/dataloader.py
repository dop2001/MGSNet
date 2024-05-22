import random

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import csv


class FeatureDataset(Dataset):
    def __init__(self, dataset_path, mode='train', ratio=0.8):
        self.dataset_path = dataset_path
        self.mode = mode
        self.ratio = ratio
        hc, mg = self.readCSV()
        #random.shuffle(hc)
        #random.shuffle(mg)
        # split dataset to train and validation
        hc_split_line = int(len(hc)*ratio)
        mg_split_line = int(len(mg)*ratio)
        if mode == 'train':
            self.data = hc[:hc_split_line] + mg[:mg_split_line]
        else:
            self.data = hc[hc_split_line:] + mg[mg_split_line:]

    def __len__(self):
        return len(self.data) * 50

    def __getitem__(self, idx):
        id = idx//50
        num = idx%50
        data = torch.tensor(self.data[id][num]['data'])
        label = torch.tensor(self.data[id][num]['label'])
        return data, label

    def readCSV(self):
        data, counter = [], 0
        with open(self.dataset_path, 'r') as file:
            reader = csv.reader(file)
            temp = []
            for idx, row in enumerate(reader):
                # the first row of csv file is column's name
                if idx != 0 and len(row) != 0:
                    temp.append({'data': list(map(float, row[1:])), 'label': int(row[0])})
                    counter += 1
                    if counter % 50 == 0:
                        data.append(temp)
                        temp = []

        return data[:10], data[10:]


if __name__ == '__main__':
    dataset = FeatureDataset(dataset_path=r'../datasets/FMCC.csv')
    print(dataset[0])
