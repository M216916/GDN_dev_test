import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config = None):
        
        self.raw_data = raw_data             # train : 28 list × 1565 ／ test : 28 list × 2049
        self.config = config                 # { slide_win : 5, slide_stride : 1}
        self.edge_index = edge_index         # tensor[[ 1,  2,  3,  ..., 23, 24, 25], [ 0,  0,  0,  ..., 26, 26, 26]] ... (2, 702)
        self.mode = mode                     # train ／ test

        x_data = raw_data[:-1]               # train : 27 list × 1565 ／ test : 27 list × 2049
        labels = raw_data[-1]                # train : 1565 (0 or 1)  ／ test : 2049  (0.0 or 1.0)

        data = x_data

        # to tensor
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        self.x, self.y, self.labels = self.process(data, labels)
    
    def __len__(self):
        return len(self.x)


    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = [self.config[k] for k
            in ['slide_win', 'slide_stride']
        ]
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        
        for i in rang:

            ft = data[:, i-slide_win:i]
            tar = data[:, i]

            x_arr.append(ft)
            y_arr.append(tar)

            labels_arr.append(labels[i])


        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()
        
        return x, y, labels

    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y, label, edge_index





