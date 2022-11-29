import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config = None, x_non = None, true = None, flag=''):
        
        self.raw_data = raw_data
        self.config = config
        self.edge_index = edge_index
        self.mode = mode
        self.x_non = x_non
        self.true = true
        self.flag = flag

        x_data = raw_data[:-1]
        labels = raw_data[-1]

        data = x_data

        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        self.x, self.y, self.labels = self.process(data, labels)

    
    def __len__(self):
        return len(self.x)


    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = [self.config[k] for k in ['slide_win', 'slide_stride']]
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

        x_non = torch.t(torch.tensor(self.x_non.values, dtype=torch.float64))
        true = torch.tensor(self.true.values, dtype=torch.int64).squeeze()

        if self.flag == 'pre':
            return feature, y, label, edge_index          #, x_non, true
        
        else:
            return feature, y, label, edge_index, x_non, true