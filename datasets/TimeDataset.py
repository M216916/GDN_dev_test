import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config = None, x_non = None, flag=''):
        
        self.raw_data = raw_data
        self.config = config
        self.edge_index = edge_index
        self.mode = mode
        self.x_non = x_non
        self.flag = flag

        x_data = raw_data[:-1]
        labels = raw_data[-1]

        data = x_data

        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        self.x, self.y, self.labels, self.true = self.process(data, labels)

    
    def __len__(self):
        return len(self.x)


    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []
        true_arr = []

        slide_win, slide_stride = [self.config[k] for k in ['slide_win', 'slide_stride']]
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        
        for i in rang:

            ft = data[:, i-slide_win:i]
            tar = data[:, i]
            
            # true 生成
            true = torch.zeros(len(ft))
            line = 0.009
            for j in range(len(ft)):
                rate = (tar[j]-ft[j,-1])/ft[j,-1]
                if rate >= -line:
                    true[j] = 1
                    if rate > line:
                        true[j] = 2
            true = true.to(torch.int64)

            x_arr.append(ft)
            y_arr.append(tar)
            labels_arr.append(labels[i])
            true_arr.append(true)

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()
        true = torch.stack(true_arr).contiguous()
        
        return x, y, labels, true


    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()
        edge_index = self.edge_index.long()
        label = self.labels[idx].double()
        true = self.true[idx]
        x_non = torch.t(torch.tensor(self.x_non.values, dtype=torch.float64))

        if self.flag == 'pre':
            return feature, y, label, edge_index
        
        else:
            return feature, y, label, edge_index, x_non, true