import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from util.data import *
from util.preprocess import *


def CE_loss_func(y_pred, y_true):
    return F.cross_entropy(y_pred, y_true)


def pre_test(model, dataloader):

    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]

        with torch.no_grad():
            predicted = model(x, edge_index).float().to(device)
            
            loss = loss_func(predicted, y)
            
            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
        
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    test_predicted_list = t_test_predicted_list.tolist()        
    test_ground_list = t_test_ground_list.tolist()        
    test_labels_list = t_test_labels_list.tolist()      
    
    avg_loss = sum(test_loss_list)/len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]






def fin_test(model, dataloader, config, flag):

    device = get_device()

    test_loss_list = []
    matrix = np.array([[0,0,0],[0,0,0],[0,0,0]])
    now = time.time()

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index, x_non, true in dataloader:
        x, y, labels, edge_index, x_non, true = [item.to(device).float() for item in [x, y, labels, edge_index, x_non, true]] 
                        
        with torch.no_grad():
            out = model(x, edge_index, x_non)
            out = out.float().to(device)

            true = true.to(torch.int64)
            true = true.view(-1)

            if flag != 'val':
                matrix = matrix + confusion_matrix(true, torch.argmax(out, dim=1), labels = [0,1,2])

            CE_loss = CE_loss_func(out, true)
        
        test_loss_list.append(CE_loss.item())
        acu_loss += CE_loss.item()

        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    avg_loss = sum(test_loss_list)/len(test_loss_list)

    if flag == 'val':
        return avg_loss

    else:
        print("=" * 50)
        print(matrix)

        accuracy = np.trace(matrix)/np.sum(matrix)
        print("=" * 50)
        print('accuracy      :{0:4f}'.format(accuracy))
        print("=" * 50)

        precision_ave = 0
        recall_ave = 0
        F1_ave = 0

        for i in range(3):
            precision = matrix[i,i]/np.sum(matrix[:,i])
            precision_ave = precision_ave + precision 
            recall = matrix[i,i]/np.sum(matrix[i,])
            recall_ave = recall_ave + recall
            F1 = (2*precision * recall)/(precision + recall)
            F1_ave = F1_ave + F1
            print('【{0}】precision:{1:4f}  recall:{2:4f}  F1:{3:4f}'.format(i,precision,recall,F1))

        print("=" * 50)
        print(precision_ave/3,recall_ave/3,F1_ave/3)
        print("=" * 50)

        return avg_loss