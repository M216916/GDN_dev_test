import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')
    return loss


def CE_loss(input, target, Dice_gamma):
#    print(F.cross_entropy(input, target))
    return F.cross_entropy(input, target)


def Dice_loss(input, target, Dice_gamma):

    Dice_score = 0

    _, class_num = torch.unique(target, return_counts=True)
    class_num = torch.zeros(3)
    class_num[0] = 5000 # 1064
    class_num[1] = 5000 # 3206
    class_num[2] = 5000 # 1130

    for i in range(input.shape[0]):
        p = torch.exp(input[i,target[i]]) / (torch.sum(torch.exp(input[i].unsqueeze(0)), 1))
        Dice_score = Dice_score + 1/class_num[target[i]] * (2 * p + Dice_gamma)/(p + 1 + Dice_gamma)

    return 1 - 1/3 * Dice_score


def pre_training(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

    seed = config['seed']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])
    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()

    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['pre_epoch']
    early_stop_win = 15

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index in dataloader:
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

###############################################################################################
            x_ave = torch.mean(input=x, dim=2) #
            for i in range(x.shape[2]): #
                x[:,:,i] = x[:,:,i] / x_ave  #
            optimizer.zero_grad()
            out, _= model(x, edge_index)
            out = out.float().to(device)
            out = out * x_ave #
###############################################################################################

            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()
                
            i += 1

        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                i_epoch, epoch, acu_loss/len(dataloader), acu_loss), flush=True)

        if val_dataloader is not None:
            val_loss, val_result = pre_test(model, val_dataloader)

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:                                                    # ×
            if acu_loss < min_loss :                             # ×
                torch.save(model.state_dict(), save_path)        # ×
                min_loss = acu_loss                              # ×

    return train_loss_list



def fine_tuning(model=None, save_path='', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, train_dataset=None, dataset_name='swat'):

    seed = config['seed']
    Dice_gamma = config['Dice_gamma']
    loss_function = config['loss_function']

    if loss_function=='CE_loss':
        loss_func = CE_loss
    elif loss_function=='Dice_loss':
        loss_func = Dice_loss
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])
    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()

    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['fin_epoch']
    early_stop_win = 15

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        sum_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index, x_non, true in dataloader:
            _start = time.time()

            x, labels, edge_index, x_non, true = [item.float().to(device) for item in [x, labels, edge_index, x_non, true]]

            x_ave = torch.mean(input=x, dim=2) #
            for i in range(x.shape[2]): #
                x[:,:,i] = x[:,:,i] / x_ave  #
                
            optimizer.zero_grad()

            out = model(x, edge_index, x_non)
            out = out.float().to(device)

            true = true.to(torch.int64)
            true = true.view(-1)      

            loss = loss_func(out, true, Dice_gamma)
            
            loss.backward()
            optimizer.step()
            
            train_loss_list.append(loss.item())
            sum_loss += loss.item()
                
            i += 1

        print('epoch ({} / {}) (Loss:{:.8f})'.
            format(i_epoch, epoch, sum_loss/len(dataloader)), flush=True)

        # use val dataset to judge
        if val_dataloader is not None:

            val_loss = fin_test(model, val_dataloader, config, 'val')

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:                                                    # ×
            if sum_loss < min_loss :                             # ×
                torch.save(model.state_dict(), save_path)        # ×
                min_loss = sum_loss                              # ×

    return train_loss_list