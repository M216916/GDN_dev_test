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
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def embedded_out(model, train_dataloader, val_dataloader, test_dataloader):

    device = get_device()
    model.eval()
    lgb_train = pd.DataFrame()
    lgb_val   = pd.DataFrame()
    lgb_test  = pd.DataFrame()


    for x, labels, attack_labels, edge_index, x_non, true in train_dataloader:
        x, labels, edge_index, x_non, true = [item.float().to(device) for item in [x, labels, edge_index, x_non, true]]

        x_ave = torch.mean(input=x, dim=2) #
        for i in range(x.shape[2]): #
            x[:,:,i] = x[:,:,i] / x_ave #

        with torch.no_grad():
            _, embedded_out = model(x, edge_index)

        embedded_out = torch.reshape(embedded_out, (embedded_out.shape[0]*embedded_out.shape[1], embedded_out.shape[2]))
        x_non = torch.reshape(x_non, (x_non.shape[0]*x_non.shape[1], x_non.shape[2]))
        true = torch.reshape(true, (true.shape[0]*true.shape[1], 1))

        data_ = torch.cat((embedded_out, x_non, true), dim=1)
        data_ = data_.to('cpu').detach().numpy().copy()
        data_ = pd.DataFrame(data_)
        lgb_train = lgb_train.append(data_)


    for x, labels, attack_labels, edge_index, x_non, true in val_dataloader:
        x, labels, edge_index, x_non, true = [item.float().to(device) for item in [x, labels, edge_index, x_non, true]]

        x_ave = torch.mean(input=x, dim=2) #
        for i in range(x.shape[2]): #
            x[:,:,i] = x[:,:,i] / x_ave #

        with torch.no_grad():
            _, embedded_out = model(x, edge_index)

        embedded_out = torch.reshape(embedded_out, (embedded_out.shape[0]*embedded_out.shape[1], embedded_out.shape[2]))
        x_non = torch.reshape(x_non, (x_non.shape[0]*x_non.shape[1], x_non.shape[2]))
        true = torch.reshape(true, (true.shape[0]*true.shape[1], 1))

        data_ = torch.cat((embedded_out, x_non, true), dim=1)
        data_ = data_.to('cpu').detach().numpy().copy()
        data_ = pd.DataFrame(data_)
        lgb_val = lgb_val.append(data_)


    for x, labels, attack_labels, edge_index, x_non, true in test_dataloader:
        x, labels, edge_index, x_non, true = [item.float().to(device) for item in [x, labels, edge_index, x_non, true]]

        x_ave = torch.mean(input=x, dim=2) #
        for i in range(x.shape[2]): #
            x[:,:,i] = x[:,:,i] / x_ave #

        with torch.no_grad():
            _, embedded_out = model(x, edge_index)

        embedded_out = torch.reshape(embedded_out, (embedded_out.shape[0]*embedded_out.shape[1], embedded_out.shape[2]))
        x_non = torch.reshape(x_non, (x_non.shape[0]*x_non.shape[1], x_non.shape[2]))
        true = torch.reshape(true, (true.shape[0]*true.shape[1], 1))

        data_ = torch.cat((embedded_out, x_non, true), dim=1)
        data_ = data_.to('cpu').detach().numpy().copy()
        data_ = pd.DataFrame(data_)
        lgb_test = lgb_test.append(data_)

    return lgb_train, lgb_val, lgb_test



def lgb_training(lgb_train, lgb_val, lgb_test):

    RANDOM_STATE = 10   # ランダムシード値（擬似乱数）
    TEST_SIZE = 0.2     # 学習データと評価データの割合

    x_train = pd.concat([lgb_train, lgb_val], axis=0).iloc[:,:-1]
    y_train = pd.concat([lgb_train, lgb_val], axis=0).iloc[:,-1]

    x_test = lgb_test.iloc[:,:-1]
    y_test = lgb_test.iloc[:,-1]

    # trainのデータセットの2割をモデル学習時のバリデーションデータとして利用する
    x_train, x_valid, y_train, y_valid = train_test_split(x_train,
                                                        y_train,
                                                        test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE)


    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)        

    params = {'objective' : 'multiclass',
                'metric' : 'multi_logloss',
                'num_class' : 3,
                'learning_rate': 0.3,}

    # LightGBM学習
    evaluation_results = {}
    evals = [(lgb_train, 'train'), (lgb_eval, 'eval')]

    model = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_names = evals,
                    valid_sets = [lgb_train, lgb_eval],
                    early_stopping_rounds=20)

    pred = model.predict(x_test, num_iteration = model.best_iteration)
    label = pred.argmax(axis = 1)

    print("=" * 50)
    matrix = confusion_matrix(y_test, label, labels = [0,1,2])
    print(matrix)
    print("=" * 50)
    accuracy = np.trace(matrix)/np.sum(matrix)
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



def xgb_training(xgb_train, xgb_val, xgb_test):
    
    RANDOM_STATE = 10   # ランダムシード値（擬似乱数）
    TEST_SIZE = 0.2     # 学習データと評価データの割合

    x_train = pd.concat([xgb_train, xgb_val], axis=0).iloc[:,:-1]
    y_train = pd.concat([xgb_train, xgb_val], axis=0).iloc[:,-1]

    x_test = xgb_test.iloc[:,:-1]
    y_test = xgb_test.iloc[:,-1]

    # trainのデータセットの2割をモデル学習時のバリデーションデータとして利用する
    x_train, x_valid, y_train, y_valid = train_test_split(x_train,
                                                        y_train,
                                                        test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE)
    
    xgb_train = xgb.DMatrix(x_train, label=y_train)
    xgb_eval = xgb.DMatrix(x_valid, label=y_valid)
    xgb_test = xgb.DMatrix(x_test, label=y_test)     # , feature_names=feature_name

    params = {
    "objective" : "multi:softmax",
    "eval_metric" : "mlogloss",
    "num_class" : 6,
    "eta" : 0.3,
    "max_depth" : 1}

    evaluation_results = {}
    evals = [(xgb_train, 'train'), (xgb_eval, 'eval')]
    model = xgb.train(params,
                      xgb_train,
                      num_boost_round=500,
                      evals=evals,
                      evals_result=evaluation_results,
                      early_stopping_rounds=10,)

    pred = model.predict(xgb.DMatrix(x_test))

    print("=" * 50)
    matrix = confusion_matrix(y_test, pred ,labels = [0,1,2])
    print(matrix)
    print("=" * 50)
    accuracy = np.trace(matrix)/np.sum(matrix)
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