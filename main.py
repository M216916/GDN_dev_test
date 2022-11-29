import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.preprocessing import MinMaxScaler
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep
from datasets.TimeDataset import TimeDataset
from models.GDN import pre_GDN
from models.GDN import fin_GDN
from train import pre_training
from train import fine_tuning
from test  import pre_test
from test  import fin_test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores
import sys
from datetime import datetime
import os
import argparse
from pathlib import Path
import json
import random
import warnings
warnings.filterwarnings('ignore')


class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset']
        train = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test  = pd.read_csv(f'./data/{dataset}/test.csv',  sep=',', index_col=0)
        true  = pd.read_csv(f'./data/{dataset}/true.csv',  sep=',', index_col=0)
        x_non = pd.read_csv(f'./data/{dataset}/x_non.csv', sep=',', index_col=0)

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.feature_map = feature_map

        cfg = {'slide_win': train_config['slide_win'],
               'slide_stride': train_config['slide_stride'],}


        # pre_training data

        pre_train_dataset_indata = construct_data(train, feature_map, labels=0)
        pre_test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())

        pre_train_dataset = TimeDataset(pre_train_dataset_indata, fc_edge_index, mode='train', config=cfg, x_non=x_non, true=true, flag='pre')   
        pre_test_dataset  = TimeDataset(pre_test_dataset_indata,  fc_edge_index, mode='test',  config=cfg, x_non=x_non, true=true, flag='pre')

        pre_train_dataloader, pre_val_dataloader = self.get_loaders(
            pre_train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.pre_train_dataset = pre_train_dataset
        self.pre_test_dataset = pre_test_dataset
        self.pre_train_dataloader = pre_train_dataloader
        self.pre_val_dataloader = pre_val_dataloader
        self.pre_test_dataloader = DataLoader(pre_test_dataset, batch_size=train_config['batch'], shuffle=False, num_workers=0)


        # fine_tunig data

        fin_train = pd.read_csv(f'./data/{dataset}/test.csv',  sep=',', index_col=0)  # test.csv → 学習用
        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        fin_train_dataset_indata = construct_data(fin_train, feature_map, labels=0)
        
        fin_train_dataset = TimeDataset(fin_train_dataset_indata, fc_edge_index, mode='train', config=cfg, x_non=x_non, true=true, flag='fin')

        fin_train_dataloader, fin_val_dataloader = self.get_loaders(
            fin_train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.fin_train_dataset = fin_train_dataset
        self.fin_test_dataset = None
        self.fin_train_dataloader = fin_train_dataloader
        self.fin_val_dataloader = fin_val_dataloader
        self.fin_test_dataloader = None

        self.pre_model = pre_GDN(edge_index_sets, len(feature_map),
                dim=train_config['dim'],
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']).to(self.device)

        self.fin_model = fin_GDN(edge_index_sets, len(feature_map),
                dim=train_config['dim'],
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']).to(self.device)


    def run(self):

        print('\n▼▼▼ Pre-training : regression')

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']

        else:
            pre_model_save_path = self.pre_get_save_path()[0]   # モデル格納するpath生成

            self.train_log = pre_training(self.pre_model, pre_model_save_path, 
                config = train_config,
                train_dataloader=self.pre_train_dataloader,
                val_dataloader=self.pre_val_dataloader,
                feature_map=self.feature_map,
                test_dataloader=self.pre_test_dataloader,
                test_dataset=self.pre_test_dataset,
                train_dataset=self.pre_train_dataset,
                dataset_name=self.env_config['dataset']
            )

        pre_load_weights = torch.load(pre_model_save_path)

        self.pre_model.load_state_dict(torch.load(pre_model_save_path))
        best_model = self.pre_model.to(self.device)

        _, self.test_result = pre_test(best_model, self.pre_test_dataloader)
        _, self.val_result = pre_test(best_model, self.pre_val_dataloader)

        self.get_figure(self.test_result, self.val_result)


########################################################################################################
        print('\n▼▼▼ Fine-tuning : classifier')

        fin_model_save_path = self.fin_get_save_path()[0]        

        fin_load_weights = self.fin_model.state_dict()
        for i in list(pre_load_weights.keys())[:17]:
            fin_load_weights[i] = pre_load_weights[i]
        self.fin_model.load_state_dict(fin_load_weights)

        fine_tuning(self.fin_model, fin_model_save_path, 
                config = train_config,
                train_dataloader=self.fin_train_dataloader,
                val_dataloader=self.fin_val_dataloader,
                feature_map=self.feature_map,
                test_dataloader=self.fin_test_dataloader,
                test_dataset=self.fin_test_dataset,
                train_dataset=self.fin_train_dataset,
                dataset_name=self.env_config['dataset']
            )
########################################################################################################



    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):

        dataset_len = int(len(train_dataset)) 
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)
        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        return train_dataloader, val_dataloader
    

    def get_figure(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)
        
        GDN_num = os.getcwd().replace('/home/inaba/', '')
        dataset = self.env_config['dataset']

        folder_path = f'/home/inaba/GDN_img/{GDN_num}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        folder_path = f'/home/inaba/GDN_img/{GDN_num}/{dataset}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)        

        for i in range(np_test_result.shape[2]):
            fig = plt.figure()
            plt.plot(np_test_result[0,:,i], label='Prediction')
            plt.plot(np_test_result[1,:,i], label='GroundTruth')
            plt.legend()
            plt.show()
            fig.savefig(folder_path + "img_" + str(i) + ".png")


    def pre_get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [f'./pretrained/{dir_path}/best_{datestr}.pt',
                 f'./results/{dir_path}/{datestr}.csv',]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


    def fin_get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [f'./finetuning/{dir_path}/best_{datestr}.pt',
                 f'./results/{dir_path}/{datestr}.csv',]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='wadi')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=20)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }
    

    main = Main(train_config, env_config, debug=False)
    main.run()