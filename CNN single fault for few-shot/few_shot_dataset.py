# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:52:47 2020

@author: Administrator
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
import scipy.io as sio

class few_shot_dataset(Dataset):
    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, train_test_split, selected_train=True):
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize
        self.selected_train = selected_train
        self.train_test_split = train_test_split
        
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query' % (
        mode, batchsz, n_way, k_shot, k_query))
        
        self.path = os.path.join(root, mode + '.mat')
        self.data = sio.loadmat(self.path)
        self.cls_num = self.data['label'].max()+1
        if selected_train:
            train_data = self.data['data'][0:int(1092*(1)*self.train_test_split),:]
            train_label = self.data['label'][0:int(1092*(1)*self.train_test_split),:]
            for i in range(1, self.cls_num):
                split_data = self.data['data'][1092*i:int(1092*(i+self.train_test_split)),:]
                split_label = self.data['label'][1092*i:int(1092*(i+self.train_test_split)),:]
                train_data = np.concatenate((train_data, split_data), axis = 0)
                train_label = np.concatenate((train_label, split_label), axis = 0)
            self.train_data = train_data
            self.train_label = train_label
        else:
            train_data = self.data['data'][int(1092 * (self.train_test_split)):1092 * (1), :]
            train_label = self.data['label'][int(1092 * (self.train_test_split)):1092 * (1), :]
            for i in range(1, self.cls_num):
                split_data = self.data['data'][int(1092 * (i+self.train_test_split)):1092 * (i + 1), :]
                split_label = self.data['label'][int(1092 * (i+self.train_test_split)):1092 * (i + 1), :]
                train_data = np.concatenate((train_data, split_data), axis=0)
                train_label = np.concatenate((train_label, split_label), axis=0)
            self.train_data = train_data
            self.train_label = train_label

        self.create_batch(self.batchsz)
        
    def create_batch(self, batchsz):
        self.support_x_batch = []  # support set batch
        self.support_y_batch = []
        self.query_x_batch = []  # query set batch
        self.query_y_batch = []
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            support_y = []
            query_x = []
            query_y = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                train_data_size = int(self.train_data.shape[0]/(self.train_label.max()+1))
                selected_data = self.train_data[train_data_size*cls:train_data_size*(cls+1),:]
                selected_label = self.train_label[train_data_size*cls:train_data_size*(cls+1),:]
                selected_imgs_idx = np.random.choice(len(selected_data), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(np.array(selected_data)[indexDtrain].tolist())  # get all images filename for current Dtrain
                support_y.append(np.array(selected_label)[indexDtrain].tolist())
                query_x.append(np.array(selected_data)[indexDtest].tolist())
                query_y.append(np.array(selected_label)[indexDtest].tolist())

#            # shuffle the correponding relation between support set and query set
#            random.shuffle(support_x)
#            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.support_y_batch.append(support_y)
            self.query_x_batch.append(query_x)  # append sets to current sets
            self.query_y_batch.append(query_y)
    
    def __getitem__(self, index):
#        support_x = torch.FloatTensor(self.setsz, 1, self.resize, 1)
        support_x = torch.Tensor(self.support_x_batch[index])
        support_x = support_x.reshape([self.setsz, self.data['data'].shape[1]])
#        support_y = np.zeros((self.setsz), dtype=np.int)
        support_y = torch.Tensor(self.support_y_batch[index])
        support_y = support_y.reshape([self.setsz, self.data['label'].shape[1]])
#        query_x = torch.FloatTensor(self.querysz, 1, self.resize, 1)
        query_x = torch.Tensor(self.query_x_batch[index])
        query_x = query_x.reshape([self.querysz, self.data['data'].shape[1]])
#        query_y = np.zeros((self.querysz), dtype=np.int)
        query_y = torch.Tensor(self.query_y_batch[index])
        query_y = query_y.reshape([self.querysz, self.data['label'].shape[1]])
        
        support_y = support_y.long().squeeze()
        query_y = query_y.long().squeeze()
        
        support_yy = support_y.numpy()
        unique = np.unique(support_yy)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx
        

        return support_x.unsqueeze(1), torch.LongTensor(support_y_relative), query_x.unsqueeze(1), torch.LongTensor(query_y_relative)
        
    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz
        
        
        
        
        
        
        
        
        
        
        
        
        