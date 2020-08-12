# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:55:57 2020

@author: Administrator
"""

import  torch, os
import  numpy as np
from    few_shot_dataset import few_shot_dataset
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from meta import meta
from pre_train import pre_train
import scipy.io as sio


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    # config = [
    #     ('conv1d', [16, 1, 25, 1, 12]),
    #     ('bn', [16]),
    #     ('relu', [True]),
    #     ('max_pool1d', [2]),
    #     ('conv1d', [16, 16, 25, 1, 12]),
    #     ('bn', [16]),
    #     ('relu', [True]),
    #     ('max_pool1d', [2]),
    #     ('flatten', []),
    #     ('linear', [100, 4800]),
    #     ('linear', [args.n_way, 100])
    # ]

    # config_pre = [
    #         ('conv1d', [16, 1, 25, 1, 12]),
    #         ('bn', [16]),
    #         ('relu', [True]),
    #         ('max_pool1d', [2]),
    #         ('conv1d', [16, 16, 25, 1, 12 ]),
    #         ('bn', [16]),
    #         ('relu', [True]),
    #         ('max_pool1d', [2]),
    #         ('flatten', []),
    #         ('linear', [100, 4800]),
    #         ('linear', [args.pre_class, 100])
    #          ]

    config = [
        ('conv1d', [16, 1, 25, 1, 12, 1]),
        ('bn', [16]),
        ('relu', [True]),
        ('conv1d', [16, 16, 25, 1, 12, 2]),
        ('bn', [16]),
        ('relu', [True]),
        ('conv1d', [16, 16, 25, 1, 12, 3]),
        ('concat', []),
        ('bn', [48]),
        ('relu', [True]),
        ('conv1d', [16, 48, 25, 1, 12, 0]),
        ('max_pool1d', [2, 1]),
        ('bn', [16]),
        ('relu', [True]),
        ('conv1d', [16, 16, 25, 1, 12, 2]),
        ('bn', [16]),
        ('relu', [True]),
        ('conv1d', [16, 16, 25, 1, 12, 3]),
        ('concat', []),
        ('bn', [48]),
        ('relu', [True]),
        ('conv1d', [16, 48, 25, 1, 12, 0]),
        ('max_pool1d', [2, 0]),
        ('flatten', []),
        ('linear', [100, 4800]),
        ('linear', [args.n_way, 100])
    ]

    # VOVnet
    config_pre = [
        ('conv1d', [16, 1, 25, 1, 12, 1]),
        ('bn', [16]),
        ('relu', [True]),
        ('conv1d', [16, 16, 25, 1, 12, 2]),
        ('bn', [16]),
        ('relu', [True]),
        ('conv1d', [16, 16, 25, 1, 12, 3]),
        ('concat', []),
        ('bn', [48]),
        ('relu', [True]),
        ('conv1d', [16, 48, 25, 1, 12, 0]),
        ('max_pool1d', [2, 1]),
        ('bn', [16]),
        ('relu', [True]),
        ('conv1d', [16, 16, 25, 1, 12, 2]),
        ('bn', [16]),
        ('relu', [True]),
        ('conv1d', [16, 16, 25, 1, 12, 3]),
        ('concat', []),
        ('bn', [48]),
        ('relu', [True]),
        ('conv1d', [16, 48, 25, 1, 12, 0]),
        ('max_pool1d', [2, 0]),
        ('flatten', []),
        ('linear', [100, 4800]),
        ('linear', [args.pre_class, 100])
    ]

    #pre_train
    pre_model = pre_train(args, config_pre)
    pre_data = sio.loadmat(r'E:\DL\JiangXinwei\python\训练数据集\bearing_cwru_train.mat')
    pre_model(pre_data)


    device = torch.device('cuda')
    maml = meta(args, config).to(device)

    model_dict = maml.state_dict()
    #
    # model_dict['net.vars.0'] = pretrained_dict['vars.0']
    # model_dict['net.vars.1'] = pretrained_dict['vars.1']
    # model_dict['net.vars.2'] = pretrained_dict['vars.2']
    # model_dict['net.vars.3'] = pretrained_dict['vars.3']
    # model_dict['net.vars.4'] = pretrained_dict['vars.4']
    # model_dict['net.vars.5'] = pretrained_dict['vars.5']
    # model_dict['net.vars.6'] = pretrained_dict['vars.6']
    # model_dict['net.vars.7'] = pretrained_dict['vars.7']
    # model_dict['net.vars_bn.0'] = pretrained_dict['vars_bn.0']
    # model_dict['net.vars_bn.1'] = pretrained_dict['vars_bn.1']
    # model_dict['net.vars_bn.2'] = pretrained_dict['vars_bn.2']
    # model_dict['net.vars_bn.3'] = pretrained_dict['vars_bn.3']
    
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
    
    mini = few_shot_dataset('E:\DL\JiangXinwei\python\训练数据集', mode='zhuzhou_bearing_meta_danyi_train', batchsz=args.task, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, resize=1200, train_test_split=args.train_test_radio, selected_train=True)
    mini_test = few_shot_dataset('E:\DL\JiangXinwei\python\训练数据集', mode='zhuzhou_bearing_meta_danyi_train', batchsz=args.task_test, n_way=args.n_way_test, k_shot=args.k_spt_test, k_query=args.k_qry_test, resize=1200, train_test_split=args.train_test_radio, selected_train=False)
    
    for epoch in range(1, args.epoch+1):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=0, pin_memory=True)
        
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)

        if epoch % 50 == 0:
            print('epoch:', epoch, '\ttraining acc:', accs)

        if epoch % 50 == 0:  # evaluation
            db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=0, pin_memory=True)
            accs_all_test = []

            for i in range(10):
                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

            # [b, update_step+1]
            accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)
                

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000)
    argparser.add_argument('--task', type=int, help='task number', default=5)
    argparser.add_argument('--n_way', type=int, help='n way', default=4)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--task_test', type=int, help='task test number', default=1)
    argparser.add_argument('--n_way_test', type=int, help='n way test', default=4)
    argparser.add_argument('--k_spt_test', type=int, help='k shot for support set test', default=1)
    argparser.add_argument('--k_qry_test', type=int, help='k shot for query set test', default=75)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5) #批量训练大小
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=15)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--train_test_radio', type=float, help='train_test_split radio', default=0.7)

    argparser.add_argument('--pre_class', type=int, help='n class', default=4)
    argparser.add_argument('--pre_epoch', type=int, help='pre train epoch number', default=2)
    argparser.add_argument('--pre_batch_size', type=int, help='pre train batch size', default=32)
    argparser.add_argument('--pre_lr', type=float, help='pre train learning rate', default=1e-3)

    args = argparser.parse_args()

    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    