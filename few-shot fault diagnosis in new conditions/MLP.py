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
from pre_train_MLP import pre_train
import scipy.io as sio
import xlwt

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    config = [
         ('linear', [1200, 1200]),
         ('bn', [1200]),
         ('relu', [True]),
         ('linear', [400, 1200]),
         ('bn', [400]),
         ('relu', [True]),
         ('linear', [100, 400]),
         ('linear', [args.n_way, 100])
     ]
    
    mini = few_shot_dataset('E:\DL\JiangXinwei\python\训练数据集', mode='zhuzhou_bearing_meta_train_new_condition', batchsz=args.task, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, resize=1200, train_test_split=args.train_test_radio, selected_train=True)
    mini_test = few_shot_dataset('E:\DL\JiangXinwei\python\训练数据集', mode='zhuzhou_bearing_meta_test_new_condition', batchsz=args.task_test, n_way=args.n_way_test, k_shot=args.k_spt_test, k_query=args.k_qry_test, resize=1200, train_test_split=args.train_test_radio, selected_train=True)
    db = DataLoader(mini, args.task_num, shuffle=True, num_workers=0, pin_memory=True)
    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=0, pin_memory=True)
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(), y_spt.squeeze(), x_qry.squeeze(), y_qry.squeeze()
        x_spt = x_spt.reshape([args.task_num*4, 1200])
        y_spt = y_spt.reshape([args.task_num*4])
        x_qry = x_qry.reshape([args.task_num*20, 1200])
        y_qry = y_qry.reshape([args.task_num*20])
        x_train = torch.cat((x_spt, x_qry), 0)
        y_train = torch.cat((y_spt, y_qry), 0)
        
    for x_spt_test, y_spt_test, x_qry_test, y_qry_test in db_test:
        x_spt_test, y_spt_test, x_qry_test, y_qry_test = x_spt_test.squeeze(), y_spt_test.squeeze(), x_qry_test.squeeze(), y_qry_test.squeeze()
        
        x_qry_test = x_qry_test.reshape([300, 1200])
        y_qry_test = y_qry_test.reshape([300])
        x_test = x_qry_test
        y_test = y_qry_test

    #train
    train_acc = np.zeros(12)
    test_acc = np.zeros(12)
    for i in range(10):
        model = pre_train(args, config)
    
        train_acc[i], test_acc[i], test_confusion, f_score, g_mean = model(x_train, y_train, x_test, y_test)
    train_acc[10] = np.mean(train_acc[:10])
    train_acc[11] = np.std(train_acc[:10])
    test_acc[10] = np.mean(test_acc[:10])
    test_acc[11] = np.std(test_acc[:10])
    
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet = book.add_sheet('MLP',cell_overwrite_ok=True)   
    for i in range(4):
        for j in range(4):
            sheet.write(i, j, str(test_confusion[i][j]))
    for i in range(12):
        sheet.write(i, 5, train_acc[i])
        sheet.write(i, 6, test_acc[i])
    sheet.write(5, 1, 'f_score')
    sheet.write(5, 2, f_score)
    sheet.write(6, 1, 'g_mean')
    sheet.write(6, 2, g_mean)
    book.save('MLP.xls')

    
                

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=300)
    argparser.add_argument('--task', type=int, help='task number', default=10)
    argparser.add_argument('--n_way', type=int, help='n way', default=4)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--task_test', type=int, help='task test number', default=1)
    argparser.add_argument('--n_way_test', type=int, help='n way test', default=4)
    argparser.add_argument('--k_spt_test', type=int, help='k shot for support set test', default=5)
    argparser.add_argument('--k_qry_test', type=int, help='k shot for query set test', default=75)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=10) #批量训练大小
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.002)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--train_test_radio', type=float, help='train_test_split radio', default=0.7)

    argparser.add_argument('--pre_class', type=int, help='n class', default=4)
    argparser.add_argument('--pre_epoch', type=int, help='pre train epoch number', default=50)
    argparser.add_argument('--pre_batch_size', type=int, help='pre train batch size', default=32)
    argparser.add_argument('--pre_lr', type=float, help='pre train learning rate', default=1e-3)
    argparser.add_argument('--training', type=bool, help='dropout train', default= True)

    args = argparser.parse_args()

    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    