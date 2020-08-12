# -*- coding: utf-8 -*-
"""
Created on Sat May  2 21:47:50 2020

@author: Administrator
"""

import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learn_model import learn_model
from    copy import deepcopy
from sklearn.metrics import confusion_matrix, f1_score

class meta(nn.Module):
    def __init__(self, args, config):
        super(meta, self).__init__()
        
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.pre_class = args.pre_class

        self.net = learn_model(config)
        model_dict = self.net.state_dict()  # 取出自己网络的参数
        self.pretrained_dict = torch.load("E:\DL\JiangXinwei\python\训练数据集\pre_model.pth")

        model_dict['vars.0'] = self.pretrained_dict['vars.0']
        model_dict['vars.1'] = self.pretrained_dict['vars.1']
        model_dict['vars.2'] = self.pretrained_dict['vars.2']
        model_dict['vars.3'] = self.pretrained_dict['vars.3']
        model_dict['vars.4'] = self.pretrained_dict['vars.4']
        model_dict['vars.5'] = self.pretrained_dict['vars.5']
        model_dict['vars.6'] = self.pretrained_dict['vars.6']
        model_dict['vars.7'] = self.pretrained_dict['vars.7']
        model_dict['vars.8'] = self.pretrained_dict['vars.8']
        model_dict['vars.9'] = self.pretrained_dict['vars.9']
        model_dict['vars.10'] = self.pretrained_dict['vars.10']
        model_dict['vars.11'] = self.pretrained_dict['vars.11']
        model_dict['vars.12'] = self.pretrained_dict['vars.12']
        model_dict['vars.13'] = self.pretrained_dict['vars.13']
        model_dict['vars.14'] = self.pretrained_dict['vars.14']
        model_dict['vars.15'] = self.pretrained_dict['vars.15']
        model_dict['vars.16'] = self.pretrained_dict['vars.16']
        model_dict['vars.17'] = self.pretrained_dict['vars.17']
        model_dict['vars.18'] = self.pretrained_dict['vars.18']
        model_dict['vars.19'] = self.pretrained_dict['vars.19']
        model_dict['vars.20'] = self.pretrained_dict['vars.20']
        model_dict['vars.21'] = self.pretrained_dict['vars.21']
        model_dict['vars.22'] = self.pretrained_dict['vars.22']
        model_dict['vars.23'] = self.pretrained_dict['vars.23']
        model_dict['vars.24'] = self.pretrained_dict['vars.24']
        model_dict['vars.25'] = self.pretrained_dict['vars.25']
        model_dict['vars.26'] = self.pretrained_dict['vars.26']
        model_dict['vars.27'] = self.pretrained_dict['vars.27']
        model_dict['vars.28'] = self.pretrained_dict['vars.28']
        model_dict['vars.29'] = self.pretrained_dict['vars.29']
        model_dict['vars_bn.0'] = self.pretrained_dict['vars_bn.0']
        model_dict['vars_bn.1'] = self.pretrained_dict['vars_bn.1']
        model_dict['vars_bn.2'] = self.pretrained_dict['vars_bn.2']
        model_dict['vars_bn.3'] = self.pretrained_dict['vars_bn.3']
        model_dict['vars_bn.4'] = self.pretrained_dict['vars_bn.4']
        model_dict['vars_bn.5'] = self.pretrained_dict['vars_bn.5']
        model_dict['vars_bn.6'] = self.pretrained_dict['vars_bn.6']
        model_dict['vars_bn.7'] = self.pretrained_dict['vars_bn.7']
        model_dict['vars_bn.8'] = self.pretrained_dict['vars_bn.8']
        model_dict['vars_bn.9'] = self.pretrained_dict['vars_bn.9']
        model_dict['vars_bn.10'] = self.pretrained_dict['vars_bn.10']
        model_dict['vars_bn.11'] = self.pretrained_dict['vars_bn.11']

        self.net.load_state_dict(model_dict)
        model_dict1 = self.net.state_dict()
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)




        
    def forward(self, x_spt, y_spt, x_qry, y_qry):
        task_num, setsz, w, h = x_spt.size()
        querysz = x_qry.size(1)
        
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], self.net.parameters(), bn_training=True, dropout_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True, dropout_training=False)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

#                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                pred_q = logits_q.argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True, dropout_training=False)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
#                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                pred_q = logits_q.argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True, dropout_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True, dropout_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
#                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    pred_q = logits_q.argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()
        
        accs = np.array(corrects) / (querysz * task_num)

        return accs, losses_q
        
    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        
        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt, dropout_training=True)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        
        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True, dropout_training=False)
            # [setsz]
#                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            pred_q = logits_q.argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True, dropout_training=False)
            # [setsz]
#                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            pred_q = logits_q.argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True, dropout_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True, dropout_training=False)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
#                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                pred_q = logits_q.argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        del net

        accs = np.array(corrects) / querysz

        return accs
    
    def test(self, x_spt, y_spt, x_qry, y_qry):
        querysz = x_qry.size(0)
        with torch.no_grad():
            # [setsz, nway]
            logits_q = self.net(x_qry, bn_training=True, dropout_training=False)
            # [setsz]
#                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            pred_q = logits_q.argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            #confusion_matrix
            y_test_confu = y_qry.cpu().numpy().astype(np.int).squeeze()
            predicted_confu = pred_q.cpu().numpy()
            f_score = f1_score(y_test_confu, predicted_confu, average='macro')
            test_confusion = confusion_matrix(y_test_confu, predicted_confu)
            accr_confusion = self.accuracy(test_confusion, y_test_confu, num_classes=self.pre_class)
            g_mean = np.power(self.accr_confusion_multiply(accr_confusion, num_classes=self.pre_class), 1/self.pre_class)
        
        accs = np.array(correct) / querysz
        return accs, test_confusion, f_score, g_mean
    
    def train(self, x_spt, y_spt, x_qry, y_qry):
        task_num, setsz, w, h = x_spt.size()
        querysz = x_qry.size(1)
        correct = 0
        for i in range(task_num):
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], bn_training=True, dropout_training=False)
                # [setsz]
    #                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                pred_q = logits_q.argmax(dim=1)
                # scalar
                correct = torch.eq(pred_q, y_qry[i]).sum().item()+correct
        accs = np.array(correct) / (querysz * task_num)
        return accs
    
    def accuracy(self, confusion_matrix, true_labels, num_classes):
        list_data = self.count_nums(true_labels, num_classes)
     
        initial_value = 0
        list_length = num_classes
        true_pred = [ initial_value for i in range(list_length)]
        for i in range(0,num_classes-1):
            true_pred[i] = confusion_matrix[i][i]
    
        acc = []
        for i in range(0,num_classes-1):
            acc.append(0)
     
        for i in range(0,num_classes-1):
            acc[i] = true_pred[i] / list_data[i]
     
        return acc
    
    def count_nums(self, true_labels, num_classes):
            initial_value = 0
            list_length = num_classes
            list_data = [ initial_value for i in range(list_length)]
            list_data = np.bincount(true_labels)
            return list_data   
    
    def accr_confusion_multiply(self, accr_confusion,num_classes):
        accr_confusion_multiply = 1
        for i in range(0,num_classes-1):
            accr_confusion_multiply=accr_confusion_multiply*accr_confusion[i]
        return accr_confusion_multiply
            

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    