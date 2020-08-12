# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:47:56 2020

@author: Administrator
"""

import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np

#class learn_model(nn.Module):
#    def __init__(self, config):
#        super(learn_model, self).__init__()
#        self.config = config
#        self.conv1 = nn.Sequential(nn.Conv1d(1, 16, kernel_size=25, stride=1, padding=12),
#                                   nn.BatchNorm1d(16),
#                                   nn.ReLU(),
#                                   nn.MaxPool1d(2)
#                                   )
#        self.conv2 = nn.Sequential(nn.Conv1d(16, 16, kernel_size=25, stride=1, padding=12),
#                                   nn.BatchNorm1d(16),
#                                   nn.ReLU(),
#                                   nn.MaxPool1d(2)
#                                   )
#        self.mlp1 = nn.Linear(4800, 100)
#        self.mlp2 = nn.Linear(100, 4)
#        
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.conv2(x)
#        x = self.mlp1(x.view(x.size(0), -1))
#        x = self.mlp2(x)
#        return x
        
class learn_model(nn.Module):
    """

    """

    def __init__(self, config):

        super(learn_model, self).__init__()


        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        
        for i, (name, param) in enumerate(self.config):
            if name is 'conv1d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:3]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                
            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                
            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
                
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool1d', 'max_pool1d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'concat']:
                continue
            else:
                raise NotImplementedError
                
    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        
        for name, param in self.config:
            if name is 'conv1d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv1d(x, w, b, stride=param[3], padding=param[4])
                idx += 2
                if param[5] == 1:
                    x1 = x
                elif param[5] == 2:
                    x2 = x
                elif param[5] == 3:
                    x3 = x

            elif name is 'concat':
                x = torch.cat([x1, x2, x3], 1)
                
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
                
            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
                
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
                
            elif name is 'max_pool1d':
                x = F.max_pool1d(x, param[0])
                if param[1] == 1:
                    x1 = x
                
            else:
                raise NotImplementedError
                
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)


        return x
        
    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
        
        























    