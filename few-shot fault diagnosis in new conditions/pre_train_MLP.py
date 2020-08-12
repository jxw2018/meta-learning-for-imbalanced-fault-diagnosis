import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learn_model import learn_model
from    copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score

class pre_train(nn.Module):
    def __init__(self, args, config):
        super(pre_train, self).__init__()

        self.pre_epoch = args.pre_epoch
        self.pre_batch_size = args.pre_batch_size
        self.pre_lr = args.pre_lr
        self.pre_class = args.pre_class

        self.device = torch.device('cuda')

        self.net = learn_model(config).to(self.device)
        self.pre_optim = optim.Adam(self.net.parameters(), lr=self.pre_lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x_train, y_train, x_test, y_test):
        
#        x_train_r = np.zeros((len(x_train), 1200, 1))
#        x_train_r[:, :, 0] = x_train[:, :]
#        x_train_r = x_train_r.transpose(0, 2, 1)
#        x_test_r = np.zeros((len(x_test), 1200, 1))
#        x_test_r[:, :, 0] = x_test[:, :]
#        x_test_r = x_test_r.transpose(0, 2, 1)
#        x_train = torch.Tensor(x_train_r)
#        y_train = torch.Tensor(y_train)
#        x_test = torch.Tensor(x_test_r)
#        y_test = torch.Tensor(y_test)
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.pre_batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=int(y_test.shape[0]/1), shuffle=False)

        for i in range(self.pre_epoch):
            print('\nEpoch: %d'%(i+1))
            self.net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0

            for i, data in enumerate(trainloader, 0):
                length = len(trainloader)
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.pre_optim.zero_grad()

                outputs = self.net(inputs)
                labels = labels.long()
                loss = self.criterion(outputs, labels.squeeze())
                loss.backward()
                self.pre_optim.step()

                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.squeeze()).cpu().sum()
                
            train_acc = correct.item() / total
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (i + 1, (i + 1 + i * length), sum_loss / (i + 1), 100. * correct.item() / total))

            print('testing!')
            with torch.no_grad():
                correct = 0.0
                total = 0.0
                for data in testloader:
                    self.net.eval()
                    x_test, y_test = data
                    x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                    output_test = self.net(x_test)
                    _, predicted = torch.max(output_test.data, 1)
                    total += y_test.size(0)
#                    correct += (predicted == y_test.long().squeeze()).sum()
                    correct += predicted.eq(y_test.long().squeeze()).cpu().sum()
                    
                    #confusion_matrix
                    y_test_confu = y_test.cpu().numpy().astype(np.int).squeeze()
                    predicted_confu = predicted.cpu().numpy()
                    f_score = f1_score(y_test_confu, predicted_confu, average='macro')
                    test_confusion = confusion_matrix(y_test_confu, predicted_confu)
                    accr_confusion = self.accuracy(test_confusion, y_test_confu, num_classes=self.pre_class)
                    g_mean = np.power(self.accr_confusion_multiply(accr_confusion, num_classes=self.pre_class), 1/self.pre_class)
                
                test_acc = correct.item() / total
                print('测试分类准确率为：%.3f%%' % (100 * correct.item() / total))
        torch.save(self.net.state_dict(), 'E:\DL\JiangXinwei\python\训练数据集\pre_model.pth')
        return train_acc, test_acc, test_confusion, f_score, g_mean
        
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