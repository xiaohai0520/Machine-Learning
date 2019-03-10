import re
import pandas as pd
import pickle
import numpy as np
import torch
import pickle
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch import optim
import matplotlib.pyplot as plt

TARGET = ".\\stock_data\\DowJones.csv"
PLOT_PATH = '.\\plot\\'
num_epochs = 300

TIME_STEP = 10
LR = 0.001
batch_size = 128
models = ['RNN','LSTM','GRU']


class DataSet():

    def __init__(self,pricesfile,time_step,split_ratio=0.7):

        #the price file path
        self.pricesfile = pricesfile

        #read from the csv
        price_trend = self.get_data()

        #get input and output
        self.x,self.y = self.create_dataset(price_trend,time_step)

        self.train_size = int(split_ratio * (len(self.y) - time_step - 1))

        # self.validation_size = (len(self.y) - self.train_size)//2

        self.test_size = len(self.y) - self.train_size


    def get_size(self):
        # return self.train_size, self.validation_size, self.test_size

        return self.train_size, self.test_size

    def get_num_features(self):
        return self.x[0].shape[1]

    def get_train_set(self):
        return self.x[:self.train_size], self.y[:self.train_size]

    # def get_validation_set(self):
    #     return self.x[self.train_size:self.train_size+self.validation_size],self.y[self.train_size:self.train_size+self.validation_size]

    def get_test_set(self):
        return self.x[self.train_size:], self.y[self.train_size:]


    def get_data(self):
        """
        get trend data
        :return: the data array
        """
        #read from csv
        dj = pd.read_csv(self.pricesfile)

        # Remove unneeded features
        dj = dj.drop(['High', 'Low', 'Close', 'Volume', 'Adj Close'], 1)
        dj = dj[::-1]
        # print(dj.head(10))

        # use the date as index to get the different each day
        dj = dj.set_index('Date').diff()
        # print(dj.head(110))

        # add the date back
        dj['Date'] = dj.index
        # print(dj.head(10))
        dj = dj.reset_index(drop=True)
        # print(dj.head(10))

        # remove the first row
        dj = dj[dj.Open.notnull()]

        # print(dj.head(20))
        prices_trend = []

        for row in dj.iterrows():

            prices_trend.append([0,1] if row[1]['Open'] >= 0 else [1,0])
        # print(prices_trend)

        return prices_trend

    def create_dataset(self,prices_trend, time_step):
        """

        :param prices_trend: stock list
        :param time_step: time step 10
        :return: three np array
        """
        x,y = [],[]
        for i in range(len(prices_trend) - time_step - 1):
            #set the last day
            last = i + time_step

            #add the input data into array
            x.append(prices_trend[i:last])
            #add the target
            y.append(1 if prices_trend[last] == [0,1] else 0)

            # if i%100 == 0:
            #     print('number {}'.format(i))
            #     print('x:',x)
            #     print('y:',y)

        return np.array(x),np.array(y)


class RNN(nn.Module):
    def __init__(self,input_size, hidden_size, out_size):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1,batch_first=True)
        self.out = nn.Linear(in_features=hidden_size,out_features=out_size)
        # self.softmax = nn.Softmax
    def forward(self,x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out,h = self.rnn(x,None)
        out = self.out(r_out[:,-1,:])
        return out

class LSTM(nn.Module):
    def __init__(self,input_size, hidden_size, out_size):
        super(LSTM,self).__init__()

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1,batch_first=True)
        self.out = nn.Linear(in_features=hidden_size,out_features=out_size)

    def forward(self,x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out,h = self.rnn(x)
        out = self.out(r_out[:,-1,:])
        return out

class GRU(nn.Module):
    def __init__(self,input_size, hidden_size, out_size):
        super(GRU,self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1,batch_first=True)
        self.out = nn.Linear(in_features=hidden_size,out_features=out_size)

    def forward(self,x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out,h = self.rnn(x)
        out = self.out(r_out[:,-1,:])
        return out


class Trainer():

    def __init__(self,model_type=0):
        self.model_type = model_type
        if model_type == 0:
            self.model = RNN(2,10,2)
        elif model_type == 1:
            self.model = LSTM(2,10,2)
        else:
            self.model = GRU(2,10,2)
        if torch.cuda.is_available():
            print('cuda is OK')
            self.model = self.model.cuda()
        self.dataset = DataSet(TARGET,TIME_STEP)
        self.train_size,self.test_size = self.dataset.get_size()
        self.optimizer = optim.Adam(self.model.parameters(),lr = LR)
        self.loss_function = nn.CrossEntropyLoss()

    def to_variable(self, x,output=False):
        if not output:
            if torch.cuda.is_available():
                return Variable(torch.from_numpy(x).float()).cuda()
            else:
                return Variable(torch.from_numpy(x).float())
        else:
            if torch.cuda.is_available():
                return Variable(torch.from_numpy(x).long()).cuda()
            else:
                return Variable(torch.from_numpy(x).long())


    def get_accuracy(self,truth,pred):
        assert len(truth) == len(pred)
        right = (truth == pred).sum()
        return right/len(truth)

    def train_minibatch(self):

        train_acc_list = []
        test_acc_list = []
        train_loss_list = []
        test_loss_list = []

        x,y = self.dataset.get_train_set()
        for epoch in range(num_epochs):
            print('Start epoch {}'.format(epoch))
            batch_number = 0
            i = 0
            loss_sum = 0
            pred_res_total = []
            while i < self.train_size:
                batch_number += 1
                self.optimizer.zero_grad()
                batch_end = i + batch_size
                if (batch_end >= self.train_size):
                    batch_end = self.train_size
                var_x = self.to_variable(x[i:batch_end])
                var_y = self.to_variable(y[i:batch_end],True)

                y_res = self.model(var_x)

                # print('y_res:',y_res)
                # print('var_y:',var_y)

                loss = self.loss_function(y_res,var_y)

                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()

                i = batch_end

                pred_y = y_res.data.cpu()
                pred_y = torch.max(F.softmax(pred_y, dim=1), 1)[1]

                pred_res_total.extend(pred_y)

            acc = self.get_accuracy(y, np.array(pred_res_total))
            ave_loss = loss_sum / batch_number
            # print('epoch [%d] finished: ' % (epoch))
            # print('Train: accuracy is %.1f, average loss is %.2f' %(acc * 100,ave_loss))


            test_acc,test_ave_loss= self.test()
            # print('Test: accuracy is %.1f, average loss is %.2f' %(test_acc * 100,test_ave_loss))

            train_acc_list.append(acc)
            test_acc_list.append(test_acc)
            train_loss_list.append(ave_loss)
            test_loss_list.append(test_ave_loss)


        return train_acc_list, train_loss_list,test_acc_list,test_loss_list


    def test(self):

        x, y = self.dataset.get_test_set()
        i = 0
        pred_res_total = []
        length = len(y)
        loss_sum = 0
        batch_num = 0
        while i < length:
            batch_num += 1
            batch_end = i + batch_size
            if batch_end >= length:
                batch_end = length
            var_x = self.to_variable(x[i:batch_end])
            var_y = self.to_variable(y[i:batch_end], True)

            y_res = self.model(var_x)
            loss = self.loss_function(y_res, var_y)
            loss_sum += loss.item()

            i = batch_end

            pred_y = y_res.data.cpu()
            pred_y = torch.max(F.softmax(pred_y, dim=1), 1)[1]

            pred_res_total.extend(pred_y)
        return self.get_accuracy(y, np.array(pred_res_total)),loss_sum/batch_num

    def draw_plot(self,train_list,dev_list,acc=True):
        plt.figure()
        plt.plot(np.array(train_list))
        plt.plot(np.array(dev_list))
        if acc:
            plt.title('{} Accuracy'.format(models[self.model_type]))
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig("{}{}_Accuracy.jpg".format(PLOT_PATH,models[self.model_type]))
            # plt.show()

        else:
            plt.title('{} Loss'.format(models[self.model_type]))
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig("{}{}_Loss.jpg".format(PLOT_PATH,models[self.model_type]))
            # plt.show()


def main():
    for i in range(3):
        trainer = Trainer(i)
        train_acc_list, train_loss_list,test_acc_list,test_loss_list = trainer.train_minibatch()
        trainer.draw_plot(train_acc_list, test_acc_list)
        trainer.draw_plot(train_loss_list,test_loss_list,False)


if __name__ =='__main__':

    main()




