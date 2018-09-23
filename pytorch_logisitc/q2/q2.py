
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 



def readfile(filepath):
    df = pd.read_csv(filepath)
    alt = np.array(df[['MFCCs_10', 'MFCCs_17', 'Species']])


    x = []
    y = []
    z = []
    for data in alt:
        if data[2] == 'HylaMinuta':
            x.append([data[0],data[1]])
            y.append([0])
            z.append('red')
        else:
            x.append([data[0],data[1]])
            y.append([1])
            z.append('blue')
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return x,y,z

def genTrainTest(dat):

    dat = np.array(dat)
    N,L = dat.shape
    np.random.shuffle(dat)
    traindat = dat[:700,:L-1]
    trainlabel = dat[:700,L-1]
    testdat = dat[700:,:L-1]
    testlabel = dat[700:,L-1]
    return traindat,trainlabel,testdat,testlabel

def Accuracy(pred,label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = (np.argmax(pred,1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)



class LogisticRegression(nn.Module):

    def __init__(self):

        super(LogisticRegression, self).__init__()

        # self.input_size = input_size
        self.lr = nn.Linear(2, 1)


    def forward(self, x):
        datout = self.lr(x)
        return torch.sigmoid(datout)



if __name__=='__main__':

    files = ["Frogs.csv","Frogs-subsample.csv"]
    for file in files:
        traindat, trainlabel,z = readfile(file)

        if torch.cuda.is_available():
            model = LogisticRegression().cuda()
        else:
            model = LogisticRegression()

        Lossfunc = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.from_numpy(traindat).type(torch.FloatTensor)
        y = torch.from_numpy(trainlabel).type(torch.FloatTensor)
        # model.train()

        for epoch in range(5000):

            out = model(x)
            loss = Lossfunc(out, y)
            print_loss = loss.data.item()
            mask = out.ge(0.5).float()  # 0.5 classify
            # correct = (mask == inlabel).sum()  # correct numbers
            # acc = correct.item() / indata.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #
            if (epoch + 1) % 100 == 0:
                # print('*' * 10)

                print('*' * 10)
                print('epoch {}'.format(epoch + 1))  #
                print('loss is {:.4f}'.format(print_loss))  #
                # print('acc is {:.4f}'.format(acc))  #

        w0,w1 = model.lr.weight[0]
        w0 = float(w0.item())
        w1 = float(w1.item())

        b = float(model.lr.bias.item())


        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=z,s=10,lw=0,)
        plot_x = np.arange(-0.5,0.5,0.1)
        plot_y = (-w0*plot_x-b)/w1

        plt.plot(plot_x,plot_y,)
        plt.show()
        torch.save(model.state_dict(), "para1.pth")