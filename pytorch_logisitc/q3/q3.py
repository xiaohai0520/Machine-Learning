import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def readfile():

    datas = np.load('Hastie-data.npy')
    print(datas)

    x = []
    y = []
    z = []
    for data in datas:
        if data[2] == 0.0:
            x.append([data[0], data[1]])
            y.append([data[2]])
            z.append('red')

        else:
            x.append([data[0], data[1]])
            y.append([data[2]])
            z.append('blue')


    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    print(x)
    print(y)
    return x,y,z

def net(net_config):
    layers = []
    net_config = [2] + net_config
    for n in range(len(net_config) - 1):
        layers += [
            torch.nn.Linear(net_config[n], net_config[n + 1]),
            torch.nn.Sigmoid()
        ]
    layers += [
        torch.nn.Linear(net_config[-1], 1),
        torch.nn.Sigmoid()
    ]
    return nn.Sequential(*layers)


class simpleNet(nn.Module):

    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):

        super(simpleNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1))

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2))

        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))


    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        x = self.layer3(x)
        return torch.sigmoid(x)

if __name__=='__main__':

    traindat, trainlabel,z = readfile()

    if torch.cuda.is_available():
        model = simpleNet(2,10,10,1).cuda()
    else:
        model = simpleNet(2,10,10,1)

    Lossfunc = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = torch.from_numpy(traindat).type(torch.FloatTensor)
    y = torch.from_numpy(trainlabel).type(torch.FloatTensor)


    for epoch in range(1000):

        out = model(x)
        # print(out)
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


    x_min, x_max = traindat[:, 0].min() - 1, traindat[:, 0].max() + 1
    y_min, y_max = traindat[:, 1].min() - 1, traindat[:, 1].max() + 1
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)

    co = np.array(np.meshgrid(x, y)).transpose(1, 2, 0).reshape((-1, 2))
    dec = model(torch.FloatTensor(co)).data.numpy().squeeze()
    # decision_boundary = co[np.where(np.abs(decisions - 0.5) < 0.05)]

    X, Y = np.meshgrid(x, y)
    Z = dec.reshape(X.shape)
    plt.figure()
    plt.scatter(traindat[:, 0], traindat[:, 1], c=z, s=10, lw=0, )
    plt.contour(X, Y, Z, [0.5])
    plt.show()

    torch.save(model.state_dict(), "para.pth")