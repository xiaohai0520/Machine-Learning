import os
import matplotlib as mpl

if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pickle


batch_size = 64
lr = 0.01
momentum = 0.9
epoches = 30

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        # print('size', x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x,dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def getdataset():
    transform = torchvision.transforms.Compose([torchvision.transforms.Pad(2),torchvision.transforms.ToTensor()])

    # get the full data from the train
    full_dataset = torchvision.datasets.MNIST(root='./data/mnist', train=True,transform=transform, download=True)
    # get the size
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    test_data = torchvision.datasets.MNIST(root='./data/minst', train=False, transform=transform,download=True)

    return train_data,val_data,test_data


def getdataloader(data):

    return torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=True)


def train(net,train_loader,optimizer,criterion):
    net.train()
    total_loss = 0
    for id, (input, target) in enumerate(train_loader):

        # if torch.cuda.is_available():
        if next(net.parameters()).is_cuda:
            input = input.cuda()
            target = target.cuda()
            # net.to(device)
            # input = input.to(device)
            # target = target.to(device)

        # clear grad
        optimizer.zero_grad()
        # send into model
        output = net(input)

        # print(type(output))
        # print(output)
        # print(type(target))
        #
        # print(target)

        # cal the loss
        loss = criterion(output, target)
        total_loss += loss.item()
        # use backward
        loss.backward()
        # update
        optimizer.step()

        # if id % 200 == 0:
        #     print(' [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         id * len(input), len(train_loader.dataset),
        #         100. * id / len(train_loader), loss.item()))
    averg_loss = total_loss/(len(train_loader))
    return averg_loss

def val(net,val_loader,criterion):
    net.eval()
    total_loss = 0

    for input, target in val_loader:
        # if torch.cuda.is_available():
        if next(net.parameters()).is_cuda:
            input = input.cuda()
            target = target.cuda()
        #     input = input.to(device)
        #     target = target.to(device)

        output = net(input)

        # get val loss
        loss = criterion(output, target)
        # print(type(loss))
        total_loss += loss.item()

    # test_loss /= len(test_loader.dataset)
    final_loss = total_loss / (len(val_loader))
    print('\nval set: Loss: {:.6f}\n'.format(final_loss))

    return final_loss



# train the modele
def get_best_param(net,train_loader,val_loader,optimizer,criterion,epoches):
    train_loss_array = []
    val_loss_array = []
    min_loss = float('inf')
    # lambda1 = lambda epoch: epoch
    # lambda2 = lambda epoch: 0.95 ** epoch
    # scheduler = torch.optim.lr_scheduler.LamdbaLR(optimizer, lr_lambda=[lambda1, lambda2])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    for epoch in range(epoches):

        # scheduler.step()
        print('Epoch:', epoch)
        train_loss = train(net, train_loader,optimizer,criterion)
        val_loss = val(net, val_loader,criterion)

        train_loss_array.append(train_loss)
        val_loss_array.append(val_loss)

        if val_loss < min_loss:

            torch.save(net.state_dict(), 'params.pth')
            min_loss = val_loss

    return train_loss_array, val_loss_array


def test(net,loader):

    samples = {"correct":{}, "wrong":{}}
    results = [[0,0] for _ in range(10)]
    his = [[] for _ in range(10)]
    net.eval()
    for input,target in loader:
        # if torch.cuda.is_available():
        if next(net.parameters()).is_cuda:
            input = input.cuda()
            target = target.cuda()
        #     input = input.to(device)
        #     target = target.to(device)
            # net.to(device)

        # compare target with out

        # get target to numpy
        target = target.cpu().detach().numpy()

        #get the output to numpy
        out = net(input)
        # print(out)
        out = torch.max(out,1)[1].cpu().detach().numpy()


        # get the image to numpy
        input = input.cpu().detach().numpy()


        # save the number and image into dic

        for i, target_num in enumerate(target):
            # if guess right
            if out[i] == target_num:
                #target guess right
                results[target_num][1] += 1

                #save into dic
                if target_num not in samples["correct"]:
                    samples["correct"][target_num] = input[i].squeeze()

            else:
                #this number reg wrong +1
                results[target_num][0] += 1
                if target_num not in samples["wrong"]:
                    samples["wrong"][target_num] = input[i].squeeze()
            his[target_num].append(out[i])
    file = open('samplea.pkl', 'wb')
    pickle.dump(samples, file)
    file.close()
    return samples,results,his


def draw_loss_plot(train_losses,val_losses):
    fig = plt.figure()
    x = np.array(range(len(train_losses)))
    y1 = np.array(train_losses)
    y2 = np.array(val_losses)
    # plt.xticks(range(train_losses))
    plt.plot(x,y1,label='train',color='r')
    plt.plot(x,y2,label='val',color='b')

    # plt.axis([0, 30, 0, 5])
    plt.legend(["Train", "Validation"])
    plt.show()
    plt.savefig('loss_change')


def draw_accuracy_plot(his):
    # fig = plt.figure()
    # accuracy = [each[1]/sum(each)*100 for each in results]
    # totalAccuracy = sum([each[1] for each in results])/sum([sum(each) for each in results]) * 100
    # accuracy.append(totalAccuracy)
    # name = [str(i) for i in range(10)]
    # name.append('total')
    #
    # plt.hist(accuracy,align='center')
    # plt.xticks(range(11), name)
    # plt.xlabel('class')
    # plt.ylabel('accuracy')
    # plt.show()
    # plt.savefig('accuracy')

    for i in range(len(his)):

        nums = his[i]
        plt.figure()
        nums = np.array(nums)
        plt.hist(nums, bins=10,align='mid',width=0.5)
        plt.xlabel('class')
        plt.ylabel('frequency')
        name = [str(i) for i in range(10)]
        # name.append('a')
        plt.xticks(range(10),name)
        plt.title('number:{}'.format(i))
        plt.show()
        plt.savefig('his{}'.format(i))


def draw_allclass(results):
    draw = []
    for i,pair in enumerate(results):
        draw += [i] * pair[1]
    draw = np.array(draw)
    plt.figure()

    plt.hist(draw,bins=10,align='mid',width=0.5)
    name = [str(i) for i in range(10)]
    plt.xticks(range(10),name)
    plt.xlabel('class')
    plt.ylabel('frequency')
    plt.title('all_class')
    plt.show()
    plt.savefig('all')


def draw_samples(samples):
    keys = [i for i in range(10)]
    for i in range(10):
        if i in samples['correct'] and i in samples['wrong']:
            fig = plt.figure()
            cor,wro = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
            cor.imshow(samples['correct'][i],cmap='gray')
            cor.set_xlabel('correct')
            wro.imshow(samples['wrong'][i],cmap='gray')
            wro.set_xlabel('wrong')
            plt.show()
            plt.savefig('sample_{}'.format(i))




def main():

    net = Net()
    # net.load_state_dict(torch.load('./params.pth'))
    print(net)
    if torch.cuda.is_available():

        # net.to(device)
        net.cuda()

    #
    # get datasets
    train_data, val_data, test_data = getdataset()
    #
    # get loader
    train_loader = getdataloader(train_data)
    val_loader = getdataloader(val_data)
    test_loader = getdataloader(test_data)

    print(len(test_data))
    # create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    train_loss_array, val_loss_array = get_best_param(net, train_loader, val_loader, optimizer, criterion, epoches)

    np.save('train_losses',np.array(train_loss_array))
    np.save('val_losses', np.array(val_loss_array))


    draw_loss_plot(train_loss_array, val_loss_array)

    # net = Net()
    # net.load_state_dict(torch.load('net_params.pkl'))

    samples, results,his = test(net, test_loader)

    np.save('results',np.array(results))
    print(results)
    print(sum([each[1] for each in results])/sum([sum(each) for each in results]) * 100)

    draw_accuracy_plot(his)
    # draw_samples(samples)
    draw_allclass(results)






if __name__ == '__main__':
    main()


