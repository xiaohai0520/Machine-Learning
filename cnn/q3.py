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

batch_size = 32
lr = 0.01
momentum = 0.9
epoches = 10

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print(device)

def test(net,loader):


    samples = {"correct":{}, "wrong":{}}
    results = [[0,0] for _ in range(10)]
    his = [[] for _ in range(10)]

    net.eval()
    for input,target in loader:

        if next(net.parameters()).is_cuda:
            # input = input.cuda()
            # target = target.cuda()
            input = input.to(device)
            target = target.to(device)

        # compare target with out

        # get target to numpy
        target = target.cpu().detach().numpy()

        #get the output to numpy
        out = net(input)
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
                    samples["correct"][target_num] = input[i][0].squeeze()


            else:
                #this number reg wrong +1
                results[target_num][0] += 1
                if target_num not in samples["wrong"]:
                    samples["wrong"][target_num] = input[i][0].squeeze()

            # print(input[i].squeeze())
            his[target_num].append(out[i])
    return samples,results,his


def draw_loss_plot(train_losses,val_losses):

    x = np.array(range(len(train_losses)))
    y1 = np.array(train_losses)
    y2 = np.array(val_losses)

    plt.plot(x,y1,label='train',color='r')
    plt.plot(x,y2,label='val',color='b')
    plt.legend(["Train", "Validation"])
    plt.show()


def draw_accuracy_plot(results):

    accuracy = [each[1]/sum(each)*100 for each in results]
    totalAccuracy = sum([each[1] for each in results])/sum([sum(each) for each in results]) * 100
    accuracy.append(totalAccuracy)
    name = [str(i) for i in range(10)]
    name.append('total')

    plt.bar(range(11),accuracy,align='center')
    plt.xticks(range(11), name)
    plt.xlabel('class')
    plt.ylabel('accuracy')
    plt.show()
    plt.savefig('vgg_accuracy')

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

def getdataset():
    transform = torchvision.transforms.Compose(

    [
    torchvision.transforms.Resize(32),
    # torchvision.transforms.CenterCrop(224),
    torchvision.transforms.Grayscale(3),
    torchvision.transforms.ToTensor()
    ])
    # torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    # get the full data from the train
    full_dataset = torchvision.datasets.FashionMNIST(root='./data/fashionmnist', train=True,
                                              transform=transform, download=True)
    # get the size
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    test_data = torchvision.datasets.FashionMNIST(root='./data/fashionmnist', train=False, transform=transform,
                                           download=True)

    return train_data,val_data,test_data


def getdataloader(data):

    return torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=True)


def train(net,train_loader,optimizer,criterion):
    net.train()
    total_loss = 0
    for id, (input, target) in enumerate(train_loader):

        # if torch.cuda.is_available():
        if next(net.parameters()).is_cuda:
            # input = input.cuda()
            # target = target.cuda()
            # net.to(device)
            input = input.to(device)
            target = target.to(device)

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

        if id % 200 == 0:
            print(' [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                id * len(input), len(train_loader.dataset),
                100. * id / len(train_loader), loss.item()))
    averg_loss = total_loss/(len(train_loader))
    return averg_loss

def val(net,val_loader,criterion):
    net.eval()
    total_loss = 0

    for input, target in val_loader:
        # if torch.cuda.is_available():
        if next(net.parameters()).is_cuda:
            # input = input.cuda()
            # target = target.cuda()
            input = input.to(device)
            target = target.to(device)

        output = net(input)

        # get val loss
        loss = criterion(output, target)
        # print(type(loss))
        total_loss += loss.item()

    # test_loss /= len(test_loader.dataset)
    final_loss = total_loss / (len(val_loader))
    print('\nval set: Loss: {:.6f}\n'.format(final_loss))

    return final_loss

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



# train the modele
def get_best_param(net,train_loader,val_loader,optimizer,criterion,epoches):
    train_loss_array = []
    val_loss_array = []

    for epoch in range(epoches):

        # scheduler.step()
        print('Epoch:', epoch)
        train_loss = train(net, train_loader,optimizer,criterion)
        val_loss = val(net, val_loader,criterion)

        train_loss_array.append(train_loss)
        val_loss_array.append(val_loss)


    return train_loss_array, val_loss_array




vgg16 = torchvision.models.vgg16(pretrained=True)


for param in vgg16.features.parameters():
    param.require_grad = False

features = list(vgg16.classifier.children())[1:-1]
features = [nn.Linear(512, 4096)] + features + [nn.Linear(4096, 10)]
print(features)


vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
print(vgg16)




if torch.cuda.is_available():
    # vgg16.cuda()
    vgg16.to(device)
#
#
# #
train_data, val_data, test_data = getdataset()
train_loader = getdataloader(train_data)
val_loader = getdataloader(val_data)
test_loader = getdataloader(test_data)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vgg16.parameters(), lr=lr, momentum=momentum)
#
train_loss_array, val_loss_array = get_best_param(vgg16, train_loader, val_loader, optimizer, criterion, epoches)
draw_loss_plot(train_loss_array, val_loss_array)
samples, results,his = test(vgg16, test_loader)
# # print(his)
print(results)
accuracy = sum([each[1] for each in results]) / sum([sum(each) for each in results]) * 100
print(accuracy)
draw_accuracy_plot(his)
draw_samples(samples)
draw_allclass(results)
