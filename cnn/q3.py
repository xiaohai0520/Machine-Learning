
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

batch_size = 64
lr = 0.01
momentum = 0.9
epoches = 10




def test(net,loader):


    samples = {"correct":{}, "wrong":{}}
    results = [[0,0] for _ in range(10)]

    net.eval()
    for input,target in loader:
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()


        out = net(input)
        # criterion = nn.CrossEntropyLoss()
        # loss = criterion(out,target)
        # print(loss.item())


        # compare target with out

        # get target to numpy
        target = target.cpu().detach().numpy()

        #get the output to numpy

        #

        #
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

    return samples,results


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


def draw_samples(samples):
    keys = [i for i in range(10)]
    for i in range(10):
        fig = plt.figure()
        cor,wro = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
        cor.imshow(samples['correct'][i],cmap='gray')
        cor.set_xlabel('correct')
        wro.imshow(samples['wrong'][i],cmap='gray')
        wro.set_xlabel('wrong')
        plt.show()

def getdataset():
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
    torchvision.transforms.RandomCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Grayscale(3),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


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

vgg16 = torchvision.models.vgg16(pretrained=True)
print(vgg16)

fc_features = vgg16.classifier[6].in_features
print(fc_features)
vgg16.classifier[6] = nn.Linear(fc_features, 10)

print(vgg16)

for param in vgg16.features.parameters():
    param.require_grad = False

train_data, val_data, test_data = getdataset()
train_loader = getdataloader(train_data)
val_loader = getdataloader(val_data)
test_loader = getdataloader(test_data)

# test with pre params
samples, results = test(vgg16, test_loader)
print(results)
accuracy = sum([each[1] for each in results]) / sum([sum(each) for each in results]) * 100

print(accuracy)
draw_accuracy_plot(results)
draw_samples(samples)


