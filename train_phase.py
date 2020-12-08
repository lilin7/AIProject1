import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import CNN

#train
def load_train_data(data_path):

    train_data_path = data_path + '/kfold'
    #path = './train' # get training set (labeled with subfolders) location

    #use ImageFolder to load images in folder "train", each sub-folder is a class, 3 classes in total
    trainset = torchvision.datasets.ImageFolder(train_data_path, transform=transforms.Compose ([ #Compose several transform methods
                                                    transforms.Resize((32, 32)),  # resize to （h,w）. If input single number, is to keep the ratio and change the shortest edge to int
                                                    transforms.CenterCrop(32),
                                                    transforms.ToTensor()# convert data type, get the same format of training set as in examples
                                                ]))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, drop_last=True)
    return train_loader

def train_phase(number_of_epoch, data_path):
    acc_list = []
    train_loader = load_train_data(data_path)

    net = CNN.CNN()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # learning rate = 0.001
    criterion = nn.CrossEntropyLoss()  # Loss function: this criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    # training
    for epoch in range(number_of_epoch):  # train for 10 epochs (1 epoch is to go thru all images in training set)

        running_loss = 0.0  # variable to keep record of loss value
        for i, data in enumerate(train_loader, 0):  # use enumerate to get index and data from training set

            # get the inputs
            inputs, labels = data  # use enumerate to get data from training set, including label info

            # wrap them in Variable format
            inputs, labels = Variable(inputs), Variable(labels)

            # set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)  # send inputs (data from training set) to CNN instance net
            loss = criterion(outputs, labels)  # calculate loss
            loss.backward()  # backpropragation
            optimizer.step()  # when finishing backpropragation, update parameters
            running_loss += loss.item() # add up loss


            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            ## print average loss value for each 200 images
            if i % 200 == 199:
                print('epoch [%d, %5d]  Average loss: %.3f  Average accuracy: %.2f %%' % (epoch + 1, i + 1, running_loss / 200, (correct / total) * 100))
                running_loss = 0.0  # set loss value to 0 after each 200 images

    print('\nFinished Training')

    # when training is finished, save our CNN parameters
    torch.save(net, 'net.pkl')
    torch.save(net.state_dict(), 'net_params.pkl')