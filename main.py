import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


classes = ('NotAPerson', 'Person', 'PersonMask', ) # define 3 classes for training and testing datasets

#train
def load_train_data():
    path = './train' # get training set (labeled with subfolders) location

    #use ImageFolder to load images in folder "train", each sub-folder is a class, 3 classes in total
    trainset = torchvision.datasets.ImageFolder(path, transform=transforms.Compose ([ #Compose several transform methods
                                                    transforms.Resize((32, 32)),  # resize to （h,w）. If input single number, is to keep the ratio and change the shortest edge to int
                                                    transforms.CenterCrop(32),
                                                    transforms.ToTensor()# convert data type, get the same format of training set as in examples
                                                ]))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
    return train_loader

def load_test_data():
    path = './test' # get testing set (labeled with subfolders) location
    testset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((32, 32)),
                                                    transforms.ToTensor()])
                                                )
    test_loader = torch.utils.data.DataLoader(testset, batch_size=25, shuffle=True)
    return test_loader

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)  # input channel = 3，output channel = 6, apply 6 filters of 5*5
        self.conv2 = nn.Conv2d(6, 16, 5)  # input channel = 6，output channel = 16, apply 16 filters of 5*5

        self.fc1 = nn.Linear(5 * 5 * 16, 120) # input is 5*5*16 = 400*1, output 120*1, one dimentsion vector
        self.fc2 = nn.Linear(120, 84) # input is 120*1, output 84*1, one dimentsion vector
        self.fc3 = nn.Linear(84, 3) # input is 84*1, output 3*1, because there are 3 classes

    def forward(self, x):
        # input x, then go thru conv1, then activation function relu, then pooling
        x = self.conv1(x) # output size: 28*28*6
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # output size: 14*14*6 (apply 2*2 pooling (fliter size = 2, stride =2) to 28*28*6)

        # input x (14*14*6), then go thru conv2, then activation function relu, then pooling
        x = self.conv2(x) # output size: 10*10*16
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # output size: 5*5*16 (apply 2*2 pooling (fliter size = 2, stride =2) to 10*10*16)

        # flatten the activation maps to one dimention vector
        x = x.view(x.size()[0], -1)

        # pass thru 3 full connection layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_phase():
    acc_list = []
    train_loader = load_train_data()

    net = CNN()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # learning rate = 0.001
    criterion = nn.CrossEntropyLoss()  # Loss function: this criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    # training
    for epoch in range(5):  # train for 5 epochs (1 epoch is to go thru all images in training set)

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


def test_phase():
    test_loader = load_test_data() # load test datasets
    net = torch.load('net.pkl') # load our net parameters from file

    correct = 0 # number of correct prediction
    total = 0 # number of total test cases
    batch_counter = 0
    for images, labels in test_loader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        batch_counter = batch_counter+1

        # print for test reason
        print('\n*************For batch '+ str(batch_counter) + ':*************')
        # print('%-15s %-70s' %  ("GroundTruth:", labels))  # print in format tensor([0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 2, 2, 1, 2, 0, 2, 0, 2, 2, 0, 2, 1, 1, 1, 1])
        # print('%-15s %s' % ("Predicted:", predicted)) # print in format tensor([0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 2, 2, 1, 2, 0, 2, 0, 2, 2, 0, 2, 1, 1, 1, 1])

        print('%-15s %s' % ('GroundTruth:', " ".join('%-12s' % classes[labels[number]] for number in range(labels.size(0)))))
        print('%-15s %s' % ('Predicted:', " ".join('%-12s' % classes[predicted[number]] for number in range(labels.size(0)))))

    print('\nAccuracy of the test dataset : %.2f %%' % ((correct / total) * 100))


train_phase()
test_phase()