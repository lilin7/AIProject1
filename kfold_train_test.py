import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


import general_methods

from sklearn.model_selection import KFold

import CNN

show_images_for_how_many_batch = 2
test_batch_size = 5
acc_list = []

path = './dataset'  # get training set (labeled with subfolders) location
classes = ('NotAPerson', 'Person', 'PersonMask', ) # define 3 classes for training and testing datasets

# use ImageFolder to load images in folder "train", each sub-folder is a class, 3 classes in total
trainset = torchvision.datasets.ImageFolder(path, transform=transforms.Compose([  # Compose several transform methods
    transforms.Resize((32, 32)),
    # resize to （h,w）. If input single number, is to keep the ratio and change the shortest edge to int
    transforms.CenterCrop(32),
    transforms.ToTensor(),  # convert data type, get the same format of training set as in examples
    #torchvision.transforms.Normalize((0.1307,), (0.3081,))
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
]))
print(trainset[0])
print(trainset[0][1]) #0, calss
print(type(trainset)) #<class 'torchvision.datasets.folder.ImageFolder'>
print(len(trainset)) # 600

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

#def imshow(img, labels, predicted):
def imshow(img, text):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.text(0, 1.5, text, transform=plt.gca().transAxes)
    plt.show()



kf = KFold(n_splits=10,shuffle=True)
for i_fold, (train_index, test_index) in enumerate(kf.split(trainset)):
    train = torch.utils.data.Subset(trainset, train_index)
    test = torch.utils.data.Subset(trainset, test_index)
    print('train set has', len(train_index))
    train_loader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, shuffle=True, drop_last=False)
    print('for fold:', i_fold)

    # print(train_loader) #<torch.utils.data.dataloader.DataLoader object at 0x0000021BC4EF5550>
    # print(type(train_loader)) #<class 'torch.utils.data.dataloader.DataLoader'>
    # print(len(train_loader)) # 600

    acc_list = []
    net = CNN.CNN()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # learning rate = 0.001
    criterion = nn.CrossEntropyLoss()  # Loss function: this criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    # training
    for epoch in range(2):  # train for 10 epochs (1 epoch is to go thru all images in training set)

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
            running_loss += loss.item()  # add up loss

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            ## print average loss value for each 200 images
            if i % 200 == 199:
                print('epoch [%d, %5d]  Average loss: %.3f  Average accuracy: %.2f %%' % (
                epoch + 1, i + 1, running_loss / 200, (correct / total) * 100))
                running_loss = 0.0  # set loss value to 0 after each 200 images

    print('\nFinished Training')

    # when training is finished, save our CNN parameters
    torch.save(net, 'net.pkl')
    torch.save(net.state_dict(), 'net_params.pkl')

    net = torch.load('net.pkl')  # load our net parameters from file

    correct = 0  # number of correct prediction
    total = 0  # number of total test cases
    batch_counter = 0

    # Below is for measure the precision, recall and F1-measure
    # for class "NotAPerson" 0
    tp_NotAPerson, fp_NotAPerson, fn_NotAPerson = 0, 0, 0

    # for class "Person" 1
    tp_Person, fp_Person, fn_Person = 0, 0, 0

    # for class "PersonMask" 2
    tp_PersonMask, fp_PersonMask, fn_PersonMask = 0, 0, 0

    # for confusion matrix
    conf_matrix = torch.zeros(3, 3)

    show_image_count = 0

    for images, labels in test_loader:  # one batch
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        batch_counter = batch_counter + 1

        if show_image_count < show_images_for_how_many_batch:
            # print for test reason
            print('\n*************For batch ' + str(batch_counter) + ' (' + str(
                test_batch_size) + ' images):*************')
            print('%-15s %-70s' % ("GroundTruth:",
                                   labels))  # print in format tensor([0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 2, 2, 1, 2, 0, 2, 0, 2, 2, 0, 2, 1, 1, 1, 1])
            print('%-15s %s' % ("Predicted:",
                                predicted))  # print in format tensor([0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 2, 2, 1, 2, 0, 2, 0, 2, 2, 0, 2, 1, 1, 1, 1])

            print('%-15s %s' % (
            'GroundTruth:', " ".join('%-12s' % classes[labels[number]] for number in range(labels.size(0)))))
            print('%-15s %s' % (
            'Predicted:', " ".join('%-12s' % classes[predicted[number]] for number in range(labels.size(0)))))

            text = 'GroundTruth:' + " ".join(
                '%-12s' % classes[labels[number]] for number in range(labels.size(0))) + '\nPredicted:    ' + " ".join(
                '%-12s' % classes[predicted[number]] for number in range(labels.size(0)))

            imshow(torchvision.utils.make_grid(images, nrow=5), text)

            show_image_count = show_image_count + 1

        # for confusion matrix
        conf_matrix = confusion_matrix(outputs, labels, conf_matrix)

        for number in range(labels.size(0)):
            if classes[labels[number]] == "NotAPerson" and classes[predicted[number]] == "NotAPerson":
                tp_NotAPerson = tp_NotAPerson + 1
            elif classes[labels[number]] != "NotAPerson" and classes[predicted[number]] == "NotAPerson":
                fp_NotAPerson = fp_NotAPerson + 1
            elif classes[labels[number]] == "NotAPerson" and classes[predicted[number]] != "NotAPerson":
                fn_NotAPerson = fn_NotAPerson + 1

            if classes[labels[number]] == "Person" and classes[predicted[number]] == "Person":
                tp_Person = tp_Person + 1
            elif classes[labels[number]] != "Person" and classes[predicted[number]] == "Person":
                fp_Person = fp_Person + 1
            elif classes[labels[number]] == "Person" and classes[predicted[number]] != "Person":
                fn_Person = fn_Person + 1

            if classes[labels[number]] == "PersonMask" and classes[predicted[number]] == "PersonMask":
                tp_PersonMask = tp_PersonMask + 1
            elif classes[labels[number]] != "PersonMask" and classes[predicted[number]] == "PersonMask":
                fp_PersonMask = fp_PersonMask + 1
            elif classes[labels[number]] == "PersonMask" and classes[predicted[number]] != "PersonMask":
                fn_PersonMask = fn_PersonMask + 1

    if total != 0:
        print('\nAccuracy of the test dataset : %.2f %%' % ((correct / total) * 100))

    # for printing precision, recall, f1measure
    general_methods.printTable([[tp_NotAPerson, fp_NotAPerson, fn_NotAPerson],
                                [tp_Person, fp_Person, fn_Person],
                                [tp_PersonMask, fp_PersonMask, fn_PersonMask]])

    # for confusion matrix
    print("\nconfusion matrix:")
    print(conf_matrix)

    df_cm = pd.DataFrame(conf_matrix.numpy(),
                         index=['NotAPerson', 'Person', 'PersonMask'],
                         columns=['NotAPerson', 'Person', 'PersonMask'])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap="BuPu")
    plt.show()
