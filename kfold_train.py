import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import CNN


def kfold_train_phase(i_fold, train_loader, number_of_epoch):
    print('\nBegin training for K-fold Cross Validation iteration', i_fold)

    acc_list = []
    net = CNN.CNN()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # learning rate = 0.001
    criterion = nn.CrossEntropyLoss()  # Loss function: this criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    # training
    for epoch in range(number_of_epoch):  # train for how many epochs (1 epoch is to go thru all images in training set)

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

    print('\nFinished training for K-fold Cross Validation iteration', i_fold)

    # when training is finished, save our CNN parameters
    torch.save(net, './saved_net/net' + str(i_fold) + '.pkl')
    torch.save(net.state_dict(), './saved_net/net_params' + str(i_fold) + '.pkl')