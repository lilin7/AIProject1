import torch.nn as nn
import torch.nn.functional as F

# create a class inheriting from the nn.Module to define different layers of the network
# set 2 Convolutional Layers with activation and max pooling, and 3 Full Connection Layers
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(8 * 8 * 64, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)

        return x

# original
class CNN1(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)  # input channel = 3，output channel = 6, apply 6 filters of 5*5
        self.conv2 = nn.Conv2d(6, 16, 5)  # input channel = 6，output channel = 16, apply 16 filters of 5*5

        self.fc1 = nn.Linear(5 * 5 * 16, 120) # input is 5*5*16 = 400*1, output 120*1, one dimension vector
        self.fc2 = nn.Linear(120, 84) # input is 120*1, output 84*1, one dimension vector
        self.fc3 = nn.Linear(84, 3) # input is 84*1, output 3*1, because there are 3 classes

    def forward(self, x):
        # input x, then go thru conv1, then activation function relu, then pooling
        x = self.conv1(x) # output size: 28*28*6
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # output size: 14*14*6 (apply 2*2 pooling (filter size = 2, stride =2) to 28*28*6)

        # input x (14*14*6), then go thru conv2, then activation function relu, then pooling
        x = self.conv2(x) # output size: 10*10*16
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # output size: 5*5*16 (apply 2*2 pooling (filter size = 2, stride =2) to 10*10*16)

        # flatten the activation maps to one dimension vector
        x = x.view(x.size()[0], -1)

        # pass thru 3 full connection layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x