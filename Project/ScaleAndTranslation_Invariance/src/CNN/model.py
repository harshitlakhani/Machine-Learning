"""
https://heartbeat.comet.ml/basics-of-image-classification-with-pytorch-2f8973c51864
"""

import torch
import torch.nn as nn


class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.relu(output)

        return output

class SimpleNet(nn.Module):
    def __init__(self,num_classes=9):
        super(SimpleNet,self).__init__()

        self.unit1 = Unit(in_channels=3,out_channels=16)        
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit2 = Unit(in_channels=16, out_channels=32)        
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit3 = Unit(in_channels=32, out_channels=64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv3_drop = nn.Dropout2d(p=0.2)

        self.net = nn.Sequential(self.unit1,
                                self.pool1,
                                self.unit2,
                                self.pool2,
                                self.unit3,
                                self.pool3,
                                self.conv3_drop,
                                )

        self.fc1 = nn.Linear(in_features=64 * 25 * 25,out_features=128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 9)

    def forward(self, input):
        output = self.net(input) # Size : [batch_size, 128, 6, 6]
        output = output.view(output.size(0), -1) # Size : [batch_size, 128 * 6 * 6]
        output = self.fc2(self.relu1(self.fc1(output)))
        return output

if __name__ == '__main__':
    x = torch.randn((1, 3, 200, 200))
    model = SimpleNet()
    model(x)