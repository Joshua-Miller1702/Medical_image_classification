import torch
import torch.nn.functional as F
from torch import nn
import math


class CNNmodel(nn.Module):
    def __init__(
        self,
        conv_filters=(32, 64, 128),
        input_channels=3,
        dense_units=256,
        img_size=(50, 50),
    ):
        super(CNNmodel, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=conv_filters[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # convolutional layer 1
            nn.BatchNorm2d(conv_filters[0]),  # normalization layer 1
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2),  # pooling layer
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_filters[0],
                out_channels=conv_filters[1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # convolutional layer 2
            nn.BatchNorm2d(conv_filters[1]),  # normalization layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2),  # pooling layer
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_filters[1],
                out_channels=conv_filters[2],
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # convolutional layer 3
            nn.BatchNorm2d(conv_filters[2]),  # normalization layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2),  # pooling layer
        )

        flat_img = list(img_size)
        flat_img = [i // 2**3 for i in flat_img]
        flat_img = math.prod(flat_img)

        self.fc1 = nn.Linear(
            conv_filters[2] * flat_img, dense_units
        )  # fully connected layer (* 6 * 6 depends on input image size basically is img_size//(2^number of pooling layers))
        self.fc2 = nn.Linear(dense_units, 1)  # output layer

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits
