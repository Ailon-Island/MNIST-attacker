from torch import nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=1)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv8 = nn.Conv2d(64, 16, kernel_size=1)
        self.conv9 = nn.Conv2d(16, 6, kernel_size=1)
        self.conv10 = nn.Conv2d(6, 1, kernel_size=1)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.conv10(x)
        return x


model = NeuralNetwork()
