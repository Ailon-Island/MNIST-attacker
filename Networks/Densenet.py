from torch import nn
import torch
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, inp, outp, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inp, outp, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outp)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outp, outp, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(outp)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x1 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x2 = x
        x = self.conv2(x)
        x = self.bn2(x)
        x3 = x

        if self.downsample is not None:
            x1 = self.downsample(x1)

        x = torch.cat((x, x1, x2, x3), 1)
        x = self.relu(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2], num_classes=10):
        self.inplanes = 64
        super(DenseNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x)


