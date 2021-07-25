import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms import Compose
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import no_net
from FGSM import FGSM_attacker
from PGD import PGD_attacker
from random_chooser import  random_chooser

CUDA_on = True
cuda = CUDA_on and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# 数据导入
epochs = 1
# learning_rate = 0.001
learning_rate = 0.005
batch_size_train = 100
batch_size_test = 100
model = no_net.model
net = torch.load('Lenet5.pth')
model.to(device)
net.to(device)

#%%
#%%
# FGSM configs
FGSM_kwargs = {
    'epsilon': random_chooser([0.1, 0.15, 0.2, 0.25, 0.3]),
    'model': torch.load('Lenet5.pth'),
    'training': False,
    'train_attack_on': False,
    'device': device
}

#%%
# PGD configs
PGD_kwargs = {
    'k': random_chooser([5, 10, 15, 20]),
    'epsilon': random_chooser([0.1, 0.15, 0.2, 0.25, 0.3]),
    'alpha': random_chooser([0.03, 0.05, 0.07, 0.08, 0.1]),
    'model': torch.load('Lenet5.pth'),
    'training': False,
    'train_attack_on': False,
    'device': device
}
#%%

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=Compose([ToTensor(),
                       Normalize((0.1307,), (0.3081,))])
)
testing_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=Compose([ToTensor(),
                       Normalize((0.1307,), (0.3081,))])
)
train_data = DataLoader(training_data, batch_size=batch_size_train, shuffle=True)
test_data = DataLoader(testing_data, batch_size=batch_size_test, shuffle=True)
# 优化器设置
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%%
fgsm_attacker = FGSM_attacker(**FGSM_kwargs)
pgd_attacker = PGD_attacker(**PGD_kwargs)
#%%

def new(data, target):
    optimizer.zero_grad()
    output = model(data)
    # print(output.shape)
    # target = target.resize_(batch_size_train, 784)
    loss = F.soft_margin_loss(output, target)
    loss.backward()
    optimizer.step()
    # output = Variable(output, requires_grad=False)
    # output.resize_(batch_size_train, 1, 28, 28)
    # plt.imshow(output[0][0], cmap='gray', interpolation='none')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    return loss


# 训练
def train(epoch1):
    model.train()
    batch_idx = 0
    for data, target in train_data:
        data, target = data.to(device), target.to(device)
        # plt.imshow(data[0][0], cmap='gray', interpolation='none')
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        batch_idx += 1
        data.requires_grad = True
        x = 2
        if x == 1:
            data_fgsm = fgsm_attacker.run_attack(data, target)
        # print(data_fgsm.shape)
            target_fgsm = data
            target_fgsm.requires_grad = False
            loss = new(data_fgsm, target_fgsm)
        else:
            data_pgd = pgd_attacker.run_attack(data, target)
            target_pgd = data
            target_pgd.requires_grad = False
            loss = new(data_pgd, target_pgd)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.
                  format(epoch1, batch_idx * len(data), len(train_data.dataset),
                         loss.item()))


for i in range(epochs):
    train(i + 1)

torch.save(model, 'no.pth')
