import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms import Compose
from torch.autograd import Variable
import matplotlib.pyplot as plt
from attacker import attacker
from PGD import PGD_attacker
from train_recovery import train
from FGSM import FGSM_attacker
from random_chooser import random_chooser

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

CUDA_on = True
cuda = CUDA_on and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
# 数据导入
epochs = 3
learning_rate = 0.001
batch_size_train = 1000
batch_size_test = 64
log_interval = 100
model = torch.load('no.pth')
net = torch.load('Lenet5.pth')
model.to(device)
net.to(device)

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

testing_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=Compose([ToTensor(),
                       Normalize((0.1307,), (0.3081,))])
)
test_data = DataLoader(testing_data, batch_size=batch_size_test, shuffle=True)


#%%
fgsm_attacker = FGSM_attacker(**FGSM_kwargs)
pgd_attacker = PGD_attacker(**PGD_kwargs)


#%%

def test_fgsm():
    model.eval()
    correct = 0
    correct_new = 0
    cnt = 0
    for data, target in test_data:
        data, target = data.to(device), target.to(device)
        cnt += 1
        data.requires_grad = True
        data_fgsm, _ = fgsm_attacker.run_attack(data, target)
        data_fgsm, _ = Variable(data_fgsm, requires_grad=False)
        plt.imshow(data_fgsm[0][0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        plt.title('FGSM攻击')
        plt.show()
        data_new = model(data_fgsm)
        data_new = Variable(data_new, requires_grad=False)
        plt.imshow(data_new[0][0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        plt.title('FGSM还原')
        plt.show()
        model.zero_grad()
        output = net(data_fgsm)
        pred = torch.max(output.data, 1)[1]
        output_new = net(data_new)
        pred_new = torch.max(output_new.data, 1)[1]
        correct_new += pred_new.eq(target.data.view_as(pred_new)).sum()
        correct += pred.eq(target.data.view_as(pred)).sum()
    print('Accuracy: {}/{} ({:.0f}%) Accuracy_after: {}/{} ({:.0f}%)'.format(
        correct, cnt*batch_size_test, 100. * correct / cnt / batch_size_test,
        correct_new, cnt*batch_size_test, 100. * correct_new / cnt / batch_size_test))
    return


def test_pgd():
    model.eval()
    correct = 0
    correct_new = 0
    cnt = 0
    torch.cuda.empty_cache()
    for data, target in test_data:
        data, target = data.to(device), target.to(device)
        cnt += 1
        data.requires_grad = True
        data_pgd, _ = pgd_attacker.run_attack(data, target)
        data_new = model(data_pgd)
        model.zero_grad()
        output = net(data_pgd)
        pred = torch.max(output.data, 1)[1]
        output_new = net(data_new)
        pred_new = torch.max(output_new.data, 1)[1]
        correct_new += pred_new.eq(target.data.view_as(pred_new)).sum()
        correct += pred.eq(target.data.view_as(pred)).sum()
    print('Accuracy: {}/{} ({:.0f}%) Accuracy_after: {}/{} ({:.0f}%)'.format(
        correct, cnt * batch_size_test, 100. * correct / cnt / batch_size_test,
        correct_new, cnt * batch_size_test, 100. * correct_new / cnt / batch_size_test))
    return

test_pgd()
