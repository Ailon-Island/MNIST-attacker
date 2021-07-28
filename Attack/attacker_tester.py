#%%
# necessary packages
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import matplotlib.pyplot as plt
from random_chooser import random_chooser
from noise_MNIST_dataset import noise_MNIST
from FGSM import FGSM_attacker
from PGD import PGD_attacker
from PGD_ad import PGD_ad_attacker
from ODI_PGD import ODI_PGD_attacker
from ODI_PGD_ad import ODI_PGD_ad_attacker
from Deep_Fool import Deep_Fool_attacker
import time
#%%

from MyNet import NeuralNetwork
#%%
# parameter settings

print('Learning initializing...')
CUDA_on = True
epochs = 10
learning_rate = 0.2
gamma = 0.7
log_interval = 1
batch_size = 64
test_batch_size = 1000
model = torch.load('model4defence.pth')
#%%
# device configurations
cuda = CUDA_on and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
if cuda:
    cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True}
else:
    cuda_kwargs = {}
#%%
# FGSM configs
FGSM_kwargs = {
    'epsilon': random_chooser([1., 1.5, 2.0, 2.5, 3.0]),
    'model': model,
    'training': False,
    'train_attack_on': False,
    'device': device
}

#%%
# PGD configs
PGD_kwargs = {
    'k': random_chooser([5, 10, 15, 20]),
    'epsilon': random_chooser([1., 1.5, 2.0, 2.5, 3.0]),
    'alpha': random_chooser([.3, 0.5, 0.66, 0.83, 1.]),
    'model': model,
    'training': False,
    'train_attack_on': False,
    'device': device
}
#%%
# PGD_ad configs
PGD_ad_kwargs = {
    'k': random_chooser([5, 10, 15, 20]),
    'epsilon': random_chooser([1., 1.5, 2.0, 2.5, 3.0]),
    'alpha': random_chooser([.3, 0.5, 0.66, 0.83, 1.]),
    'model': model,
    'training': False,
    'train_attack_on': False,
    'device': device
}
#%%
# ODI_PGD configs
ODI_PGD_kwargs = {
    'ODI_k': random_chooser([1, 2, 3]),
    'k': random_chooser([5, 10, 15, 20]),
    'epsilon': random_chooser([1., 1.5, 2.0, 2.5, 3.0]),
    'ODI_alpha': random_chooser([.3, 0.5, 0.66, 0.83, 1.]),
    'alpha': random_chooser([.3, 0.5, 0.66, 0.83, 1.]),
    'model': model,
    'training': False,
    'train_attack_on': False,
    'device': device
}
#%%
# ODI_PGD_ad configs
ODI_PGD_ad_kwargs = {
    'ODI_k': 2,
    'k': 10,
    'epsilon': random_chooser([1., 1.5, 2.0, 2.5, 3.0]),
    'ODI_alpha': random_chooser([.3 , 0.5, 0.66, 0.83, 1.]),
    'alpha': random_chooser([.3, 0.5, 0.66, 0.83, 1.]),
    'model': model,
    'training': False,
    'train_attack_on': False,
    'device': device
}

#%%
# Deep Fool configs
Deep_Fool_kwargs = {
    'max_iter': random_chooser([2, 4, 6, 8]),
    'model': model,
    'epsilon': random_chooser([1., 1.5, 2.0, 2.5, 3.0]),
    'eps_control': True,
    'training': False,
    'train_attack_on': False,
    'device': device
}

#%%
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


#%% md

### 神经网络



#%%

# testing module
def test(model, test_set, device, loss_list, acc_list, attack=False, attacker=None):
    if not attack:
        attack_type = 'None'
    elif isinstance(attacker, FGSM_attacker):
        attack_type = 'FGSM'
    elif isinstance(attacker, PGD_attacker):
        attack_type = 'PGD'
    print('\nTesting...\tAttack: {}'.format(attack_type))
    model.eval()
    if attack:
        attacker.eval()

    avg_loss = 0
    correct_cnt = 0
    if not attack:
        with torch.no_grad():
            for data, target in test_set:
                data, target = data.to(device), target.to(device)
                output = model(data)
                avg_loss += F.nll_loss(output, target).item() * len(data)
                pred = output.argmax(dim=1)
                correct_cnt += pred.eq(target).sum().item()
    else:
        for data, target in test_set:
            data, target = data.to(device), target.to(device)
            data.requires_grad = True
            data_perturbed = attacker.run_attack(data, target)
            output = model(data_perturbed)
            avg_loss += F.nll_loss(output, target).item() * len(data)
            pred = output.argmax(dim=1)
            correct_cnt += pred.eq(target).sum().item()

    avg_loss /= len(test_set.dataset)
    loss_list.append(avg_loss)
    accuracy = correct_cnt / len(test_set.dataset)
    acc_list.append(accuracy)
    print('Average Loss: {:.10f}\tAccuracy: {}/{}({:.0f}%)'.format(avg_loss, correct_cnt, len(test_set.dataset), 100. * accuracy))




#%%
# data loaders
test_set = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, **cuda_kwargs)

#%%
def attacker_test( model, device, attacker, loader, log_interval):
    correct_cnt = 0
    log_correct_cnt = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # with torch.no_grad():
        #     # feed the data and get its prediction
        #     output = model(data)
        #     pred = output.max(1, keepdim=True)[1]

        # run attack
        data.requires_grad = True
        perturbed_data, _ = attacker.run_attack(data, target)

        # predict the perturbed data and compare with the label
        output = model(perturbed_data)
        pred_perturbed = output.argmax(dim=1)
        correct = pred_perturbed.eq(target).sum().item()
        correct_cnt += correct
        log_correct_cnt += correct

        if (batch_idx + 1) % log_interval == 0:
            print("Attacking: {}/{} ({:.0f}%)\tAccuracy: {}/{} ({:.0f}%)\t{}/{} ({:.0f}%)".format((batch_idx + 1) * loader.batch_size, len(loader.dataset), 100. * (batch_idx + 1) * loader.batch_size / len(loader.dataset), correct_cnt, (batch_idx + 1) * loader.batch_size, 100. * correct_cnt / (batch_idx + 1) / loader.batch_size, log_correct_cnt, log_interval * loader.batch_size, 100. * log_correct_cnt / log_interval / loader.batch_size))
            log_correct_cnt = 0
            # alpha *= 10

    accuracy = 1. *  correct_cnt / len(loader.dataset)
    print("Attacker: {}\tAccuracy: {}/{} ({:.0f}%)".format(type(attacker).__name__, correct_cnt, len(loader.dataset), 100. * accuracy))

    # Return the accuracy and an adversarial example
    return accuracy


#%%
fgsm_attacker = FGSM_attacker(**FGSM_kwargs) #95
pgd_attacker = PGD_attacker(**PGD_kwargs) #93
pgd_ad_attacker = PGD_ad_attacker(**PGD_ad_kwargs) #82
odi_pgd_attacker = ODI_PGD_attacker(**ODI_PGD_kwargs)
odi_pgd_ad_attacker = ODI_PGD_ad_attacker(**ODI_PGD_ad_kwargs)
deep_fool_attacker = Deep_Fool_attacker(**Deep_Fool_kwargs)

#%%
attacker = odi_pgd_ad_attacker


#%%
attacker_test(model, device, attacker, test_set, log_interval)

#%%
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# randomly pick a picture and test the model
random_picker  = torch.utils.data.DataLoader(test_data, batch_size=1, **cuda_kwargs)
for pic, label in random_picker:
    pic, label = pic.to(device), label.to(device)
    pic_perturbed, _ = attacker.run_attack(pic, label)
    label = label.item()
    break

with torch.no_grad():
    pred = model(pic_perturbed).argmax(dim=1).item()

# show the picture
plt.figure()
plt.subplot(1, 2, 1)
plt.title('输入的图片 图片标签 {}'.format(label))
pic = pic.cpu().detach().view(28, 28)
plt.imshow(1-pic, cmap=plt.cm.gray)

plt.subplot(1, 2, 2)
plt.title('攻击后图片 预测标签 {}'.format(pred))
pic_perturbed = pic_perturbed.cpu().detach().view(28, 28)
plt.imshow(1-pic_perturbed, cmap=plt.cm.gray)

plt.show()
