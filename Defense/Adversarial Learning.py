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
from FGSM import FGSM_attacker
from PGD import PGD_attacker
from PGD_ad import PGD_ad_attacker
from ODI_PGD import ODI_PGD_attacker
from ODI_PGD_ad import ODI_PGD_ad_attacker
from Deep_Fool import Deep_Fool_attacker
import time
from MyNet import NeuralNetwork
from Resnet import ResNet

#%%

# parameter settings

print('Learning initializing...')
CUDA_on = True
epochs = 10
learning_rate = .1
gamma = 0.7
log_interval = 10
batch_size = 64
test_batch_size = 1000
train_attack_on = True

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

# model = NeuralNetwork().to(device)
model = ResNet().to(device)

#%%
# FGSM configs
FGSM_kwargs = {
    'epsilon': random_chooser([.5, .75, 1., 1.25, 1.5]),
    'model': model,
    'training': True,
    'train_lambda': 0.1,
    'train_attack_on': True,
    'device': device
}

#%%
# PGD configs
PGD_kwargs = {
    'k': random_chooser([2, 4, 8, 12, 16]),
    'epsilon': random_chooser([.5, .75, 1., 1.25, 1.5]),
    'alpha': random_chooser([.16, 0.25, 0.33, 0.41, .5]),
    'model': model,
    'training': True,
    'train_lambda': 0.1,
    'train_attack_on': True,
    'device': device
}
#%%
# PGD_ad configs
PGD_ad_kwargs = {
    'k': random_chooser([2, 4, 8, 12, 16]),
    'epsilon': random_chooser([.5, .75, 1., 1.25, 1.5]),
    'alpha': random_chooser([.16, 0.25, 0.33, 0.41, .5]),
    'model': model,
    'training': True,
    'train_lambda': 0.12,
    'train_attack_on': True,
    'device': device
}
#%%
# ODI_PGD configs
ODI_PGD_kwargs = {
    'ODI_k': random_chooser([1, 2, 3]),
    'k': random_chooser([5, 10, 15, 20]),
    'epsilon': random_chooser([.5, .75, 1., 1.25, 1.5]),
    'ODI_alpha': random_chooser([.16, 0.25, 0.33, 0.41, .5]),
    'alpha': random_chooser([.16, 0.25, 0.33, 0.41, .5]),
    'model': model,
    'training': True,
    'train_lambda': 0.12,
    'train_attack_on': True,
    'device': device
}
#%%
# ODI_PGD_ad configs
ODI_PGD_ad_kwargs = {
    'ODI_k': random_chooser([1, 2, 3]),
    'k': random_chooser([5, 10, 15, 20]),
    'epsilon': random_chooser([.5, .75, 1., 1.25, 1.5]),
    'ODI_alpha': random_chooser([.16, 0.25, 0.33, 0.41, .5]),
    'alpha': random_chooser([.16, 0.25, 0.33, 0.41, .5]),
    'model': model,
    'training': True,
    'train_lambda': 0.13,
    'train_attack_on': True,
    'device': device
}

#%%
# Deep Fool configs
Deep_Fool_kwargs = {
    'max_iter': random_chooser([2, 4, 8, 12, 16]),
    'model': model,
    'epsilon': random_chooser([.5, .75, 1., 1.25, 1.5]),
    'eps_control': True,
    'training': True,
    'train_lambda': 0.15,
    'train_attack_on': True,
    'device': device
}

#%%
# data reader
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


#%% md

### 神经网络

#%%
# training module
def train(model, train_set, optimizer, log_interval, epoch_id, device, loss_list, attack=False, attacker=None,):
    # start training
    model.train()

    for batch_id, (data, target) in enumerate(train_set):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        loss = F.nll_loss(output, target)
        # print(loss)
        # print(grad)
        if attack:
            # 对抗样本学习
            for att in attacker:
                att.train()
                _, att_loss = att.run_attack(data.clone(), target)
                # print(att_loss == torch.zeros_like(att_loss))

                loss = loss + att_loss
        # print(data.grad.data)
        optimizer.zero_grad()
        # print(data.grad.data)
        # print(loss)
        loss.backward()
        data = data.grad.data
        # print(data)
        optimizer.step()

        loss_list.append(loss.detach().cpu().item())

        if batch_id % log_interval == 0:
            print('Epoch: {}\tData: {}/{} ({:.0f}%)\tMemory used: {:.2f} GB\tLoss: {:.10f}'.format(epoch_id, batch_id * len(data), len(train_set.dataset), 100. * batch_id / len(train_set), torch.cuda.memory_allocated(device) / 1024 / 1024, loss.item()))

#%%

# testing module
def test(model, test_set, device, loss_list, acc_list, attack=False, attacker=None):
    if not attack:
        attack_type = 'None'
    elif isinstance(attacker, FGSM_attacker):
        attack_type = 'FGSM'
    elif isinstance(attacker, PGD_attacker):
        attack_type = 'PGD'
    elif isinstance(attacker, PGD_ad_attacker):
        attack_type = 'PGD ad'
    elif isinstance(attacker, ODI_PGD_attacker):
        attack_type = 'ODI PGD'
    elif isinstance(attacker, ODI_PGD_ad_attacker):
        attack_type = 'ODI PGD ad'
    elif isinstance(attacker, Deep_Fool_attacker):
        attack_type = 'Deep Fool'
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
        attacker.eval()
        for data, target in test_set:
            data, target = data.to(device), target.to(device)
            data.requires_grad = True
            data_perturbed, _ = attacker.run_attack(data, target)
            data_perturbed = data_perturbed.detach()
            output = model(data_perturbed)
            avg_loss += F.nll_loss(output, target).item() * len(data)
            pred = output.argmax(dim=1)
            correct_cnt += pred.eq(target).sum().item()

    avg_loss /= len(test_set.dataset)
    loss_list.append(avg_loss)
    accuracy = correct_cnt / len(test_set.dataset)
    acc_list.append(accuracy)
    print('Memory used: {:.2f} GB\tAverage Loss: {:.10f}\tAccuracy: {}/{}({:.0f}%)'.format(torch.cuda.memory_allocated(device) / 1024 / 1024, avg_loss, correct_cnt, len(test_set.dataset), 100. * accuracy))


#%%
# data loaders
train_set = torch.utils.data.DataLoader(training_data, batch_size=batch_size, **cuda_kwargs, drop_last=True)
test_set  = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, **cuda_kwargs)

#%% md

### 模型训练


#%%
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
lr_scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
fgsm_attacker = FGSM_attacker(**FGSM_kwargs)
pgd_attacker = PGD_attacker(**PGD_kwargs)
pgd_ad_attacker = PGD_ad_attacker(**PGD_ad_kwargs)
odi_pgd_attacker = ODI_PGD_attacker(**ODI_PGD_kwargs)
odi_pgd_ad_attacker = ODI_PGD_ad_attacker(**ODI_PGD_ad_kwargs)
deep_fool_attacker = Deep_Fool_attacker(**Deep_Fool_kwargs)


#%%

adopted_train_set = train_set
# the real work
train_losses = []
test_losses = []
test_accuracies = []
fgsm_attack_losses = []
fgsm_attack_accuracies = []
pgd_attack_losses = []
pgd_attack_accuracies = []
pgd_ad_attack_losses = []
pgd_ad_attack_accuracies = []
odi_pgd_attack_losses = []
odi_pgd_attack_accuracies = []
odi_pgd_ad_attack_losses = []
odi_pgd_ad_attack_accuracies = []
deep_fool_attack_losses = []
deep_fool_attack_accuracies = []

# test before training
test(model, test_set, device, test_losses, test_accuracies, attacker=None)
test(model, test_set, device, fgsm_attack_losses, fgsm_attack_accuracies, attack=True, attacker=fgsm_attacker)
test(model, test_set, device, pgd_attack_losses, pgd_attack_accuracies, attack=True, attacker=pgd_attacker)
test(model, test_set, device, pgd_ad_attack_losses, pgd_ad_attack_accuracies, attack=True, attacker=pgd_ad_attacker)
test(model, test_set, device, odi_pgd_attack_losses, odi_pgd_attack_accuracies, attack=True, attacker=odi_pgd_attacker)
test(model, test_set, device, odi_pgd_ad_attack_losses, odi_pgd_ad_attack_accuracies, attack=True, attacker=odi_pgd_ad_attacker)
test(model, test_set, device, deep_fool_attack_losses, deep_fool_attack_accuracies, attack=True, attacker=deep_fool_attacker)

for epoch_id in range(1, epochs + 1):
    torch.cuda.empty_cache()
    train(model, adopted_train_set, optimizer, log_interval, epoch_id, device, train_losses, attack=train_attack_on, attacker=[fgsm_attacker, pgd_attacker, pgd_ad_attacker, deep_fool_attacker],)
    test(model, test_set, device, test_losses, test_accuracies, attacker=None)
    test(model, test_set, device, fgsm_attack_losses, fgsm_attack_accuracies, attack=True, attacker=fgsm_attacker)
    test(model, test_set, device, pgd_attack_losses, pgd_attack_accuracies, attack=True, attacker=pgd_attacker)
    test(model, test_set, device, pgd_ad_attack_losses, pgd_ad_attack_accuracies, attack=True, attacker=pgd_ad_attacker)
    test(model, test_set, device, odi_pgd_attack_losses, odi_pgd_attack_accuracies, attack=True, attacker=odi_pgd_attacker)
    test(model, test_set, device, odi_pgd_ad_attack_losses, odi_pgd_ad_attack_accuracies, attack=True, attacker=odi_pgd_ad_attacker)
    test(model, test_set, device, deep_fool_attack_losses, deep_fool_attack_accuracies, attack=True, attacker=deep_fool_attacker)
    lr_scheduler.step()

#%%

# visualization

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

fig, ax_loss = plt.subplots(figsize=(20,15))
plt.tick_params(labelsize=18)

length = len(train_losses)
test_len = len(test_losses)
x_train = range(length)
x_test = np.linspace(0, length, test_len)
ax_acc = ax_loss.twinx()
plt.tick_params(labelsize=18)

# three lines
lns = []

lns += ax_loss.plot(x_train, train_losses, linewidth=3, c='#c45e81', label='train loss',)

lns += ax_loss.plot(x_test, test_losses, linewidth=3, c='#6850d0', label='test loss')
lns += ax_acc.plot(x_test, test_accuracies, linewidth=3, c='#6850d0', label='test accuracy')

lns += ax_loss.plot(x_test, fgsm_attack_losses, linewidth=3, c='#9d655a', label='FGSM attacked test loss')
lns += ax_acc.plot(x_test, fgsm_attack_accuracies, linewidth=3, c='#9d655a', label='FGSM attacked test accuracy')

lns += ax_loss.plot(x_test, pgd_attack_losses, linewidth=3, c='#edc399', label='PGD attacked test loss')
lns += ax_acc.plot(x_test, pgd_attack_accuracies, linewidth=3, c='#edc399', label='PGD attacked test accuracy')

lns += ax_loss.plot(x_test, pgd_ad_attack_losses, linewidth=3, c='#445225', label='PGD ad attacked test loss')
lns += ax_acc.plot(x_test, pgd_ad_attack_accuracies, linewidth=3, c='#445225', label='PGD ad attacked test accuracy')

lns += ax_loss.plot(x_test, odi_pgd_attack_losses, linewidth=3, c='#c8d448', label='ODI PGD attacked test loss')
lns += ax_acc.plot(x_test, odi_pgd_attack_accuracies, linewidth=3, c='#c8d448', label='ODI PGD attacked test accuracy')

lns += ax_loss.plot(x_test, odi_pgd_ad_attack_losses, linewidth=3, c='#8fb1cf', label='ODI PGD ad attacked test loss')
lns += ax_acc.plot(x_test, odi_pgd_ad_attack_accuracies, linewidth=3, c='#8fb1cf', label='ODI PGD ad attacked test accuracy')

lns += ax_loss.plot(x_test, deep_fool_attack_losses, linewidth=3, c='#719382', label='Deep Fool attacked test loss')
lns += ax_acc.plot(x_test, deep_fool_attack_accuracies, linewidth=3, c='#719382', label='Deep Fool attacked test accuracy')

ax_loss.set_ylim(0., (max(train_losses) + max(max(test_losses), max(fgsm_attack_losses), max(pgd_attack_losses), max(pgd_ad_attack_losses), max(deep_fool_attack_losses))) / 8)
ax_acc.set_ylim(min(min(fgsm_attack_accuracies[1:]), min(pgd_attack_accuracies[1:]), min(pgd_ad_attack_accuracies[1:]), min(deep_fool_attack_accuracies[1:]) * 7 / 8, min(test_accuracies[1:])), max(max(test_accuracies), max(fgsm_attack_accuracies), max(pgd_attack_accuracies), max(pgd_ad_attack_accuracies), max(deep_fool_attack_accuracies)))
plt.xlim(0, length)
ax_loss.set_xlabel('Training Process', fontsize=24)
ax_loss.set_ylabel('Loss', fontsize=24)
ax_acc.set_ylabel('Accuracy', fontsize=24)
plt.title('MNIST 手写数字分类器学习曲线（攻击防御已{}）'.format('开启' if train_attack_on else '关闭'), fontsize=30)

# labels
labs = [l.get_label() for l in lns]
ax_loss.legend(lns, labs, loc=0, fontsize=18)

plt.show()
if not os.path.exists('./results'):
    os.mkdir('./results')
cur_time = time.strftime('%Y-%m-%d_%H.%M.%S',time.localtime(time.time()))
save_folder = './results/'+cur_time
os.mkdir(save_folder)
with open(save_folder+"/configs.txt", "w") as file:
    file.write('epoches={}\n'
               'learning rate={}\n'
               'gamma={}\n'
               'batch size={}\n'
               'FGSM kwargs={}\n'
               'PGD kwargs={}\n'
               'PGD ad kwargs={}\n'
               'ODI PGD kwargs={}\n'
               'ODI PGD ad kwargs={}'
               'Deep Fool kwargs={}'.format(epochs, learning_rate, gamma, batch_size, FGSM_kwargs, PGD_kwargs, PGD_ad_kwargs, ODI_PGD_kwargs, ODI_PGD_ad_kwargs, Deep_Fool_kwargs))

fig.savefig(save_folder + '/picture.png')


#%% md
### 模型保存与读取

#%%


# save model
torch.save(model, 'model.pth')
torch.save(model, save_folder+'/model.pth')

#%%

# load the model

model = torch.load('model.pth')
model.eval()

#%%

# randomly pick a picture and test the model
random_picker  = torch.utils.data.DataLoader(test_data, batch_size=1, **cuda_kwargs)
for pic, label in random_picker:
    pic = pic.to(device)
    label = label.item()
    break

with torch.no_grad():
    pred = model(pic).argmax(dim=1).item()

# show the picture
print('图片标签：{}\t预测结果：{}'.format(label, pred))
print('输入的图片：')
pic = pic.cpu().view(28, 28)
plt.imshow(1-pic, cmap=plt.cm.gray)
plt.show()
