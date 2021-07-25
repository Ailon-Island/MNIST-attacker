#%%
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yes_net
from FGSM import FGSM_attacker
from Homework.大作业.还原神经网络.PGD import PGD_attacker
from noise_MNIST_dataset import noise_classifier_MNIST
from random_chooser import random_chooser

#%%
# 数据导入
device = 'cuda'
epochs = 10
learning_rate = 0.01
batch_size_train = 64
batch_size_test = 1000

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
    transform=ToTensor()
)

training_data = noise_classifier_MNIST(FGSM_kwargs.copy(), PGD_kwargs.copy())

#%%

#%%
model = yes_net.NeuralNetwork().to(device)



#%%
train_data = DataLoader(training_data, batch_size=batch_size_train, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
test_data = DataLoader(testing_data, batch_size=batch_size_test, shuffle=True, num_workers=0, pin_memory=True)

#%%
fgsm_attacker = FGSM_attacker(**FGSM_kwargs)
pgd_attacker = PGD_attacker(**PGD_kwargs)
#%%
# 优化器设置
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)

#%%
def new(data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss

#%%

# 训练
def train(epoch_idx):
    model.train()
    for batch_idx, (data, target) in enumerate(train_data):
        data, target = data.to(device), target.to(device)

        data.requires_grad = True
        loss = new(data, target)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                  format(epoch_idx, batch_idx * len(data), len(train_data.dataset),
                         100. * batch_idx / len(train_data), loss.item()))

#%%
def test():
    model.eval()
    correct = 0
    cnt = 0
    for data, target in test_data:
        data, target = data.to(device), target.to(device)

        cnt += 1
        data.requires_grad = True
        data_fgsm = fgsm_attacker.run_attack(data, target)
        data_pgd = pgd_attacker.run_attack(data, target)
        target = torch.tensor([0 for i in range(batch_size_test)]).to(device)
        target_fgsm = torch.tensor([1 for i in range(batch_size_test)]).to(device)
        target_pgd = torch.tensor([2 for i in range(batch_size_test)]).to(device)
        model.zero_grad()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        output_fgsm = model(data_fgsm)
        pred_fgsm = output_fgsm.data.max(1, keepdim=True)[1]
        output_pgd = model(data_pgd)
        pred_pgd = output_pgd.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        correct += pred_fgsm.eq(target_fgsm.data.view_as(pred_fgsm)).sum()
        correct += pred_pgd.eq(target_pgd.data.view_as(pred_pgd)).sum()
        if cnt * test_data.batch_size % 1000 == 0:
            print("Attacking: {}/{} ({:.0f}%)\tAccuracy: {}/{} ({:.0f}%))".format(
                3 * cnt * test_data.batch_size, 3 * len(test_data.dataset),
                100. * cnt * test_data.batch_size / len(test_data.dataset),
                correct, 3 * cnt * test_data.batch_size, 100. * correct / cnt / 3/test_data.batch_size, ))
    print('Test set:Accuracy: {}/{} ({:.0f}%)'.format(
        correct,3 * len(test_data.dataset),
        100. * correct / (3 * len(test_data.dataset))))

#%%
for i in range(epochs):
    train(i + 1)
    test()
#%%
torch.save(model, 'yes.pth')

