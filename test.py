

# necessary packages
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

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




### 神经网络



from torch import nn

# network definition
from MyNet import NeuralNetwork



CUDA_on = True

max_iter = 10
overshoot = 0.02
log_interval = 1000
batch_size = 1000



# device configurations
cuda = CUDA_on and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")




attacker_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
model = torch.load('model4defence.pth').to(device)
model.eval()



# PGD
def Deep_Fool(img_raw, max_iter, overshoot, model, device):
    img_raw.requires_grad = True
    f_image = model.forward(img_raw).data.squeeze().view(img_raw.shape[0], -1)
    I = torch.argsort(f_image, dim=1, descending=True)
    label = I[:, 0].view(-1, 1)

    img = img_raw.clone()
    w = torch.zeros(img_raw.shape).to(device)
    r_tot = torch.zeros(img_raw.shape).to(device)

    x = Variable(img, requires_grad=True)
    fs = model.forward(x)

    for i in range(max_iter):
        # if not k_i.equal(label):
        #     break
        perturbation = torch.ones(img_raw.shape[0]).to(device) * np.inf
        perturbation = perturbation.view(-1, 1)
        fs_label = fs[np.arange(fs.shape[0])[:,None], label].sum()
        fs_label.backward(retain_graph=True)
        orig_grad = x.grad.data.clone()

        for k in range(len(I[0])):
            if k == 0:
                continue

            I_k = I[:,k].view(-1, 1)

            model.zero_grad()
            fs_k = fs[np.arange(fs.shape[0])[:,None], I_k].sum()
            fs_k.backward(retain_graph=True)
            cur_grad = x.grad.data.clone()

            w_k = cur_grad - orig_grad
            f_k = (fs[np.arange(fs.shape[0])[:,None], I_k] - fs[np.arange(fs.shape[0])[:,None], label]).data

            perturbation_k = abs(f_k) / torch.norm(w_k.flatten())
            index = perturbation_k < perturbation
            index = index.view(-1, 1).long()
            perturbation = torch.cat((perturbation, perturbation_k), 1)
            w = torch.cat((w, w_k), 1)
            perturbation = perturbation[np.arange(perturbation.shape[0])[:,None], index]
            w = w[np.arange(w.shape[0])[:,None], index]
        perturbation = perturbation.view(perturbation.shape[0], 1 ,1 ,1)
        r_i = w * (perturbation + 1e-4) / torch.norm(w)
        r_tot = r_tot + r_i

        img = img_raw + (1 + overshoot) * r_tot
        x = Variable(img, requires_grad=True)
        fs = model.forward(x)
        # k_i = torch.argmax(fs.data, dim=1).item()

    return img



def Deep_Fool_test( model, device, loader, max_iter, overshoot, log_interval):
    correct_cnt = 0
    log_correct_cnt = 0
    cnt = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        cnt += 1

        # with torch.no_grad():
        #     # feed the data and get its prediction
        #     output = model(data)
        #     pred = output.max(1, keepdim=True)[1]

        # run attack
        perturbed_data = Deep_Fool(data, max_iter, overshoot, model, device)
        pic = perturbed_data.detach()[8].cpu().view(28, 28)
        plt.imshow(1-pic, cmap=plt.cm.gray)

        # predict the perturbed data and compare with the label
        output = model(perturbed_data)
        pred_perturbed = output.argmax(dim=1)
        correct = pred_perturbed.eq(target).sum().item()
        correct_cnt += correct
        log_correct_cnt += correct

        if cnt * loader.batch_size % log_interval == 0:
            print("Attacking: {}/{} ({:.0f}%)\tMax Iteration: {}\tOvershoot: {}\tAccuracy: {}/{} ({:.0f}%)\t{}/{} ({:.0f}%)".format(cnt * loader.batch_size, len(loader.dataset), 100. * cnt * loader.batch_size / len(loader.dataset), max_iter, overshoot, correct_cnt, cnt * loader.batch_size, 100. * correct_cnt / cnt / loader.batch_size, log_correct_cnt, log_interval, 100. * log_correct_cnt / log_interval))
            log_correct_cnt = 0
            # alpha *= 10

    accuracy = 1. *  correct_cnt / len(loader.dataset)
    print("Max Iteration: {}\tOvershoot: {}\tAccuracy: {}/{} ({:.0f}%)".format(max_iter, overshoot, correct_cnt, len(loader.dataset), 100. * accuracy))

    # Return the accuracy and an adversarial example
    return accuracy



Deep_Fool_test(model, device, attacker_loader, max_iter, overshoot, log_interval)



# randomly pick a picture and test the model
random_picker  = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

for pic, label in random_picker:
    pic, label = pic.to(device), label.to(device)

    pred_raw = model(pic).argmax(dim=1).item()
    pic = Deep_Fool(pic, max_iter, overshoot, model, device).detach()
    pred = model(pic).argmax(dim=1).item()

    if pred_raw != pred:
        break

# show the picture
print('图片标签：{}\t攻击前预测结果：{}\t攻击后预测结果：{}'.format(label.item(), pred_raw, pred))
print('输入的图片：')
pic = pic.cpu().view(28, 28)
plt.imshow(1-pic, cmap=plt.cm.gray)
plt.show()

