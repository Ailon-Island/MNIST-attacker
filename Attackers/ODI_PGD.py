# 我试着在 PGD 的基础上，改成直接利用归一化的梯度进行 perturbation 的方式，使得对原图像作等量改变的时候，对模型的计算产生更大的影响
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from random_chooser import random_chooser
from projector import project
from attacker import attacker

class ODI_PGD_attacker(attacker):
    def __init__(self, ODI_k, k, epsilon, ODI_alpha, alpha, model, training=True, train_lambda=1., train_attack_on=False,
                 device='cpu'):
        super(ODI_PGD_attacker, self).__init__(model, epsilon, training, train_lambda, train_attack_on, device)
        self.ODI_k = ODI_k
        self.ODI_alpha = ODI_alpha
        self.k = k
        self.alpha = alpha

    def generate_perturbation(self, img_raw, target):
        # randomize the input image
        if isinstance(self.k, random_chooser):
            k = self.k.choice(size=img_raw.shape[0], dtype=torch.long).to(self.device).view(-1, 1, 1, 1)
        else:
            k = torch.full([img_raw.shape[0]], self.k, device=self.device,).view(-1, 1, 1, 1)
        if isinstance(self.eps, random_chooser):
            eps = self.eps.choice(size=img_raw.shape[0]).to(self.device).view(-1, 1, 1, 1)
        else:
            eps = torch.full([img_raw.shape[0]], self.eps, device=self.device,).view(-1, 1, 1, 1)
        if isinstance(self.alpha, random_chooser):
            alpha = self.alpha.choice(size=img_raw.shape[0]).to(self.device).view(-1, 1, 1, 1)
        else:
            alpha = torch.full([img_raw.shape[0]], self.alpha, device=self.device,).view(-1, 1, 1, 1)
        if isinstance(self.ODI_k, random_chooser):
            ODI_k = self.ODI_k.choice(size=img_raw.shape[0], dtype=torch.long).to(self.device).view(-1, 1, 1, 1)
        else:
            ODI_k = torch.full([img_raw.shape[0]], self.ODI_k, device=self.device,).view(-1, 1, 1, 1)
        if isinstance(self.ODI_alpha, random_chooser):
            ODI_alpha = self.ODI_alpha.choice(size=img_raw.shape[0]).to(self.device).view(-1, 1, 1, 1)
        else:
            ODI_alpha = torch.full([img_raw.shape[0]], self.ODI_alpha, device=self.device,).view(-1, 1, 1, 1)

        image = Variable(img_raw.data, requires_grad=True)
        randVector_ = torch.FloatTensor(*self.model(image).shape).uniform_(-1., 1.).to(self.device)
        image = Variable(image.data, requires_grad=True)
        iter = max(k + ODI_k)
        for i in range(iter):
            in_ODI = i < ODI_k
            in_ODI = in_ODI.long()
            index = (1 - in_ODI , in_ODI)
            index = torch.cat(index, dim=1).view(-1, 2, 1, 1)
            self.model.zero_grad()
            loss_ODI = (self.model(image) * randVector_).sum(dim=1)
            loss_PGD = F.nll_loss(self.model(image), target, reduce=False)
            loss = torch.transpose(torch.stack([loss_PGD, loss_ODI]), 0, 1)
            loss = loss * index.view_as(loss)
            loss = loss.mean()
            loss.backward()

            iter_alpha = torch.cat((alpha, ODI_alpha), dim=1)
            iter_alpha = (iter_alpha.view(-1, 2) * index.view(-1, 2)).sum(dim=1).view_as(alpha)
            perturbation = project(image.grad.data.sign(), iter_alpha)
            image = image.data + perturbation

            perturbation1 = project(image - img_raw, eps)
            perturbation2 = perturbation1 * ODI_alpha / alpha
            perturbation = torch.cat((perturbation1, perturbation2), dim=1) * index
            perturbation = perturbation.sum(dim=[1]).view(-1, 1, 28, 28)
            image = img_raw.data + perturbation
            image = Variable(torch.clamp(image, 0, 1), requires_grad=True)

        return image
