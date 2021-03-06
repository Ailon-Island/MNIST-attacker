# 我试着在 PGD 的基础上，改成直接利用归一化的梯度进行 perturbation 的方式，使得对原图像作等量改变的时候，对模型的计算产生更大的影响

import torch
import torch.nn.functional as F
from random_chooser import random_chooser
from projector import project
from attacker import attacker

class PGD_ad_attacker(attacker):
    def __init__(self, k, epsilon, alpha, model, training = True, train_lambda=1., train_attack_on = False, device='cpu'):
        super(PGD_ad_attacker, self).__init__(model, epsilon, training, train_lambda, train_attack_on, device)
        self.k = k
        self.alpha = alpha

    def generate_perturbation(self, img_raw, target):
        # randomize the input image
        if isinstance(self.k, random_chooser):
            k = self.k.choice(size=img_raw.shape[0], dtype=torch.long).to(self.device).view(-1, 1, 1, 1)
        else:
            k = torch.full([img_raw.shape[0]], self.k, device=self.device, ).view(-1, 1, 1, 1)
        k_max = torch.max(k)
        if isinstance(self.eps, random_chooser):
            eps = self.eps.choice(size=img_raw.shape[0]).to(self.device).view(-1, 1, 1, 1)
        else:
            eps = self.eps
        if isinstance(self.alpha, random_chooser):
            alpha = self.alpha.choice(size=img_raw.shape[0]).to(self.device).view(-1, 1, 1, 1)
        else:
            alpha = self.alpha

        init_perb = torch.rand(img_raw.shape).to(self.device) * 2 - 1
        img = img_raw + project(init_perb, eps)
        img = torch.clamp(img, 0, 1)  # control the values in image

        for t in range(k_max.item()):
            continue_perturbation = t < k
            continue_perturbation = continue_perturbation.view(-1, 1, 1, 1)

            img.detach_()
            img.requires_grad = True

            # get the gradient of the t-th image
            loss = F.nll_loss(self.model(img), target)
            self.model.zero_grad()
            loss.backward()
            grad = img.grad.data

            # get the (t+1)-th image
            perturbation = project(grad, alpha)
            img = img + perturbation * continue_perturbation  # perturbed
            perturbation = img - img_raw  # true perturbation
            perturbation = project(perturbation, eps)
            img = img_raw + perturbation
            img = torch.clamp(img, 0, 1)  # 保证不超出数据范围
        return img