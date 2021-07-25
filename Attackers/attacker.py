import torch
import torch.nn.functional as F
import numpy as np
from random_chooser import random_chooser
from projector import project

class attacker:
    def __init__(self, model, epsilon=None, training=True, train_lambda=1., train_attack_on=False, device='cpu'):
        self.model = model.to(device)
        self.training = training
        self.train_lambda = train_lambda
        self.train_attack_on = train_attack_on
        self.device = device
        self.eps = epsilon
        self.grad_bak = None
    def train_attack_start(self, toggle=True):
        self.train_attack_on = toggle

    def train(self, training=True):
        self.training = training

    def eval(self, training=False):
        self.training = training

    def run_attack(self, img, target):
        if self.training and not self.train_attack_on:
            return
        if self.training:
            # self.grad_bak = img.grad.data
            pass

        img_perturbed = self.generate_perturbation(img, target).detach()
        loss = None
        if self.training:
            img_perturbed.requires_grad = True
            self.model.zero_grad()
            output = self.model.forward(img_perturbed)
            loss = F.nll_loss(output, target)
            loss = loss * self.train_lambda
            # grad_perturbed = torch.autograd.grad(loss, img_perturbed)[0]
            # print(self.grad_bak)
            # print(grad_perturbed)
            # img.grad = self.add_grad(grad_perturbed)
        return img_perturbed, loss
    def add_grad(self, grad_perturbed):
        grad = self.grad_bak + grad_perturbed
        return grad

    def generate_perturbation(self, image, target):
        return image