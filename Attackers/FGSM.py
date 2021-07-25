import torch
import torch.nn.functional as F
from attacker import attacker
from random_chooser import random_chooser
from projector import project

class FGSM_attacker(attacker):
    def __init__(self, epsilon, model, training=True, train_lambda=1., train_attack_on=False, device='cpu'):
        super(FGSM_attacker, self).__init__(model, epsilon, training, train_lambda, train_attack_on, device)

    def generate_perturbation(self, image, target):
        if isinstance(self.eps, random_chooser):
            eps = self.eps.choice(size=image.shape[0]).to(self.device).view(-1, 1, 1, 1)
        else:
            eps = self.eps

        img = image.detach_()
        img.requires_grad = True
        self.model.zero_grad()
        loss = F.nll_loss(self.model(img), target)
        loss.backward()
        grad = img.grad.data

        sign_data_grad = grad.sign()
        perturbation = project(sign_data_grad, eps)
        perturbed_image = image + perturbation
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image