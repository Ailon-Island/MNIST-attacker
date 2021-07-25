import torch
from attacker import attacker
import numpy as np
from random_chooser import random_chooser
from projector import project

class Deep_Fool_attacker(attacker):
    def __init__(self, max_iter, model, epsilon=None, eps_control=False, training=True, train_lambda=1., train_attack_on=False, device='cpu'):
        super(Deep_Fool_attacker, self).__init__(model, epsilon, training, train_lambda, train_attack_on, device)
        self.max_iter = max_iter
        self.eps_control = eps_control

    def generate_perturbation(self, image, target):
        if self.eps_control:
            if isinstance(self.eps, random_chooser):
                eps = self.eps.choice(size=image.shape[0]).to(self.device).view(-1, 1, 1, 1)
            else:
                eps = self.eps
        if isinstance(self.max_iter, random_chooser):
            max_iter = self.max_iter.choice(size=image.shape[0], dtype=torch.long).to(self.device).view(-1, 1, 1, 1)
        else:
            max_iter = self.max_iter
        max_max_iter = max(max_iter)

        img = image.clone().detach()
        img.requires_grad = True
        self.model.zero_grad()
        fs = self.model.forward(img)
        I = torch.argsort(fs.data.squeeze().view(len(target), -1), dim=1, descending=True)  # 最高概率 正确
        label = target.view(-1, 1)  # 原标签 正确

        fs_label = fs[np.arange(fs.shape[0])[:, None], label]
        fs_label_sum = fs_label.sum()
        orig_grad = torch.autograd.grad(fs_label_sum, img)[0]
        del fs_label_sum, fs
        del label

        for i in range(max_max_iter):
            w = torch.empty_like(image, device=self.device)

            img.requires_grad = True
            fs = self.model(img)
            pred = fs.argmax(dim=1)
            comparison = pred.eq(target.view_as(pred)).long().view(-1, 1, 1, 1)
            del pred
            check_iter = i < max_iter
            comparison = comparison * check_iter
            del check_iter

            if comparison.sum() == 0:
                break

            perturbation = torch.full_like(target, np.inf, device=self.device, dtype=torch.float).view(-1, 1)

            for k in range(1, 10):
                img.requires_grad = True
                fs = self.model.forward(img)
                I_k = I[:, k].view(-1, 1)

                self.model.zero_grad()
                fs_k = fs[np.arange(fs.shape[0])[:, None], I_k].sum()
                cur_grad = torch.autograd.grad(fs_k, img)[0]
                del fs_k
                w_k = cur_grad - orig_grad
                f_k = (fs[np.arange(fs.shape[0])[:, None], I_k] - fs_label).data
                perturbation_k = abs(f_k) / (torch.norm(w_k, dim=[2, 3]) + 1e-30)
                del f_k
                index = perturbation_k < perturbation
                index = index.view(-1, 1).long()
                perturbation = torch.cat((perturbation, perturbation_k), 1)
                w = torch.cat((w, w_k), 1)

                perturbation = perturbation[np.arange(perturbation.shape[0])[:, None], index]
                w = w[np.arange(w.shape[0])[:, None], index]
                del index, perturbation_k, w_k
            perturbation = perturbation.view(-1, 1, 1, 1)
            r_i = w * perturbation / (torch.norm(w, dim=[2, 3]).view(-1, 1, 1, 1) + 1e-30) * comparison
            del comparison, perturbation, w
            img = img + r_i
            del r_i
            if self.eps_control:
                img = image + project(img-image, eps)
            img = torch.clamp(img, 0, 1).detach()

        return img