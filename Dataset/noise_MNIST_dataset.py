from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import torch
from FGSM import FGSM_attacker
from PGD import PGD_attacker
from PGD_ad import PGD_ad_attacker
from Deep_Fool import Deep_Fool_attacker
import numpy as np
from typing import Any, Tuple
import os

#%%
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

#%%
cuda_on = False
#%%
if cuda_on and torch.cuda.is_available():
    device = 'cuda'
    cuda_kwargs = {'num_workers': 0,
                   'pin_memory': True,
                   'shuffle': False}
else:
    device = 'cpu'
    cuda_kwargs = {}

#%%
train_loader = torch.utils.data.DataLoader(training_data, batch_size=2000, **cuda_kwargs)
#%%
if torch.cuda.is_available():
    cuda_kwargs = {'num_workers': 0,
                   'pin_memory': True,
                   'shuffle': True}
else:
    cuda_kwargs = {}

#%%
train_loader = torch.utils.data.DataLoader(training_data, batch_size=2000, **cuda_kwargs)
#%%
class noise_classifier_MNIST(Dataset):
    def __init__(self, *attacker_kargs, force=False):
        # %%
        self.save_dir = './noise_MNIST_dataset/'
        self.set = [None, None, None, None]

        attacker_kargs = tuple(map(lambda x: self.reset_kwargs(x), attacker_kargs))
        FGSM_kwargs, PGD_kwargs, PGD_ad_kwargs, Deep_Fool_kwargs, = attacker_kargs
        FGSM_kwargs1 = FGSM_kwargs.copy()
        PGD_kwargs1 = PGD_kwargs.copy()
        PGD_ad_kwargs1 = PGD_ad_kwargs.copy()
        Deep_Fool_kwargs1 = Deep_Fool_kwargs.copy()
        try:
            if force:
                raise IOError

            if not os.path.exists(self.save_dir):
                raise IOError

            FGSM_kwargs_in_file, PGD_kwargs_in_file, PGD_ad_kwargs_in_file, Deep_Fool_kwargs_in_file, = np.load(self.save_dir + 'configs.npy', allow_pickle=True)
            FGSM_kwargs, PGD_kwargs, PGD_ad_kwargs, Deep_Fool_kwargs, = tuple(map(lambda x: self.delete_unwanted(x), attacker_kargs))
            del attacker_kargs
            if not(FGSM_kwargs_in_file == FGSM_kwargs and PGD_kwargs_in_file == PGD_kwargs and PGD_ad_kwargs_in_file == PGD_ad_kwargs and Deep_Fool_kwargs_in_file == Deep_Fool_kwargs):
                print('FGSM:')
                print(FGSM_kwargs_in_file)
                print(FGSM_kwargs)
                del FGSM_kwargs_in_file, FGSM_kwargs
                print('PGD:')
                print(PGD_kwargs_in_file)
                print(PGD_kwargs)
                del PGD_kwargs_in_file, PGD_kwargs
                print('PGD ad:')
                print(PGD_ad_kwargs_in_file)
                print(PGD_ad_kwargs)
                del PGD_ad_kwargs_in_file, PGD_ad_kwargs
                print('Deep Fool:')
                print(Deep_Fool_kwargs_in_file)
                print(Deep_Fool_kwargs)
                del Deep_Fool_kwargs_in_file, Deep_Fool_kwargs
                raise IOError
        except (IOError, TypeError, ValueError, FileNotFoundError):
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            try:
                del attacker_kargs
            except UnboundLocalError:
                pass
            try:
                del FGSM_kwargs, PGD_kwargs, PGD_ad_kwargs, Deep_Fool_kwargs
            except UnboundLocalError:
                pass
            print('noise classifier MNIST:\tProcessed dataset not found. Generating it now...')
            # %%
            fgsm_transform = FGSM_attacker(**FGSM_kwargs1)
            pgd_transform = PGD_attacker(**PGD_kwargs1)
            pgd_ad_transform = PGD_ad_attacker(**PGD_ad_kwargs1)
            deep_fool_transform = Deep_Fool_attacker(**Deep_Fool_kwargs1)

            # %%
            self.delete_unwanted(FGSM_kwargs1)
            self.delete_unwanted(PGD_kwargs1)
            self.delete_unwanted(PGD_ad_kwargs1)
            self.delete_unwanted(Deep_Fool_kwargs1)
            configs = np.array([FGSM_kwargs1, PGD_kwargs1, PGD_ad_kwargs1, Deep_Fool_kwargs1])
            del FGSM_kwargs1, PGD_kwargs1, PGD_ad_kwargs1, Deep_Fool_kwargs1
            # %%
            num_seg = len(train_loader.dataset) // train_loader.batch_size
            # %%
            for batch_idx, (data, label) in enumerate(train_loader):
                print('Segment {}:'.format(batch_idx))
                data, label = data.to(device), label.to(device)

                print('FGSM training data generating...')
                fgsm_training_data, _ = fgsm_transform.run_attack(data, label)
                np.save('FGSM_seg_{}.npy'.format(batch_idx), fgsm_training_data.detach().cpu().numpy(),
                        allow_pickle=True)
                del fgsm_training_data
                if batch_idx == num_seg - 1:
                    del fgsm_transform
                print('PGD training data generating...')
                pgd_training_data, _ = pgd_transform.run_attack(data, label)
                np.save('PGD_seg_{}.npy'.format(batch_idx), pgd_training_data.detach().cpu().numpy(), allow_pickle=True)
                del pgd_training_data
                if batch_idx == num_seg - 1:
                    del pgd_transform
                print('PGD ad training data generating...')
                pgd_ad_training_data, _ = pgd_ad_transform.run_attack(data, label)
                np.save('PGD_ad_seg_{}.npy'.format(batch_idx), pgd_ad_training_data.detach().cpu().numpy(),
                        allow_pickle=True)
                del pgd_ad_transform
                if batch_idx == num_seg - 1:
                    del pgd_ad_training_data
                print('Deep Fool training data generating...')
                deep_fool_training_data, _ = deep_fool_transform.run_attack(data, label)
                np.save('Deep_Fool_seg_{}.npy'.format(batch_idx), deep_fool_training_data.detach().cpu().numpy(),
                        allow_pickle=True)
                del deep_fool_training_data
                if batch_idx == num_seg - 1:
                    del deep_fool_transform

            # %%
            # %%
            images = []
            for batch_idx in range(num_seg):
                images.append(np.load(self.save_dir + 'FGSM_seg_{}.npy'.format(batch_idx), allow_pickle=True))
            images = np.row_stack(images)
            np.save(self.save_dir + '1.npy', images, allow_pickle=True)
            del images

            images = []
            for batch_idx in range(num_seg):
                images.append(np.load(self.save_dir + 'PGD_seg_{}.npy'.format(batch_idx), allow_pickle=True))
            images = np.row_stack(images)
            np.save(self.save_dir + '2.npy', images, allow_pickle=True)
            del images

            images = []
            for batch_idx in range(num_seg):
                images.append(np.load(self.save_dir + 'PGD_ad_seg_{}.npy'.format(batch_idx), allow_pickle=True))
            images = np.row_stack(images)
            np.save(self.save_dir + '3.npy', images, allow_pickle=True)
            del images

            images = []
            for batch_idx in range(num_seg):
                images.append(
                    np.load(self.save_dir + 'Deep_Fool_seg_{}.npy'.format(batch_idx), allow_pickle=True))
            images = np.row_stack(images)
            np.save(self.save_dir + '4.npy', images, allow_pickle=True)
            del images

            np.save(self.save_dir + 'configs.npy', configs, allow_pickle=True)
            del configs



    def __len__(self):
        return 5 * len(training_data)

    def __getitem__(self,index) -> Tuple[Any, Any]:
        img_raw_index = index // 5
        perturbation_type = index % 5
        if perturbation_type == 0:
            img = training_data[img_raw_index][0]
        else:
            if not isinstance(self.set[perturbation_type - 1], np.ndarray):
                self.set[perturbation_type - 1] = np.load(self.save_dir + '{}.npy'.format(perturbation_type),
                                                          allow_pickle=True)
            img = self.set[perturbation_type - 1][img_raw_index][0]
        return torch.tensor(img), perturbation_type

    @staticmethod
    def reset_kwargs(kwargs):
        kwargs['device'], kwargs['training'] = device, False
        return kwargs

    @staticmethod
    def delete_unwanted(kwargs):
        del kwargs['model']
        del kwargs['training']
        return kwargs

# %%
class noise_MNIST(Dataset):
    def __init__(self, *attacker_kargs, force=False):
        # %%
        self.save_dir = './noise_MNIST_dataset/'
        self.set = [None, None, None, None]

        attacker_kargs = tuple(map(lambda x: self.reset_kwargs(x), attacker_kargs))
        FGSM_kwargs, PGD_kwargs, PGD_ad_kwargs, Deep_Fool_kwargs, = attacker_kargs
        FGSM_kwargs1 = FGSM_kwargs.copy()
        PGD_kwargs1 = PGD_kwargs.copy()
        PGD_ad_kwargs1 = PGD_ad_kwargs.copy()
        Deep_Fool_kwargs1 = Deep_Fool_kwargs.copy()
        try:
            if force:
                raise IOError

            if not os.path.exists(self.save_dir):
                raise IOError

            FGSM_kwargs_in_file, PGD_kwargs_in_file, PGD_ad_kwargs_in_file, Deep_Fool_kwargs_in_file, = np.load(self.save_dir + 'configs.npy', allow_pickle=True)
            FGSM_kwargs, PGD_kwargs, PGD_ad_kwargs, Deep_Fool_kwargs, = tuple(map(lambda x: self.delete_unwanted(x), attacker_kargs))
            del attacker_kargs
            if not(FGSM_kwargs_in_file == FGSM_kwargs and PGD_kwargs_in_file == PGD_kwargs and PGD_ad_kwargs_in_file == PGD_ad_kwargs and Deep_Fool_kwargs_in_file == Deep_Fool_kwargs):
                print('FGSM:')
                print(FGSM_kwargs_in_file)
                print(FGSM_kwargs)
                del FGSM_kwargs_in_file, FGSM_kwargs
                print('PGD:')
                print(PGD_kwargs_in_file)
                print(PGD_kwargs)
                del PGD_kwargs_in_file, PGD_kwargs
                print('PGD ad:')
                print(PGD_ad_kwargs_in_file)
                print(PGD_ad_kwargs)
                del PGD_ad_kwargs_in_file, PGD_ad_kwargs
                print('Deep Fool:')
                print(Deep_Fool_kwargs_in_file)
                print(Deep_Fool_kwargs)
                del Deep_Fool_kwargs_in_file, Deep_Fool_kwargs
                raise IOError
        except (IOError, TypeError, ValueError, FileNotFoundError):
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            try:
                del attacker_kargs
            except UnboundLocalError:
                pass
            try:
                del FGSM_kwargs, PGD_kwargs, PGD_ad_kwargs, Deep_Fool_kwargs
            except UnboundLocalError:
                pass
            print('noise MNIST:\tProcessed dataset not found. Generating it now...')

            # %%
            fgsm_transform = FGSM_attacker(**FGSM_kwargs1)
            pgd_transform = PGD_attacker(**PGD_kwargs1)
            pgd_ad_transform = PGD_ad_attacker(**PGD_ad_kwargs1)
            deep_fool_transform = Deep_Fool_attacker(**Deep_Fool_kwargs1)

            # %%
            self.delete_unwanted(FGSM_kwargs1)
            self.delete_unwanted(PGD_kwargs1)
            self.delete_unwanted(PGD_ad_kwargs1)
            self.delete_unwanted(Deep_Fool_kwargs1)
            configs = np.array([FGSM_kwargs1, PGD_kwargs1, PGD_ad_kwargs1, Deep_Fool_kwargs1])
            del FGSM_kwargs1, PGD_kwargs1, PGD_ad_kwargs1, Deep_Fool_kwargs1

            # %%
            num_seg = len(train_loader.dataset) // train_loader.batch_size

            # %%
            for batch_idx, (data, label) in enumerate(train_loader):
                print('Segment {}:'.format(batch_idx))
                data, label = data.to(device), label.to(device)

                print('FGSM training data generating...')
                fgsm_training_data = fgsm_transform.run_attack(data, label)
                np.save(self.save_dir+'FGSM_seg_{}.npy'.format(batch_idx), fgsm_training_data.detach().cpu().numpy(),
                        allow_pickle=True)
                del fgsm_training_data
                if batch_idx == num_seg - 1:
                    del fgsm_transform
                print('PGD training data generating...')
                pgd_training_data = pgd_transform.run_attack(data, label)
                np.save(self.save_dir+'PGD_seg_{}.npy'.format(batch_idx), pgd_training_data.detach().cpu().numpy(), allow_pickle=True)
                del pgd_training_data
                if batch_idx == num_seg - 1:
                    del pgd_transform
                print('PGD ad training data generating...')
                pgd_ad_training_data = pgd_ad_transform.run_attack(data, label)
                np.save(self.save_dir+'PGD_ad_seg_{}.npy'.format(batch_idx), pgd_ad_training_data.detach().cpu().numpy(),
                        allow_pickle=True)
                del pgd_ad_training_data
                if batch_idx == num_seg - 1:
                    del pgd_ad_transform
                print('Deep Fool training data generating...')
                deep_fool_training_data = deep_fool_transform.run_attack(data, label)
                np.save(self.save_dir+'Deep_Fool_seg_{}.npy'.format(batch_idx), deep_fool_training_data.detach().cpu().numpy(),
                        allow_pickle=True)
                del deep_fool_training_data
                if batch_idx == num_seg - 1:
                    del deep_fool_transform

            # %%
            images = []
            for batch_idx in range(num_seg):
                images.append(np.load(self.save_dir+'FGSM_seg_{}.npy'.format(batch_idx), allow_pickle=True))
            images = np.row_stack(images)
            np.save(self.save_dir+'1.npy', images, allow_pickle=True)
            del images

            images = []
            for batch_idx in range(num_seg):
                images.append(np.load(self.save_dir+'PGD_seg_{}.npy'.format(batch_idx), allow_pickle=True))
            images = np.row_stack(images)
            np.save(self.save_dir+'2.npy', images, allow_pickle=True)
            del images

            images = []
            for batch_idx in range(num_seg):
                images.append(np.load(self.save_dir+'PGD_ad_seg_{}.npy'.format(batch_idx), allow_pickle=True))
            images = np.row_stack(images)
            np.save(self.save_dir+'3.npy', images, allow_pickle=True)
            del images

            images = []
            for batch_idx in range(num_seg):
                images.append(np.load(self.save_dir+'Deep_Fool_seg_{}.npy'.format(batch_idx), allow_pickle=True))
            images = np.row_stack(images)
            np.save(self.save_dir+'4.npy', images, allow_pickle=True)
            del images

            np.save(self.save_dir + 'configs.npy', configs, allow_pickle=True)
            del configs


    def __len__(self):
        return 5 * len(training_data)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        img_raw_index = index // 5
        perturbation_type = index % 5
        if perturbation_type == 0:
            img, label = training_data[img_raw_index]
        else:
            if not isinstance(self.set[perturbation_type - 1], np.ndarray):
                self.set[perturbation_type - 1] = np.load(self.save_dir+'{}.npy'.format(perturbation_type), allow_pickle=True)
            img = self.set[perturbation_type - 1][img_raw_index]
            label = training_data[img_raw_index][1]
        return torch.tensor(img), label

    @staticmethod
    def reset_kwargs(kwargs):
        kwargs['device'], kwargs['training'] = device, False
        return kwargs

    @staticmethod
    def delete_unwanted(kwargs):
        del kwargs['model']
        del kwargs['training']
        return kwargs
