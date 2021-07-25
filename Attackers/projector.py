import torch

def project(data, eps):
    data = data / (data.norm(p=2, dim=[2, 3]).view(-1, 1, 1, 1) + 1e-30) * eps
    return data