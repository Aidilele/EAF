import torch


class GaussianNormalizer():
    def __init__(self, x: torch.Tensor):
        self.mean, self.std = x.mean(0), x.std(0)
        self.std[torch.where(self.std == 0.)] = 1.

    def normalize(self, x: torch.Tensor):
        return (x - self.mean[None,]) / self.std[None,]

    def unnormalize(self, x: torch.Tensor):
        return x * self.std[None,] + self.mean[None,]


class MineMaxMinNormalizer():
    def __init__(self, x: torch.Tensor):
        self.min = x.min()
        self.scale = x.max() - x.min()

    def normalize(self, x: torch.Tensor):
        return (2*(x - self.min[None,]) / self.scale[None,])-1

    def unnormalize(self, x: torch.Tensor):
        return ((x+1)/2)*self.scale[None,]+self.min[None,]

class MaxMinNormalizer():
    def __init__(self, x: torch.Tensor):
        self.min = x.min()
        self.scale = x.max() - x.min()

    def normalize(self, x: torch.Tensor):
        return (x - self.min[None,]) / self.scale[None,]

    def unnormalize(self, x: torch.Tensor):
        return x*self.scale[None,]+self.min[None,]