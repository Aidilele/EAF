import torch


def norm_kl_div(u1, lv1, u2, lv2):
    v1 = (2 * lv1).exp()
    v2 = (2 * lv2).exp()
    kl = lv2 - lv1 + (v1 + (u1 - u2) ** 2) / v2 * 0.5 - 0.5
    return kl


def euc_distance(x, y):
    distances = torch.sqrt(torch.sum((x - y) ** 2, dim=1))
    return distances
