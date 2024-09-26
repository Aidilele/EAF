import torch


def norm_kl_div(u1, lv1, u2, lv2):
    N = u1.shape[0] // u2.shape[0]
    u2 = u2.repeat(N, 1)
    lv2 = lv2.repeat(N, 1)
    v1 = (2 * lv1).exp()
    v2 = (2 * lv2).exp()
    kl = lv2 - lv1 + (v1 + (u1 - u2) ** 2) / v2 * 0.5 - 0.5
    return kl.view(N, -1, u1.shape[-1]).mean(dim=0)


def euc_distance(x, y):
    N = x.shape[0] // y.shape[0]
    y = y.repeat(N, 1)
    distances = torch.sqrt(torch.sum((x - y) ** 2, dim=1))
    return distances.view(N, -1, x.shape[-1]).mean(dim=0)
