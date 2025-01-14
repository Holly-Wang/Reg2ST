from scipy.stats import pearsonr
import torch
import numpy as np

def get_R(data1: torch.Tensor, data2: torch.Tensor, dim: int=1, func=pearsonr):
    r1, p1 = [], []
    for g in range(data1.shape[dim]):

        if dim == 1:
            r, pv = func(data1[:, g], data2[:, g])
        elif dim == 0:
            r, pv = func(data1[g, :], data2[g, :])
        r1.append(r)
        p1.append(pv)
    r1 = np.array(r1)
    p1 = np.array(p1)
    return r1, p1