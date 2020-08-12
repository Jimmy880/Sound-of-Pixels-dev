from .music import *
from .music import RAWDataset, STFTDataset

def warpgrid(bs, HO, WO, warp=True):
    import numpy as np
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid

def compute_mask(mags, mag_mix, is_binary_mask=False):
    gt_masks = []

    for n in range(len(mags)):
        if is_binary_mask:
            # for simplicity, mag_N > 0.5 * mag_mix
            gt_masks.append((mags[n] > 0.5 * mag_mix).float())
        else:
            gt_mask = mags[n] / mag_mix
            # clamp to avoid large numbers in ratio masks
            gt_masks.append(gt_mask.clamp(0., 5.))
    return gt_masks

def process_mag(mags=None, mag_mix=None, device='cpu'):
    import torch
    import torch.nn.functional as F
    if mag_mix is not None:
        B = mag_mix.size(0)
        T = mag_mix.size(3)
    elif mags is not None:
        B = mags[0].size(0)
        T = mags[0].size(3)
    else: return

    grid_warp = torch.from_numpy(
        warpgrid(B, 256, T, warp=True)).to(device)
    if mags is not None:
        for n in range(len(mags)):
            mags[n] = F.grid_sample(mags[n], grid_warp)
    if mag_mix is not None:
        mag_mix = F.grid_sample(mag_mix, grid_warp)
    
    return mags, mag_mix

def compute_weight(mag_mix, weighted_loss=False):
    import torch
    # 0.1 calculate loss weighting coefficient: magnitude of input mixture
    if weighted_loss:
        weight = torch.log1p(mag_mix)
        weight = torch.clamp(weight, 1e-3, 10)
    else:
        weight = torch.ones_like(mag_mix)
    return weight