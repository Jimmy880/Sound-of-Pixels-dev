import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        N = len(preds) # shape = mix, batch, len
        if weight is None:
            weight = preds[0].new_ones(1)

        errs = [self._forward(preds[n], targets[n], weight)
                for n in range(N)]
        err = torch.mean(torch.stack(errs))

        return err


class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))


class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.pow(pred - target, 2))


class BCELoss(BaseLoss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy(pred, target, weight=weight)


def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    if torch.isnan(20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))).any():
        print(eps, t, x, s, l2norm(t), l2norm(x_zm - t), (l2norm(x_zm - t)+eps))
        raise ValueError
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

class SISNRLoss(BaseLoss):
    def __init__(self):
        super(SISNRLoss, self).__init__()

    def _forward(self, pred, target, weight):
        return - sisnr(pred, target)

class UPITLoss(nn.Module):
    def __init__(self):
        super(UPITLoss, self).__init__()

    def forward(self, ests, egs):
        # spks x n x S
        refs = egs
        num_spks = len(refs)

        def sisnr_loss(permute):
            # for one permute, return `[batch]`, on this permute
            z = sum(
                [sisnr(ests[s], refs[t])
                for s, t in enumerate(permute)]) / len(permute)
                # average the value
            # print(z)
            return z

        # P x N
        N = egs[0].size(0)
        from itertools import permutations
        sisnr_mat = torch.stack( # `[permute, batch]`
            [sisnr_loss(p) for p in permutations(range(num_spks))])
        # print(sisnr_mat)
        max_perutt, _ = torch.max(sisnr_mat, dim=0) # find max-permute opt, return `[batch]`

        if torch.isnan(-torch.sum(max_perutt) / N):
            print(N, max_perutt, sisnr_mat)
            raise ValueError

        # si-snr
        return -torch.sum(max_perutt) / N # make average on spks
