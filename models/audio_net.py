import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self, fc_dim=64, num_downs=5, ngf=64, use_dropout=False):
        super(Unet, self).__init__()

        # construct unet structure
        unet_block = UnetBlock(
            ngf * 8, ngf * 8, input_nc=None,
            submodule=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(
                ngf * 8, ngf * 8, input_nc=None,
                submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetBlock(
            ngf * 4, ngf * 8, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            ngf * 2, ngf * 4, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            ngf, ngf * 2, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            fc_dim, ngf, input_nc=1,
            submodule=unet_block, outermost=True)

        self.bn0 = nn.BatchNorm2d(1)
        self.unet_block = unet_block

    def forward(self, x):
        # N = args.num_mix
        # B = mag_mix.size(0)
        # T = mag_mix.size(3)

        # # 0.0 warp the spectrogram
        # if args.log_freq:
        #     grid_warp = torch.from_numpy(
        #         warpgrid(B, 256, T, warp=True)).to(args.device)
        #     mag_mix = F.grid_sample(mag_mix, grid_warp)
        #     for n in range(N):
        #         mags[n] = F.grid_sample(mags[n], grid_warp)

        # # 0.1 calculate loss weighting coefficient: magnitude of input mixture
        # if args.weighted_loss:
        #     weight = torch.log1p(mag_mix)
        #     weight = torch.clamp(weight, 1e-3, 10)
        # else:
        #     weight = torch.ones_like(mag_mix)

        # # 0.2 ground truth masks are computed after warpping!
        # gt_masks = [None for n in range(N)]
        # for n in range(N):
        #     if args.binary_mask:
        #         # for simplicity, mag_N > 0.5 * mag_mix
        #         gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
        #     else:
        #         gt_masks[n] = mags[n] / mag_mix
        #         # clamp to avoid large numbers in ratio masks
        #         gt_masks[n].clamp_(0., 5.)

        # # LOG magnitude
        # log_mag_mix = torch.log(mag_mix).detach()

        # # 1. forward net_sound -> BxCxHxW
        # feat_sound = self.net_sound(log_mag_mix)
        # feat_sound = activate(feat_sound, args.sound_activation)

        # # 2. forward net_frame -> Bx1xC
        # feat_frames = [None for n in range(N)]
        # for n in range(N):
        #     feat_frames[n] = self.net_frame.forward_multiframe(frames[n])
        #     feat_frames[n] = activate(feat_frames[n], args.img_activation)

        # # 3. sound synthesizer
        # pred_masks = [None for n in range(N)]
        # for n in range(N):
        #     pred_masks[n] = self.net_synthesizer(feat_frames[n], feat_sound)
        #     pred_masks[n] = activate(pred_masks[n], args.output_activation)

        # # 4. loss
        # err = self.crit(pred_masks, gt_masks, weight).reshape(1)

        # return err, \
        #     {'pred_masks': pred_masks, 'gt_masks': gt_masks,
        #      'mag_mix': mag_mix, 'mags': mags, 'weight': weight}

        x = self.bn0(x)
        x = self.unet_block(x)
        return x


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_input_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 use_dropout=False, inner_output_nc=None, noskip=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.noskip = noskip
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        if innermost:
            inner_output_nc = inner_input_nc
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_input_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)
        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        if outermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1)

            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)

            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost or self.noskip:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
