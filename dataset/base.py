import random
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import torchaudio
import librosa
from PIL import Image

from . import video_transforms as vtransforms


class BaseDataset(torchdata.Dataset):
    def __init__(self, list_sample, opt, max_sample=-1, split='train'):
        # params
        self.num_frames = opt.num_frames
        self.stride_frames = opt.stride_frames
        self.frameRate = opt.frameRate
        self.imgSize = opt.imgSize
        self.audRate = opt.audRate
        self.audLen = opt.audLen
        self.audSec = 1. * self.audLen / self.audRate
        self.binary_mask = opt.binary_mask

        # STFT params
        self.log_freq = opt.log_freq
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.HS = opt.stft_frame // 2 + 1
        self.WS = (self.audLen + 1) // self.stft_hop

        self.split = split
        self.seed = opt.seed
        random.seed(self.seed)

        # initialize video transform
        self._init_vtransform()

        # list_sample can be a python list or a csv file of list
        # parser.add_argument('--instr', nargs='+', type=str, default=['Cello', 'Bassoon'])
        self.instr = opt.instr

        if isinstance(list_sample, str):
            # self.list_sample = [x.rstrip() for x in open(list_sample, 'r')]
            self.list_sample = []
            for row in csv.reader(open(list_sample, 'r'), delimiter=','):
                if len(row) < 2:
                    print('warn', row)
                    continue
                import os.path as P
                if P.basename(P.dirname(row[0])) in self.instr: # specific instr
                    self.list_sample.append(row)

        elif isinstance(list_sample, list):
            self.list_sample = list_sample
        else:
            raise('Error list_sample!')

        if self.split == 'train':
            self.list_sample *= opt.dup_trainset
        elif self.split != 'train' and max_sample > len(self.list_sample):
            self.list_sample *= (max_sample // len(self.list_sample)) + 1
        random.shuffle(self.list_sample)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def __len__(self):
        return len(self.list_sample)

    # video transform funcs
    def _init_vtransform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            transform_list.append(vtransforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC))
            transform_list.append(vtransforms.RandomCrop(self.imgSize))
            transform_list.append(vtransforms.RandomHorizontalFlip())
        else:
            transform_list.append(vtransforms.Resize(self.imgSize, Image.BICUBIC))
            transform_list.append(vtransforms.CenterCrop(self.imgSize))

        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
        transform_list.append(vtransforms.Stack())
        self.vid_transform = transforms.Compose(transform_list)

    # image transform funcs, deprecated
    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            self.img_transform = transforms.Compose([
                transforms.Scale(int(self.imgSize * 1.2)),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Scale(self.imgSize),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

    def _load_frames(self, paths):
        frames = []
        for path in paths:
            frames.append(self._load_frame(path))
        frames = self.vid_transform(frames)
        return frames

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _stft(self, audio): # audio [len] or [batch, len]
        '''

        Return: amp,pha: [1,512,173]
        '''
        if type(audio) == np.ndarray:
            audio = torch.from_numpy(audio)
        spec = torch.stft(
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        if audio.ndim == 1:
            spec.unsqueeze_(0) # 1, XX

        rea = spec[:, :, :, 0]
        img = spec[:, :, :, 1]

        amp = torch.abs(torch.sqrt(torch.pow(rea, 2) + torch.pow(img, 2)))
        phase = torch.atan2(img, rea)
        if audio.ndim == 2:
            amp.unsqueeze_(1)
            phase.unsqueeze_(1) # batch, 1, XX

        return amp, phase
        # return amp.squeeze(0), phase.squeeze(0)
        # return torch.from_numpy(amp), torch.from_numpy(phase)

    def _load_audio_file(self, path):
        if path.endswith('.mp3'):
            audio_raw, rate = torchaudio.load(path)
            audio_raw = audio_raw.numpy().astype(np.float32)

            # range to [-1, 1]
            audio_raw *= (2.0**-31)

            # convert to mono
            if audio_raw.shape[1] == 2:
                audio_raw = (audio_raw[:, 0] + audio_raw[:, 1]) / 2
            else:
                audio_raw = audio_raw[:, 0]
        else:
            audio_raw, rate = librosa.load(path, sr=None, mono=True)

        return audio_raw, rate

    def _load_audio(self, path, center_timestamp, nearest_resample=False):
        audio = np.zeros(self.audLen, dtype=np.float32)

        # silent
        if path.endswith('silent'):
            return audio

        # load audio
        audio_raw, rate = self._load_audio_file(path)

        # repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audSec:
            n = int(rate * self.audSec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample
        if rate > self.audRate:
            print('resmaple {}->{}'.format(rate, self.audRate))
            if nearest_resample:
                audio_raw = audio_raw[::rate//self.audRate]
            else:
                audio_raw = librosa.resample(audio_raw, rate, self.audRate)

        # crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_timestamp * self.audRate)
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)

        audio[self.audLen//2-(center-start): self.audLen//2+(end-center)] = \
            audio_raw[start:end]

        # randomize volume
        if self.split == 'train':
            scale = random.random() + 0.5     # 0.5-1.5
            audio *= scale
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.

        return audio

    def _mix_n_and_stft(self, audios, no_mix=False): # audio [num_mix, batch, len] or [num_mix, len]
        N = len(audios)
        mags = [None for n in range(N)]

        # sep + STFT
        if not no_mix:
            for n in range(N):
                # if type(audios[n]) == torch.Tensor:
                #     audios[n] = audios[n].detach().cpu().numpy()
                audios[n] /= N

        for n in range(N):
            ampN, _ = self._stft(audios[n])
            mags[n] = ampN
        if no_mix:
            return mags

        # Mix + STFT
        audio_mix = np.asarray(audios).sum(axis=0)
        amp_mix, phase_mix = self._stft(audio_mix)

        # to tensor
        audio_mix = torch.from_numpy(audio_mix)
        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])

        return amp_mix, mags, phase_mix, audio_mix

    def _mix_raw(self, audios): # and transfer to tensor
        N = len(audios)

        # mix
        for n in range(N):
            audios[n] /= N
        audio_mix = np.asarray(audios).sum(axis=0)

        # to tensor
        audio_mix = torch.from_numpy(audio_mix)
        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])

        return audio_mix

    def _dump_stft(self, audios, batch_data, args, preprocess=False): # pre-compute at the first time

        # pred_masks_ = outputs['pred_masks']
        # gt_masks_ = outputs['gt_masks']
        # mag_mix_ = outputs['mag_mix']
        # weight_ = outputs['weight']
        mag_mix = batch_data['mag_mix'].to(args.device)
        mags = [i.to(args.device) for i in batch_data['mags']]
        pred_mags = self._mix_n_and_stft(audios, no_mix=True)

        N = len(audios)
        # print('0', mags[0].shape, len(pred_mags), pred_mags[0].shape, mag_mix.shape)

        # 0.0 warp the spectrogram
        from . import process_mag
        if not preprocess:
            mag_mix = mag_mix + 1e-10
            if args.log_freq: mags, mag_mix = process_mag(mags, mag_mix, args.device)
        if args.log_freq: pred_mags, _ = process_mag(pred_mags, None, args.device)
        # print('1', mags[0].shape, pred_mags[0].shape, mag_mix.shape)

        # 0.2 ground truth masks are computed after warpping!
        from . import compute_mask
        gt_masks = compute_mask(mags, mag_mix, args.binary_mask)
        pred_masks = compute_mask(pred_mags, mag_mix, args.binary_mask)

        if 'weight' not in batch_data:
            from . import compute_weight
            batch_data['weight'] = compute_weight(mag_mix, args.weighted_loss)

        return {
            'mag_mix' : mag_mix,
            'pred_masks' : pred_masks,
            'gt_masks' : gt_masks,
            'weight' : batch_data['weight'],
        }

    def dummy_mix_data(self, N):
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        mags = [None for n in range(N)]

        amp_mix = torch.zeros(1, self.HS, self.WS)
        phase_mix = torch.zeros(1, self.HS, self.WS)

        for n in range(N):
            frames[n] = torch.zeros(
                3, self.num_frames, self.imgSize, self.imgSize)
            audios[n] = torch.zeros(self.audLen)
            mags[n] = torch.zeros(1, self.HS, self.WS)

        return amp_mix, mags, frames, audios, phase_mix, torch.zeros(self.audLen)
