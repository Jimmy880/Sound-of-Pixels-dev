# System libs
import os

# Numerical libs
import torch

# Our libs
from arguments import ArgParser
from dataset import STFTDataset, RAWDataset
from models import ModelBuilder, activate

from utils import makedirs

# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets, crit, ckpt=None):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_frame = nets
        self.crit = crit

        if ckpt is not None:
            self.net_sound.load_state_dict(ckpt['sound'])
            self.net_frame.load_state_dict(ckpt['frame'])

    def forward(self, batch_data, args):
        audio_mix = batch_data['audio_mix'] # B, audio_len
        audios = batch_data['audios'] # num_mix, B, audio_len
        frames = batch_data['frames'] # num_mix, B, xxx

        N = args.num_mix
        B = audio_mix.size(0)

        # 2. forward net_frame -> Bx1xC
        feat_frames = [None for n in range(N)]
        for n in range(N):
            feat_frames[n] = self.net_frame.forward_multiframe(frames[n])
            feat_frames[n] = activate(feat_frames[n], args.img_activation)

        # 3. sound synthesizer
        pred_audios = [None for n in range(N)]
        for n in range(N):
        #     pred_masks[n] = self.net_synthesizer(feat_frames[n], feat_sound)
        #     pred_masks[n] = activate(pred_masks[n], args.output_activation)
            pred_audios[n] = self.net_sound(audio_mix, feat_frames[n])
            activate(pred_audios[n], args.sound_activation)

        # 4. loss
        err = self.crit(pred_audios, audios).reshape(1)
        # print("\"", self.crit([audio_mix, audio_mix], audios).item(), self.crit(audios, audios).item(), err.item(),"\"")

        return err, pred_audios # or masks


def create_optimizer(nets, args, checkpoint):
    (net_sound, net_frame) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound},
                    {'params': net_frame.features.parameters(), 'lr': args.lr_frame},
                    {'params': net_frame.fc.parameters(), 'lr': args.lr_sound}]
    # optimizer = torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    if checkpoint is not None and args.resume_optim:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return optimizer


def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    args.lr_synthesizer *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        fc_dim=args.num_channels,)
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        fc_dim=args.num_channels,
        pool_type=args.img_pool,)

    nets = (net_sound, net_frame)
    crit = builder.build_criterion(arch=args.loss)

    # Dataset and Loader
    dataset_train = RAWDataset(
        args.list_train, args, split='train')
    dataset_val = STFTDataset(
        args.list_val, args, max_sample=args.num_val, split='val')

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    args.epoch_iters = len(loader_train)
    args.disp_iter = len(loader_train) // args.disp_iter
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # Wrap networks
    netWrapper = NetWrapper(nets, crit, checkpoint)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)

    # Set up optimizer
    optimizer = create_optimizer(nets, args, checkpoint)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'sdr': [], 'sir': [], 'sar': []}
    } if checkpoint is None else checkpoint['history']

    from epoch import train, evaluate
    # Eval mode
    # evaluate(netWrapper, loader_val, history, 0, args)
    # if args.mode == 'eval':
    #     print('Evaluation Done!')
    #     return

    # Training loop
    init_epoch = 1 if checkpoint is None else checkpoint['epoch']
    print('Training start at ', init_epoch)
    for epoch in range(1, args.num_epoch + 1):
        train(netWrapper, loader_train, optimizer, history, epoch, args)

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(netWrapper, loader_val, history, epoch, args)

            # checkpointing
            from utils import save_checkpoint
            save_checkpoint(nets, history, optimizer, epoch, args)

        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)

    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # experiment name
    if args.mode == 'train':
        args.id += '-{}mix'.format(args.num_mix)
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-{}-{}-{}'.format(
            args.arch_frame, args.arch_sound, args.arch_synthesizer)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        args.id += '-{}'.format(args.img_pool)
        
        args.id += '-' + args.loss
        if args.weighted_loss:
            args.id += '-weightedLoss'
        args.id += '-channels{}'.format(args.num_channels)
        args.id += '-epoch{}'.format(args.num_epoch)
        # args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization/')
    
    if args.mode == 'eval' or args.resume:
        try:
            checkpoint = torch.load(os.path.join(args.ckpt, 'best.pth'), map_location='cpu')
            # checkpoint = os.path.join(args.ckpt, 'lastest.pth')
            print('Loaded', args.ckpt)
        except:
            print('Load model failed')
            checkpoint = None
    elif args.mode == 'train':
        makedirs(args.ckpt, remove=True)
        checkpoint = None
    else: raise ValueError

    # initialize best error with a big number
    args.best_err = float("inf")

    from utils import set_seed
    set_seed(args.seed)
    main(args, )
