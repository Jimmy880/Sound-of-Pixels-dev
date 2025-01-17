#!/bin/bash

OPTS=""
OPTS+="--id SOLO "
OPTS+="--list_train data/train.csv "
OPTS+="--list_val data/val.csv "

# Models
OPTS+="--arch_sound dprnn6 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 256 "

# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 1 "
OPTS+="--loss sisnr "
# OPTS+="--weighted_loss 1 "
# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 44100 " # 44100, 65535
OPTS+="--audRate 11025 "

# learning params
OPTS+="--workers 24 "
OPTS+="--batch_size_per_gpu 4 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--dup_trainset 50 "
OPTS+="--lr_steps 40 80 "

# display, viz
OPTS+="--disp_iter 9 "
OPTS+="--num_vis 5 "
OPTS+="--num_val 100 "
OPTS+="--resume "
# OPTS+="--instr Cello Basson Clarinet DoubleBass Flute Horn Oboe Saxophone Trumpet Violin Tuba "

#export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1
python -u main.py $OPTS "$@"
