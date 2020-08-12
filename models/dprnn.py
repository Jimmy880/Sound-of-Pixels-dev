import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ResRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type='LSTM', dropout=0, skip=False, bidirectional=False):
        super(ResRNN, self).__init__()
        
        self.input_size = input_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.proj = nn.Linear(hidden_size*self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        self.rnn.flatten_parameters()
        rnn_output, _ = self.rnn(output)
        proj_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return proj_output
    
# dual-path RNN
class DPRNN(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, output_size, 
                 dropout=0, num_layers=1, bidirectional=True):
        super(DPRNN, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.factor = int(bidirectional)+1
        
        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_rnn.append(ResRNN(input_size, hidden_size, 'LSTM', dropout, bidirectional=True))  # intra-segment RNN is always noncausal
            self.col_rnn.append(ResRNN(input_size, hidden_size, 'LSTM', dropout, bidirectional=bidirectional))
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size, output_size, 1)
                                   )
            
    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = output.permute(0,3,2,1).contiguous().view(batch_size*dim2, dim1, -1)  # B*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0,3,2,1).contiguous()  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output
            
            col_input = output.permute(0,2,3,1).contiguous().view(batch_size*dim1, dim2, -1)  # B*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0,3,1,2).contiguous()  # B, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output
            
        output = self.output(output)
            
        return output
    
    
# base module for DPRNN-based modules
class DPRNN_base(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, num_spk=2, 
                 layer=4, segment_size=100, bidirectional=True):
        super(DPRNN_base, self).__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk
        
        self.eps = 1e-8
        
        # bottleneck
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)
        
        # dual-path RNN
        self.DPRNN = DPRNN('LSTM', self.feature_dim, self.hidden_dim, self.feature_dim*self.num_spk, 
                            num_layers=layer, bidirectional=bidirectional)
        
        # mask estimation layer
        self.output = nn.Sequential(nn.Conv1d(self.feature_dim, self.output_dim, 1),
                                    nn.ReLU(inplace=True)
                                   )     

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2
        
        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
    
    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)
        
        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2
        
        segments1 = input[:,:,:-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:,:,segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)
        
        return segments.contiguous(), rest
        
    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)
        
        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size*2)  # B, N, K, L
        
        input1 = input[:,:,:,:segment_size].contiguous().view(batch_size, dim, -1)[:,:,segment_stride:]
        input2 = input[:,:,:,segment_size:].contiguous().view(batch_size, dim, -1)[:,:,:-segment_stride]
        
        output = input1 + input2
        if rest > 0:
            output = output[:,:,:-rest]
        
        return output.contiguous()  # B, N, T
    
        
    def forward(self, input):
        
        batch_size = input.size(0)
        enc_feature = self.BN(input)
        
        # split the encoder output into overlapped, longer segments
        # this is for faster processing
        # first pad the segments accordingly
        enc_segments, enc_rest = self.split_feature(enc_feature, self.segment_size)  # B, N, L, K
        
        # pass to DPRNN
        output = self.DPRNN(enc_segments).view(batch_size*self.num_spk, self.feature_dim, self.segment_size, -1)  # B, C, N, L, K
        
        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest)
        masks = self.output(output)  # B*C, K, T
        masks = masks.view(batch_size, self.num_spk, self.output_dim, -1)  # B, C, K, T
        
        return masks
    
class DPRNN_TasNet(nn.Module):
    # def __init__(self, enc_dim=256, feature_dim=128, hidden_dim=256, sr=44100, win=2,
    #              layer=6, num_spk=2, segment_size=120):
    #     super(DPRNN_TasNet, self).__init__()
    # def __init__(self, enc_dim=64, feature_dim=64, hidden_dim=128, sr=16000, win=2,
    #          layer=6, num_spk=2, segment_size=100):
    # super(DPRNN_TasNet, self).__init__()
    def __init__(self, enc_dim=256, feature_dim=256, hidden_dim=512, sr=11025, win=10,
                 layer=6, num_spk=2, segment_size=120):
        super(DPRNN_TasNet, self).__init__()
        
        # hyper parameters
        self.num_spk = num_spk

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.win = int(sr*win/1000)
        self.stride = self.win // 2
        
        # input encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        self.norm = nn.GroupNorm(1, self.enc_dim, 1e-8)
        
        # DPRNN separator
        self.separator = DPRNN_base(self.enc_dim, self.feature_dim, self.hidden_dim, 
                                    self.enc_dim, self.num_spk, layer, segment_size)
        
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)
        
        from .synthesizer_net import InnerProd, Bias
        self.synthesizer = InnerProd(fc_dim=self.enc_dim)
        self.synthesizer2 = InnerProd(fc_dim=self.enc_dim)
        # self.synthesizer = Bias()

    def pad_input(self, input, window):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape
        stride = window // 2

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest
        
    def forward(self, input, frame):
        if self.num_spk > 1:
            # padding
            output, rest = self.pad_input(input, self.win)
            batch_size = output.size(0)
            
            # waveform encoder
            enc_output = self.encoder(output.unsqueeze(1))  # B, N, T
            seq_len = enc_output.shape[-1]

            # normalize features
            enc_feature = self.norm(enc_output)

            # separation module
            mask = self.separator(enc_feature)  # B, C, N, T
            output = mask * enc_output.unsqueeze(1)  # B, C, N, T
            
            # waveform decoder
            output = self.decoder(output.view(batch_size*self.num_spk, self.enc_dim, seq_len))  # B*C, 1, L
            output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
            output = output.view(batch_size, self.num_spk, -1)
            
            return output
        else:
            # padding
            output, rest = self.pad_input(input, self.win)
            batch_size = output.size(0)
            
            # waveform encoder
            enc_output = self.encoder(output.unsqueeze(1))  # B, N, T
            enc_output = self.synthesizer.forward_nosum(frame, enc_output, act='relu')
            # print(enc_output.shape, frame.shape)
            
            # print(enc_output.shape)
            seq_len = enc_output.shape[-1]

            # normalize features
            enc_feature = self.norm(enc_output)
            
            # separation module
            mask = self.separator(enc_feature)  # B, C, N, T
            output = mask * enc_output.unsqueeze(1)  # B, C, N, T
            output = output.view(batch_size*self.num_spk, self.enc_dim, seq_len)

            # waveform decoder
            output = self.synthesizer2.forward_nosum(frame, enc_output, act='relu')
            output = self.decoder(output)  # B*C, 1, L
            output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
            output = output.view(batch_size, -1)
            
            return output