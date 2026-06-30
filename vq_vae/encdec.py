import pdb

import torch as t
import torch.nn as nn
from .resnet import Resnet, Resnet1D
from .utils.torch_utils import assert_shape

class EncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv,
                 dilation_growth_rate=1, dilation_cycle=None, zero_out=False,
                 res_scale=False, kernel_t=5, pad_t=1):
        super().__init__()
        blocks = []

        '''
        # Initial convolutional layer to capture edge information
        initial_conv = nn.Sequential(
            nn.Conv1d(input_emb_width, width, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        '''
        #blocks.append(initial_conv)
        # filter_t = 3, pad_t = 1, so final results is (75 - 5 + 2*1) / 3 + 1 = 25. In case we use parameters that return a float value, in reconstruction we solve with extra padding.
        filter_t, pad_t = int(kernel_t), int(pad_t) ####stride_t * 2, stride_t // 2  #kernel size. stride_t = [2,], so kernel_size = 4, pad_t = 1
        if down_t > 0: #down_t=2
            for i in range(down_t): #we apply convolution down_t times. This reduces the dimension by stride each time
                block = nn.Sequential(
                    #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
                    #nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t, padding_mode='replicate'),
                    # filter t is 4. with strides=2, padding = 1 (bilateral), then shape = ((in-filter_t+2*padding)/stride)+1 =
                    # (in-4+2)/s + 1 = 58/2 + 1= 30 with 512 filters (output_width=n_channels). Each time we have down_t we divide by two the shape
                    nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t),
                    #depth = 3, so we have 3 residual conv1d blocks, each with 3x3 conv followed by 1x1 conv
                    #width = 512, depth = 3, m_conv = 1, dilation_growth_rate = 3, dilation_cycle = None, zero_out = False, res_scale=False
                    #dilation is the spacing between kernel elements, with 1 being no spacing. It expands the receptive field of kernel.
                    #Lout = ((Lin + 2*padding - dilation * (kernel_size-1) - 1) / stride) +1
                    Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out, res_scale),
                )
                blocks.append(block)
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1) #with this we do not change the shape since the filter dimension,3, is compensated by pad=1
            # block = nn.Conv1d(width, output_emb_width, 3, 1, 1, padding_mode='replicate')
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        #at the end shape(x) = batch_size,512,15 (=n_frames // stride_t**down_t)
        res = self.model(x)
        return res #self.model(x)

class DecoderConvBock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None,
                 zero_out=False, res_scale=False, reverse_decoder_dilation=False,
                 checkpoint_res=False, output_paddings=None, kernel_t=5, pad_t=1):
        super().__init__()
        blocks = []
        self.output_paddings = output_paddings
        # Initial transpose convolutional layer to expand embedding
        '''
        initial_deconv = nn.Sequential(
            nn.ConvTranspose1d(output_emb_width, width, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        '''
        #blocks.append(initial_deconv)
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1)) #First layer
        if down_t > 0:
            filter_t, pad_t = int(kernel_t), int(pad_t) #stride_t // 2
            #block = nn.Conv1d(output_emb_width, width, kernel_t, stride_t, pad_t)
            #
            #blocks.append(block)
            for i in range(down_t):
                out_pad = 0
                if self.output_paddings is not None:
                    if isinstance(self.output_paddings, (list, tuple)):
                        out_pad = int(self.output_paddings[i])
                    else:
                        out_pad = int(self.output_paddings)
                    if out_pad < 0 or out_pad >= stride_t:
                        raise ValueError(f"Invalid output_padding={out_pad} for stride_t={stride_t}")
                block = nn.Sequential(
                    Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out=zero_out,
                             res_scale=res_scale, reverse_dilation=reverse_decoder_dilation,
                             checkpoint_res=checkpoint_res),
                    nn.ConvTranspose1d(width, input_emb_width if i == (down_t - 1) else width, filter_t, stride_t,
                                       pad_t, output_padding=out_pad),
                )
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t,
                 strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width # 9 * 6 = 54
        self.output_emb_width = output_emb_width # 512
        self.levels = levels #1
        self.downs_t = downs_t #1
        self.strides_t = strides_t #3

        block_kwargs_copy = dict(**block_kwargs)
        if 'reverse_decoder_dilation' in block_kwargs_copy:
            del block_kwargs_copy['reverse_decoder_dilation']
        if 'output_paddings' in block_kwargs_copy:
            del block_kwargs_copy['output_paddings']
        if 'output_paddings_by_level' in block_kwargs_copy:
            del block_kwargs_copy['output_paddings_by_level']
        level_block = lambda level, down_t, stride_t: EncoderConvBlock(input_emb_width if level == 0 else output_emb_width,
                                                           output_emb_width,
                                                           down_t, stride_t,
                                                           **block_kwargs_copy)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []


        # 64, 32, ...
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            #emb, T = self.output_emb_width, T // (stride_t ** down_t)
            emb = self.output_emb_width
            T = x.shape[-1]
            assert_shape(x, (N, emb, T))
            xs.append(x)
        return xs

class Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t,
                 strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels

        self.downs_t = downs_t

        self.strides_t = strides_t

        output_paddings_by_level = block_kwargs.pop('output_paddings_by_level', None)
        output_paddings = block_kwargs.pop('output_paddings', None)

        def _level_output_paddings(level):
            if output_paddings_by_level is not None:
                return output_paddings_by_level.get(level, None)
            return output_paddings

        level_block = lambda level, down_t, stride_t: DecoderConvBock(output_emb_width,
                                                          output_emb_width,
                                                          down_t, stride_t,
                                                          output_paddings=_level_output_paddings(level),
                                                          **block_kwargs)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))
        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)
        # self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1, padding_mode='replicate')
    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))

        # 32, 64 ...
        iterator = reversed(list(zip(list(range(self.levels)), self.downs_t, self.strides_t)))
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T * (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x)
        return x
