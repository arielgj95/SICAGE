import math
import torch.nn as nn
from .utils import dist_adapter as dist
from .utils.checkpoint import checkpoint

class ResConvBlock(nn.Module): #same as ResConv1DBlock but here in 2D
    def __init__(self, n_in, n_state):
        super().__init__()

        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_in, n_state, 3, 1, 1), #with m_conv=1 we have the same in channels as n_state. stride = 1, pad = 1, ker = 3
            nn.ReLU(),
            nn.Conv2d(n_state, n_in, 1, 1, 0), #1x1 convolution
        )

    def forward(self, x):
        return x + self.model(x)

class Resnet(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0):
        super().__init__()
        self.model = nn.Sequential(*[ResConvBlock(n_in, int(m_conv * n_in)) for _ in range(n_depth)])

    def forward(self, x):
        return self.model(x)

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, zero_out=False, res_scale=1.0):
        super().__init__()
        padding = dilation #by doing this, we compensate the dilation with padding
        self.model = nn.Sequential( #same as  ResConvBlock
            nn.ReLU(),
            nn.Conv1d(n_in, n_state, 3, 1, padding, dilation),
            nn.ReLU(),
            nn.Conv1d(n_state, n_in, 1, 1, 0,),
        )
        if zero_out: #false by default.
            out = self.model[-1] #last layer of net
            nn.init.zeros_(out.weight) #inits with zeros the weights of the last layer
            nn.init.zeros_(out.bias)  #and also the biases
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.model(x) # + is used to add residual. Scale is used to give more or less importance to current output

class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_dilation=False, checkpoint_res=False):
        super().__init__()
        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle
        #changing the dilation of kernels during training may be useful for different purposes. Search "dilated convolutions"
        #basically: we have larger receptive field, dilated convolutions allow for feature extraction at different scales
        #(contexts changes depending on scale, useful specially in segmentation and audio processing)
        blocks = [ResConv1DBlock(n_in, int(m_conv * n_in),
                                 dilation=dilation_growth_rate ** _get_depth(depth),
                                 zero_out=zero_out,
                                 res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth))
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.checkpoint_res = checkpoint_res
        if self.checkpoint_res == 1:
            if dist.get_rank() == 0: #for distributed systems
                print("Checkpointing convs")
            self.blocks = nn.ModuleList(blocks)
        else:
            self.model = nn.Sequential(*blocks)

    def forward(self, x):
        if self.checkpoint_res == 1:
            for block in self.blocks:
                x = checkpoint(block, (x, ), block.parameters(), True)
            return x
        else:
            return self.model(x)
