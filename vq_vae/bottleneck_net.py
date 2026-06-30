import pdb

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from .utils import dist_adapter as dist
import sys
[sys.path.append(i) for i in ['.', '..']]
from .configs.parse_args import parse_args

class BottleneckBlock(nn.Module):
    def __init__(self, k_bins, emb_width, mu):
        super().__init__()
        self.k_bins = k_bins #512
        self.emb_width = emb_width #512
        self.mu = mu #0.99
        self.threshold = 1.0

        # Buffers that follow model.to(device)
        self.register_buffer("k", t.zeros(k_bins, emb_width))
        self.register_buffer("k_sum", t.zeros(k_bins, emb_width))
        self.register_buffer("k_elem", t.zeros(k_bins))
        self.init = False
        self.reset_k()  # creates a matrix 512 x 512 with zeros

    def reset_k(self):
        self.init = False
        self.k.zero_()
        self.k_sum.zero_()
        self.k_elem.zero_()

    def _tile(self, x):
        d, ew = x.shape     # 3840, 512
        if d < self.k_bins: #k_bins = 512
            n_repeats = (self.k_bins + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + t.randn_like(x) * std  #add random noise (uniform distrubution (0,1)*std)
        return x

    def init_k(self, x):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins        # mu=0.99, emb_width=512, k_bins=512
        self.init = True
        # init k_w using random vectors from x
        y = self._tile(x) #add values to x if it is not large enough
        _k_rand = y[t.randperm(y.shape[0])][:k_bins]

        # In distributed training, we broadcast rank-0 initialization so that all
        # ranks start from identical codebook vectors (critical for stable EMA VQ).
        if dist.get_world_size() > 1:
            dist.broadcast(_k_rand, 0)
        # (512, 512), a random permutation of integers from 0 to n - 1 (integers taken from embedding x)
        # dist.broadcast(_k_rand, 0)
        self.k.copy_(_k_rand)
        assert self.k.shape == (k_bins, emb_width) #(512, 512)
        self.k_sum.copy_(self.k)
        self.k_elem.fill_(1.0) #all 512 embedding vectors are filled with ones

    def restore_k(self, num_tokens=None, threshold=1.0):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        self.init = True
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum.copy_(self.k)
        self.k_elem.fill_(1.0)
        if num_tokens is not None:
            expected_usage = num_tokens / self.k_bins
            self.k_elem.data.mul_(expected_usage)
            self.k_sum.data.mul_(expected_usage)
        self.threshold = threshold

    def update_k(self, x, x_l):     # (3840), 512), (3840))
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins        # mu=0.99, emb_width=512, k_bins=512
        with t.no_grad():
            # Calculate new centres

            _k_sum = t.zeros(k_bins, self.emb_width, device=x.device, dtype=x.dtype)  #(512(k_bins), 512(w))
            _k_sum.index_add_(0, x_l, x)  # sum vectors per code
            _k_elem = t.bincount(x_l, minlength=k_bins).to(x.dtype) # # (512(k_bins))


            y = self._tile(x)       # (3840, 512)
            _k_rand = y[t.randperm(y.shape[0])][:k_bins]        # (512, 512)

            # If training is distributed, we aggregate _k_sum and _k_elem across ranks.
            # Without this, each rank learns a different codebook, which is effectively
            # training multiple inconsistent tokenizers in parallel.
            if dist.get_world_size() > 1:
                dist.broadcast(_k_rand, 0)
                dist.all_reduce(_k_sum)
                dist.all_reduce(_k_elem)

            # Update centres
            old_k = self.k.clone()
            #self.k_sum = mu * self.k_sum + (1. - mu) * _k_sum  # w, k_bins #old and new sum
            #self.k_elem = mu * self.k_elem + (1. - mu) * _k_elem  # k_bins
            self.k_sum.mul_(mu).add_(_k_sum, alpha=(1.0 - mu))
            self.k_elem.mul_(mu).add_(_k_elem, alpha=(1.0 - mu))

            usage = (self.k_elem.view(k_bins, 1) >= self.threshold).float()
            new_k = usage * (self.k_sum.view(k_bins, emb_width) / self.k_elem.view(k_bins, 1).clamp_min(1e-8)) \
                    + (1 - usage) * _k_rand
            self.k.copy_(new_k)
            _k_prob = _k_elem / _k_elem.sum().clamp_min(1e-8)  # x_l_onehot.mean(dim=-1)  # prob of each bin
            entropy = -t.sum(_k_prob * t.log(_k_prob + 1e-8))
            used_curr = (_k_elem >= self.threshold).sum()
            usage = usage.sum()
            dk = t.norm(self.k - old_k) / np.sqrt(np.prod(old_k.shape))

        return dict(entropy=entropy,
                    used_curr=used_curr,
                    usage=usage,
                    dk=dk)

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous() # before we had 512, 15 , now 15, 512. First dim, batch size, remains the same
        x = x.view(-1, x.shape[-1])  # x_en = (N * L, w), k_j = (w, k_bins), so now is batch_size x 15, 512
        # the first -1 tells "given that second dim is 512, reshape accordingly the first dimension,
        # while as second position take x.shape[-1] = 512"

        if x.shape[-1] == self.emb_width: #with actual config this is the case. Normalize x
            prenorm = t.norm(x - t.mean(x)) / np.sqrt(np.prod(x.shape))     # np.sqrt - product of array elements over a given axis
        elif x.shape[-1] == 2 * self.emb_width:
            x1, x2 = x[...,:self.emb_width], x[...,self.emb_width:]
            prenorm = (t.norm(x1 - t.mean(x1)) / np.sqrt(np.prod(x1.shape))) + (t.norm(x2 - t.mean(x2)) / np.sqrt(np.prod(x2.shape)))

            # Normalise
            x = x1 + x2
        else:
            raise AssertionError(f"Expected last dim to be (1 or 2) * {self.emb_width}, got {x.shape[-1]}")
        return x, prenorm

    def postprocess(self, x_l, x_d, x_shape):
        # [NT, C] -> NTC -> NCT
        N, T = x_shape
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous() #get back to original shape, i.e. batch_size, 512, n_coding (15)
        x_l = x_l.view(N, T)
        return x_l, x_d

    def quantise(self, x):
        # Calculate latent code x_l
        k_w = self.k.t()        # (512, 512) #k_w represents the combinations between x and the embeddings). .t transforms it to torch tensor
        #this is the distance l2 norm ||a-b|| = a^2 + b^2 - 2 ab. Where a is X (batch_size x n_codes, 512), b is the matrix 512 x 512.
        # n_codes is 60//4 = 15. so if batch = 256, then (256x15,512) = (3840,512) is x.
        # we compute the distance between each x element 1x512 and all the 512 codebooks (with width 512). We then take the index of the codebook with minimum distance
        # so at the end we will have 3840 indexes = 256x15
        distance = t.sum(x ** 2, dim=-1, keepdim=True) - 2 * t.matmul(x, k_w) + t.sum(k_w ** 2, dim=0, keepdim=True)  # (3840(N * L), 512(b))
        min_distance, x_l = t.min(distance, dim=-1)     # (3840), (3840). For each batch_size x n_codes, we find the minimum corrisponding value
        fit = t.mean(min_distance)
        return x_l, fit

    def dequantise(self, x_l):
        x = F.embedding(x_l, self.k)        # self.k: (512, 512) weighted array. x = batch_size x n_codes = 256x15 = 3840.
        return x                            #For each x element containing indexes, we extract the 512x1 vector at index i (from 1 to 512)

    def encode(self, x):
        N, width, T = x.shape

        # Preprocess.
        x, prenorm = self.preprocess(x)

        # Quantise
        x_l, fit = self.quantise(x)

        # Postprocess.
        x_l = x_l.view(N, T)
        return x_l

    def decode(self, x_l):
        N, T = x_l.shape
        width = self.emb_width

        # Dequantise
        x_d = self.dequantise(x_l)

        # Postprocess
        x_d = x_d.view(N, T, width).permute(0, 2, 1).contiguous()
        #x_d has shape batch_size, 512, 15 (or n_codes in general). .contiguous() makes the tensor continuos in memory so that we can apply operations on it
        return x_d

    def forward(self, x, update_k=True):
        N, width, T = x.shape       # batch_size, 512, 15

        # Preprocess
        x, prenorm = self.preprocess(x)     # (3840, 512),

        # Init k if not inited
        if update_k and not self.init:
            self.init_k(x)

        # Quantise and dequantise through bottleneck
        x_l, fit = self.quantise(x)     # (batch_size, n_codes)
        x_d = self.dequantise(x_l)      # (batch_size, n_codes, 512)

        # Update embeddings
        if update_k:
            update_metrics = self.update_k(x, x_l)
        else:
            update_metrics = {}

        # Loss
        #commit_loss = t.norm(x_d.detach() - x) ** 2 / np.prod(x.shape)      # L2 loss -> L1 loss
        commit_loss = F.mse_loss(x, x_d.detach(), reduction="mean")

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_l, x_d = self.postprocess(x_l, x_d, (N,T)) #get back to original position
        return x_l, x_d, commit_loss, dict(fit=fit,
                                           pn=prenorm,
                                           **update_metrics)


class Bottleneck(nn.Module):
    def __init__(self, l_bins, emb_width, mu, levels):
        super().__init__()
        self.levels = levels
        level_block = lambda level: BottleneckBlock(l_bins, emb_width, mu) #512,512,0.99
        self.level_blocks = nn.ModuleList()
        for level in range(self.levels):
            self.level_blocks.append(level_block(level))

    def encode(self, xs):
        zs = [level_block.encode(x) for (level_block, x) in zip(self.level_blocks, xs)]
        return zs

    def decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        xs_quantised = [level_block.decode(z) for (level_block, z) in zip(self.level_blocks[start_level:end_level], zs)]
        return xs_quantised

    def forward(self, xs):
        zs, xs_quantised, commit_losses, metrics = [], [], [], []
        for level in range(self.levels):
            level_block = self.level_blocks[level]
            x = xs[level]
            z, x_quantised, commit_loss, metric = level_block(x, update_k=self.training)
            zs.append(z)
            if not self.training:
                # Be extra paranoid and make sure the encoder weights can't
                # change from straight-through estimator
                x_quantised = x_quantised.detach()
            xs_quantised.append(x_quantised)
            commit_losses.append(commit_loss)
            if self.training:
                metrics.append(metric)
        return zs, xs_quantised, commit_losses, metrics

class NoBottleneckBlock(nn.Module):
    def restore_k(self):
        pass

class NoBottleneck(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.level_blocks = nn.ModuleList()
        self.levels = levels
        for level in range(levels):
            self.level_blocks.append(NoBottleneckBlock())

    def encode(self, xs):
        return xs

    def decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        return zs

    def forward(self, xs):
        device = xs[0].device
        zero = t.zeros((), device=device)
        commit_losses = [zero for _ in range(self.levels)]
        metrics = [dict(entropy=zero, usage=zero, used_curr=zero, pn=zero, dk=zero) for _ in range(self.levels)]
        return xs, xs, commit_losses, metrics

if __name__ == '__main__':
    '''
    python -m models.bottleneck --config configs/sep_vqvae.yaml --train --no_cuda 2 --gpu 2
    '''
    # x = [t.rand(32, 512, 15)]
    # bottleneck = Bottleneck(512, 512, 0.99, 1).to(mydevice)
    # zs, xs_quantised, commit_losses, quantiser_metrics = bottleneck(x)

    x = t.rand(32, 512, 15)
    model = BottleneckBlock(k_bins=512, emb_width=512, mu=0.99)
    zs, xs_quantised, commit_losses, quantiser_metrics = model(x)
