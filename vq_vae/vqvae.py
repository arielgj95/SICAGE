'''
The architecture is an adaption of https://github.com/YoungSeng/QPGesture/tree/master/codebook
It was adapted to work with TED4CL motion data; a different downs_t (down-sampling rate) hyperparameter was used
and some other hyperparamters are different,
as
'''
import numpy as np
import torch as t
import torch.nn as nn

from .encdec import Encoder, Decoder, assert_shape
from .bottleneck_net import NoBottleneck, Bottleneck
from .utils.logger import average_metrics

#import sys
#[sys.path.append(i) for i in ['.', '..']]
#from .configs.parse_args import parse_args


#mydevice = t.device('cuda:' + args.gpu)

def dont_update(params):
    for param in params:
        param.requires_grad = False

def update(params):
    for param in params:
        param.requires_grad = True

def calculate_strides(strides, downs): #stride = [2,], down = [2,]
    #[2^2 = 4]
    return [stride ** down for stride, down in zip(strides, downs)]

def _loss_fn(x_target, x_pred):
    return t.mean(t.abs(x_pred - x_target))


def _conv1d_out_len(L_in: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
    # PyTorch Conv1d: floor((L + 2p - d*(k-1) - 1)/s + 1)
    return int((L_in + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1)


def _convtranspose1d_base_out_len(L_in: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
    # PyTorch ConvTranspose1d without output_padding:
    # (L - 1)*s - 2p + d*(k-1) + 1
    return int((L_in - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1)


def _compute_output_paddings_for_level(L_in: int, down_t: int, stride_t: int, kernel_t: int, pad_t: int) -> list:
    """Compute the required output_padding for each upsampling stage (within one level).

    output_padding must be in [0, stride_t-1].
    """
    lengths = [int(L_in)]
    L = int(L_in)
    for _ in range(down_t):
        L = _conv1d_out_len(L, kernel=kernel_t, stride=stride_t, padding=pad_t, dilation=1)
        lengths.append(L)

    paddings = []
    for step in range(down_t):
        L_after = lengths[-1 - step]
        L_before = lengths[-2 - step]
        base = _convtranspose1d_base_out_len(L_after, kernel=kernel_t, stride=stride_t, padding=pad_t, dilation=1)
        out_pad = int(L_before - base)
        if out_pad < 0 or out_pad >= stride_t:
            raise ValueError(
                f"Cannot invert Conv1d shape at this level: L_before={L_before}, L_after={L_after}, "
                f"stride={stride_t}, kernel={kernel_t}, pad={pad_t} gives output_padding={out_pad}"
            )
        paddings.append(out_pad)
    return paddings

class VQVAE(nn.Module):
    def __init__(self, hps, input_dim=72):
        super().__init__()
        self.hps = hps #config.vqvae
        #75 frames x 54 (9 joint x 6 rot coordinates)
        input_shape = (hps.sample_length, input_dim)

        kernel_t = int(getattr(hps, "kernel_t", 5))
        pad_t = int(getattr(hps, "pad_t", 1))

        levels = hps.levels #1
        downs_t = hps.downs_t  #[2,]
        strides_t = hps.strides_t #[2,]
        emb_width = hps.emb_width #512 , size of input embedding
        l_bins = hps.l_bins #512 , n codebooks
        mu = hps.l_mu #0.99 (used for moving average in learning)
        commit = hps.commit #0.02, how much importance we want to give to commit loss
        # spectral = hps.spectral
        # multispectral = hps.multispectral
        multipliers = hps.hvqvae_multipliers  #[1,]
        use_bottleneck = hps.use_bottleneck #True
        if use_bottleneck:
            print('We use bottleneck!')
        else:
            print('We do not use bottleneck!')
        if not hasattr(hps, 'dilation_cycle'):
            hps.dilation_cycle = None
        # width: 512, depth: 3, m_conv : 1.0, dilation_growth_rate : 3,
        # dilatation_cycle: None,  reverse_decoder_dilation: True

        block_kwargs = dict(width=hps.width, depth=hps.depth, m_conv=hps.m_conv, \
                            dilation_growth_rate=hps.dilation_growth_rate, \
                            dilation_cycle=hps.dilation_cycle, \
                            reverse_decoder_dilation = hps.vqvae_reverse_decoder_dilation,
                            kernel_t = kernel_t,
                            pad_t = pad_t)

        self.sample_length = input_shape[0] #75
        x_shape, x_channels = input_shape[:-1], input_shape[-1] #75, 54
        self.x_shape = x_shape
        # strides_t = [2] and downs_t = 2, so 2^2 = 4 and downsamples = [4],
        # strides_t are the strides applied in filter, down_t the # of levels
        # If downs_t: [1, 1] and  strides_t: [5, 3], then downsamples first 5 at first level (75 -> 15) then 3 (15 -> 3)
        self.downsamples = calculate_strides(strides_t, downs_t) #downsamples = [4]. the number of frames must be multiple of 4
        self.hop_lengths = np.cumprod(self.downsamples) #array([4]). It creates an array of cumulative prod,e.g. [2,3,4] becomes [2,6,24]
        #self.z_shapes = [(x_shape[0] // self.hop_lengths[level],) for level in range(levels)] #60//4 = 15 (30/4 = 7 spare 2).

        # z_shapes depends on n_frames.
        # since we have 1 level, self.hop_lengths[level] is 4, so z_shapes = [4]
        self.levels = levels

        if multipliers is None:
            self.multipliers = [1] * levels #[1,]
        else:
            assert len(multipliers) == levels, "Invalid number of multipliers"
            self.multipliers = multipliers #[1,]

        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs

        # Compute per-level output_paddings
        enc_kernel_t = kernel_t
        enc_pad_t = pad_t

        level_input_lengths = []
        level_output_lengths = []
        L = int(self.sample_length)
        for level in range(levels):
            level_input_lengths.append(L)
            # advance length by this level's downsamples to get next level's input length
            for _ in range(downs_t[level]):
                L = _conv1d_out_len(L, kernel=enc_kernel_t, stride=strides_t[level], padding=enc_pad_t, dilation=1)
            level_output_lengths.append(L)
        self.z_shapes = [(int(L),) for L in level_output_lengths]

        output_paddings_by_level = {}
        for level in range(levels):
            k = strides_t[level] if enc_kernel_t is None else enc_kernel_t
            output_paddings_by_level[level] = _compute_output_paddings_for_level(
                L_in=level_input_lengths[level],
                down_t=downs_t[level],
                stride_t=strides_t[level],
                kernel_t=enc_kernel_t,
                pad_t=enc_pad_t,
            )


        # Encoder and decoder have 1 level called. x_channels = 54. Since downs and strides have one element, doing downs[:2] still takes that element
        def encoder(level):
            return Encoder(
                x_channels,
                emb_width,
                level + 1,
                downs_t[: level + 1],
                strides_t[: level + 1],
                **_block_kwargs(level),
            )

        def decoder(level):
            # Pass output paddings for the *current* level block only.
            # Decoder will create one DecoderConvBock per level internally.
            # We therefore pass output_paddings through block_kwargs and let encdec.py
            # strip it out for the Encoder and consume it for DecoderConvBock.
            bk = _block_kwargs(level)
            bk["output_paddings_by_level"] = output_paddings_by_level  # stored for Decoder to pick the right one
            return Decoder(
                x_channels,
                emb_width,
                level + 1,
                downs_t[: level + 1],
                strides_t[: level + 1],
                **bk,
            )

        self.encoders = nn.ModuleList([encoder(level) for level in range(levels)]) #in default params we have only one level, i.e. encoder block and one decoder block
        self.decoders = nn.ModuleList([decoder(level) for level in range(levels)])


        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels)     # 512, 512, 0.99, 1
        else:
            self.bottleneck = NoBottleneck(levels) #no commit losses, just copies the z of encoder to decoder

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.reg = hps.reg if hasattr(hps, 'reg') else 0 #1 -> motion regularization
        self.acc = hps.acc if hasattr(hps, 'acc') else 0 #1 -> motion acceleration
        self.vel = hps.vel if hasattr(hps, 'vel') else 0 #1 -> motion velocity
        if self.reg == 0:
            print('No motion regularization!')
        # self.spectral = spectral
        # self.multispectral = multispectral

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3 #
        x = x.permute(0,2,1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0,2,1)
        return x

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self._decode(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return t.cat(x_outs, dim=0)

    def _decode(self, zs, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantised) == end_level - start_level

        decoder = self.decoders[end_level - 1]
        x_out = decoder(xs_quantised, all_levels=(len(xs_quantised) > 1))
        x_out = self.postprocess(x_out)
        return x_out

    def _encode(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
        return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x_chunks = t.chunk(x, bs_chunks, dim=0) #split x in chunks
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)] #concatenate
        return zs

    def encode_for_net(self, x, start_level=0, end_level=None, bs_chunks=1, bottleneck=False):
        """Return encoder features either before or after quantisation.

        - bottleneck=False: returns a list of tensors (one per level) with shape (N, C, Tz).
        - bottleneck=True: returns a list (chunks) of list (levels) of tensors, so that
          callers that do x[0][0] still work when bs_chunks==1.
        """
        if end_level is None:
            end_level = self.levels

        x_chunks = t.chunk(x, bs_chunks, dim=0) #split x in chunks

        if not bottleneck: # we don't quantise
            # Collect per-level tensors across chunks and concatenate.
            per_level = None
            for x_i in x_chunks:
                x_in = self.preprocess(x_i)
                xs = []
                for level in range(self.levels):
                    encoder = self.encoders[level]
                    x_out = encoder(x_in)
                    xs.append(x_out[-1])
                if per_level is None:
                    per_level = [[x] for x in xs]
                else:
                    for l, x_l in enumerate(xs):
                        per_level[l].append(x_l)
            return [t.cat(chunks, dim=0) for chunks in per_level]

        # bottleneck=True: quantise then return quantised embeddings (chunked list-of-lists)
        zs = self.encode(x, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks)
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]

        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            xs_quantised = self.bottleneck.decode(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(xs_quantised)
        return x_outs

    def sample(self, n_samples):
        device = next(self.parameters()).device
        zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape), device=device) for z_shape in self.z_shapes]
        return self.decode(zs)

    def forward(self, x):
        metrics = {}

        N = x.shape[0] #batchsize

        # Encode/Decode
        x_in = self.preprocess(x)
        xs = []
        #we have only one level, so one enc and one dec
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        # xs[0]: (32, 512, 15) #15 comes from 60//stride ** down (4) = 15 = n_codes.
        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs) #from here we obtain  (batch_size,n_codes)
        #if we do not use bottlneck, we just return xs previously encoded (0 commit_losses, metrics dict always 0), zs=xs

        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level:level+1], all_levels=False)
            assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)

        # Loss
        # def _spectral_loss(x_target, x_out, self.hps):
        #     if hps.use_nonrelative_specloss:
        #         sl = spectral_loss(x_target, x_out, self.hps) / hps.bandwidth['spec']
        #     else:
        #         sl = spectral_convergence(x_target, x_out, self.hps)
        #     sl = t.mean(sl)
        #     return sl

        # def _multispectral_loss(x_target, x_out, self.hps):
        #     sl = multispectral_loss(x_target, x_out, self.hps) / hps.bandwidth['spec']
        #     sl = t.mean(sl)
        #     return sl

        recons_loss = t.zeros((), device=x.device)
        regularization = t.zeros((), device=x.device)
        velocity_loss = t.zeros((), device=x.device)
        acceleration_loss = t.zeros((), device=x.device)
        # spec_loss = t.zeros(()).to(x.device)
        # multispec_loss = t.zeros(()).to(x.device)
        # x_target = audio_postprocess(x.float(), self.hps)
        x_target = x.float()

        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])

            # this_recons_loss = _loss_fn(loss_fn, x_target, x_out, hps) #loss_fn is mse error
            this_recons_loss = _loss_fn(x_target, x_out)
            # this_spec_loss = _spectral_loss(x_target, x_out, hps)
            # this_multispec_loss = _multispectral_loss(x_target, x_out, hps)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            # metrics[f'spectral_loss_l{level + 1}'] = this_spec_loss
            # metrics[f'multispectral_loss_l{level + 1}'] = this_multispec_loss
            recons_loss += this_recons_loss
            # spec_loss += this_spec_loss
            # multispec_loss += this_multispec_loss
            regularization += t.mean((x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1])**2)

            velocity_loss +=  _loss_fn( x_out[:, 1:] - x_out[:, :-1], x_target[:, 1:] - x_target[:, :-1])
            acceleration_loss +=  _loss_fn(x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1], x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1])
        # if not hasattr(self.)
        commit_loss = sum(commit_losses)
        # loss = recons_loss + self.spectral * spec_loss + self.multispectral * multispec_loss + self.commit * commit_loss
        loss = recons_loss + commit_loss * self.commit + self.reg * regularization + self.vel * velocity_loss + self.acc * acceleration_loss

        with t.no_grad():
            # sc = t.mean(spectral_convergence(x_target, x_out, hps))
            # l2_loss = _loss_fn("l2", x_target, x_out, hps)
            l1_loss = _loss_fn(x_target, x_out)
            
            # linf_loss = _loss_fn("linf", x_target, x_out, hps)

        quantiser_metrics = average_metrics(quantiser_metrics)

        metrics.update(dict(
            recons_loss=recons_loss,
            l1_loss=l1_loss,
            commit_loss=commit_loss,
            regularization=regularization,
            velocity_loss=velocity_loss,
            acceleration_loss=acceleration_loss,
            **quantiser_metrics))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        return x_out, loss, metrics


if __name__ == '__main__':
    import yaml
    from pprint import pprint
    from easydict import EasyDict
    from .configs.parse_args import parse_args

    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)

    device = t.device("cuda:" + f"{args.gpu}") if t.cuda.is_available() else t.device("cpu")

    #batch_size, n_frames, n_joints * joint_dimension
    x = t.rand(32, 75, 9 * 6).to(device)
    model = VQVAE(config.VQVAE, 9 * 6)

    model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
    model = model.to(device)
    model = model.train()
    output, loss, metrics = model(x)
    print(loss.item())