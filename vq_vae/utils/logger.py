import os
import sys
from datetime import date

import torch as t
from tqdm import tqdm

from . import dist_adapter as dist


def def_tqdm(x):
    return tqdm(
        x,
        leave=True,
        file=sys.stdout,
        bar_format="{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )


def get_range(x):
    return def_tqdm(x) if dist.get_rank() == 0 else x

def init_logging(hps, local_rank, rank):
    logdir = f"{hps.local_logdir}/{hps.name}"
    if local_rank == 0:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        with open(os.path.join(logdir, "argv.txt"), "w") as f:
            f.write(hps.argv + "\n")
        print("Logging to", logdir)
    logger = Logger(logdir, rank)
    metrics = Metrics()
    logger.add_text("hps", str(hps))
    return logger, metrics


def get_name(hps):
    name = ""
    for key, value in hps.items():
        name += f"{key}_{value}_"
    return name


def average_metrics(_metrics):
    """Average a list of metric dicts (e.g., per-level quantiser metrics).

    The previous implementation used integer (floor) division, which can silently
    distort diagnostics and any heuristics that depend on them.
    """
    metrics = {}
    for _metric in _metrics:
        for key, val in _metric.items():
            metrics.setdefault(key, []).append(val)

    out = {}
    for key, vals in metrics.items():
        v0 = vals[0]
        if isinstance(v0, t.Tensor):
            out[key] = t.stack([v.float() for v in vals]).mean()
        else:
            out[key] = sum(vals) / float(len(vals))
    return out


class Metrics:
    def __init__(self, device=None):
        # Default to the current CUDA device (if available), otherwise CPU.
        if device is None:
            device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
        self.device = device
        self.sum = {}
        self.n = {}

    def update(self, tag, val, batch):
        # val is average value over batch
        sum_t = t.tensor(val * batch, dtype=t.float32, device=self.device)
        n_t = t.tensor(batch, dtype=t.float32, device=self.device)

        dist.all_reduce(sum_t)
        dist.all_reduce(n_t)

        sum_v = sum_t.item()
        n_v = n_t.item()

        self.sum[tag] = self.sum.get(tag, 0.0) + sum_v
        self.n[tag] = self.n.get(tag, 0.0) + n_v
        return sum_v / max(n_v, 1e-12)

    def avg(self, tag):
        if tag in self.sum:
            return self.sum[tag] / max(self.n[tag], 1e-12)
        return 0.0

    def reset(self):
        self.sum = {}
        self.n = {}


class Logger:
    def __init__(self, logdir, rank):
        if rank == 0:
            from tensorboardX import SummaryWriter

            self.sw = SummaryWriter(f"{logdir}/logs")
        self.iters = 0
        self.rank = rank
        self.works = []
        self.logdir = logdir

    def step(self):
        self.iters += 1

    def flush(self):
        if self.rank == 0:
            self.sw.flush()

    def add_text(self, tag, text):
        if self.rank == 0:
            self.sw.add_text(tag, text, self.iters)

    def add_audios(self, tag, auds, sample_rate=22050, max_len=None, max_log=8):
        if self.rank == 0:
            for i in range(min(len(auds), max_log)):
                if max_len:
                    self.sw.add_audio(
                        f"{i}/{tag}", auds[i][: max_len * sample_rate], self.iters, sample_rate
                    )
                else:
                    self.sw.add_audio(f"{i}/{tag}", auds[i], self.iters, sample_rate)

    def add_audio(self, tag, aud, sample_rate=22050):
        if self.rank == 0:
            self.sw.add_audio(tag, aud, self.iters, sample_rate)

    def add_images(self, tag, img, dataformats="NHWC"):
        if self.rank == 0:
            self.sw.add_images(tag, img, self.iters, dataformats=dataformats)

    def add_image(self, tag, img):
        if self.rank == 0:
            self.sw.add_image(tag, img, self.iters)

    def add_scalar(self, tag, val):
        if self.rank == 0:
            self.sw.add_scalar(tag, val, self.iters)

    def get_range(self, loader):
        if self.rank == 0:
            self.trange = def_tqdm(loader)
        else:
            self.trange = loader
        return enumerate(self.trange)

    def close_range(self):
        if self.rank == 0:
            self.trange.close()

    def set_postfix(self, *args, **kwargs):
        if self.rank == 0:
            self.trange.set_postfix(*args, **kwargs)

    def add_reduce_scalar(self, tag, layer, val):
        if self.iters % 100 == 0:
            with t.no_grad():
                val = val.float().norm() / float(val.numel())
            work = dist.reduce(val, 0, async_op=True)
            self.works.append((tag, layer, val, work))

    def finish_reduce(self):
        for tag, layer, val, work in self.works:
            if work is not None and hasattr(work, "wait"):
                work.wait()
            if self.rank == 0:
                val = val.item() / max(dist.get_world_size(), 1)
                # NOTE: self.lw is not defined in the original code; leaving behavior unchanged.
                if hasattr(self, "lw"):
                    self.lw[layer].add_scalar(tag, val, self.iters)
        self.works = []
