import torch.distributed as _dist
from enum import Enum


class ReduceOp(Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3

    def ToDistOp(self):
        return {
            ReduceOp.SUM: _dist.ReduceOp.SUM,
            ReduceOp.PRODUCT: _dist.ReduceOp.PRODUCT,
            ReduceOp.MIN: _dist.ReduceOp.MIN,
            ReduceOp.MAX: _dist.ReduceOp.MAX,
        }[self]


class _DummyWork:
    def wait(self):
        return None


def is_available() -> bool:
    # Torch distributed may be "available" but not initialized.
    return _dist.is_available() and _dist.is_initialized()


def get_rank() -> int:
    return _dist.get_rank() if is_available() else 0


def get_world_size() -> int:
    return _dist.get_world_size() if is_available() else 1


def barrier(**kwargs):
    if is_available():
        return _dist.barrier(**kwargs)
    return _DummyWork() if kwargs.get("async_op", False) else None


def all_gather(tensor_list, tensor, **kwargs):
    if is_available():
        return _dist.all_gather(tensor_list, tensor, **kwargs)
    tensor_list[0] = tensor
    return _DummyWork() if kwargs.get("async_op", False) else None


def all_reduce(tensor, op: ReduceOp = ReduceOp.SUM, **kwargs):
    if is_available():
        return _dist.all_reduce(tensor, op.ToDistOp(), **kwargs)
    return _DummyWork() if kwargs.get("async_op", False) else None


def reduce(tensor, dst: int, op: ReduceOp = ReduceOp.SUM, **kwargs):
    if is_available():
        return _dist.reduce(tensor, dst, op.ToDistOp(), **kwargs)
    return _DummyWork() if kwargs.get("async_op", False) else None


def broadcast(tensor, src: int, **kwargs):
    if is_available():
        return _dist.broadcast(tensor, src, **kwargs)
    return _DummyWork() if kwargs.get("async_op", False) else None


def init_process_group(backend, init_method=None, **kwargs):
    if _dist.is_available():
        return _dist.init_process_group(backend=backend, init_method=init_method, **kwargs)
    return None