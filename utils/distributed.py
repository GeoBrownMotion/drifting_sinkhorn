import os
import torch
import torch.distributed as dist
from contextlib import contextmanager


def init_distributed_mode():
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        dist.barrier(device_ids=[local_rank])
    else:
        device = torch.device("cuda")
    return device


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return 0


def get_local_rank():
    if is_dist_avail_and_initialized():
        return int(os.environ["LOCAL_RANK"])
    return 0


def is_main_process():
    return get_rank() == 0


def on_main_process(function):
    def wrapper(*args, **kwargs):
        if is_main_process():
            return function(*args, **kwargs)
    return wrapper


@contextmanager
def main_process_first():
    if not is_main_process():
        wait_for_everyone()
    yield
    if is_main_process():
        wait_for_everyone()


def wait_for_everyone():
    if is_dist_avail_and_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[get_local_rank()])
        else:
            dist.barrier()


def cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def broadcast_tensor(tensor):
    if is_dist_avail_and_initialized():
        dist.broadcast(tensor, src=0)
    return tensor


def reduce_tensor(tensor, op="avg"):
    if is_dist_avail_and_initialized():
        rt = tensor.clone()
        if op == "avg":
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
            rt /= get_world_size()
        elif op == "sum":
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        elif op == "max":
            dist.all_reduce(rt, op=dist.ReduceOp.MAX)
        elif op == "min":
            dist.all_reduce(rt, op=dist.ReduceOp.MIN)
        else:
            raise ValueError(f"Unknown reduce op {op}")
        return rt
    return tensor


def gather_tensor(tensor):
    if is_dist_avail_and_initialized():
        tensor_list = [torch.ones_like(tensor) for _ in range(get_world_size())]
        dist.all_gather(tensor_list, tensor)
        return tensor_list
    return [tensor]


def gather_object(obj):
    if is_dist_avail_and_initialized():
        obj_list = [None for _ in range(get_world_size())]
        dist.all_gather_object(obj_list, obj)
        return obj_list
    return [obj]
