"""Prepare optimization once and schedule actual accumulation windows."""

from math import ceil

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, LinearLR, OneCycleLR


def make_accelerator():
    return Accelerator(step_scheduler_with_optimizer=False,
                       kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])


def register_runtime_modules(model):
    """Register retained decoder modules so optimization and device transfer see them."""
    for name in ("mod_stitcher_proj_dict", "mod_static_weight_dict"):
        modules = getattr(model, name, None)
        if isinstance(modules, dict):
            setattr(model, name, nn.ModuleDict(modules))
    return model


def prepare_optimization(accelerator, model, optimizer, train_loader, config):
    if accelerator.distributed_type.name not in ("NO", "MULTI_GPU", "MULTI_CPU"):
        raise ValueError("Only ordinary single-device and PyTorch DDP training are supported")
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    updates = config.training.num_epochs * ceil(len(train_loader) / config.optimizer.gradient_accumulation_steps)
    config["optimizer"]["planned_updates"] = updates
    base_optimizer = optimizer.optimizer
    if config.optimizer.scheduler == "cosine":
        scheduler = OneCycleLR(base_optimizer, total_steps=updates, max_lr=config.optimizer.lr,
                               pct_start=config.optimizer.warmup_pct, div_factor=config.optimizer.div_factor,
                               anneal_strategy="cos")
    elif config.optimizer.scheduler == "linear":
        scheduler = LinearLR(base_optimizer, total_iters=updates)
    else:
        scheduler = LambdaLR(base_optimizer, lr_lambda=lambda step: 1.)
    return model, optimizer, train_loader, scheduler
