import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR


def build_optimizer(model, config):
    name = config['train']['optimizer']
    wd_adam = config['train']['adam_decay']
    wd_adamw = config['train']['adamw_decay']

    # base LRs for Phase A
    initial_lr = getattr(config['train'], "initial_lr", 1e-4)  # new phys heads

    if name == "adam":
        opt_cls = torch.optim.Adam
        wd = wd_adam

    elif name == "adamw":
        opt_cls = torch.optim.AdamW
        wd = wd_adamw
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")

    optimizer = opt_cls(
        [
            {"params": model.parameters(), "lr": initial_lr, "weight_decay": wd},
        ]
    )
    return optimizer


def build_scheduler(config, optimizer):
    warmup_epochs = getattr(config['train'], "warmup_epochs", 1)  # ~1 epoch
    cosine_epochs = int(config['train']['max_epochs']) - warmup_epochs
    eta_min = getattr(config['train'], "eta_min", 1e-6)
    warmup_ratio = getattr(config['train'], "warmup_ratio", 0.1)  # start_factor

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_ratio,  # e.g., 0.1 × group LR → group LR
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=eta_min,
    )

    sequential = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    return {
        "scheduler": sequential,
        "interval": getattr(config['train'], "scheduler_interval", "epoch"),
        "frequency": 1,
    }