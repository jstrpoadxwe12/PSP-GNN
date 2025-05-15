import importlib
import torch
from omegaconf import DictConfig



def dataset_factory(cfg: DictConfig):
    class_name = cfg.dataset.name
    module_name = class_name.lower()
    try:
        module = getattr(importlib.import_module(module_name), class_name)
    except:
        raise ValueError(
            f'Invalid Module File Name or Invalid Class Name.{module_name}.{class_name}')

    dataset = module(cfg)
    return dataset


def optimizers_factory(model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    if hasattr(importlib.import_module("torch.optim"), cfg.optimizer.name):
        optimizer_module = getattr(importlib.import_module("torch.optim"), cfg.optimizer.name)
    else:
        raise "No args 'optimizer_name'"
    optimizer = optimizer_module(model.parameters(),
                                 cfg.optimizer.learning_rate,
                                 weight_decay=cfg.optimizer.weight_decay)
    return optimizer


def model_factory(cfg: DictConfig):
    class_name = cfg.model.name
    module_name = "net"
    try:
        module = getattr(importlib.import_module(module_name), class_name)
    except:
        raise ValueError(
            f'Invalid Module File Name or Invalid Class Name.{module_name}.{class_name}')
    model = module(cfg)
    return model
