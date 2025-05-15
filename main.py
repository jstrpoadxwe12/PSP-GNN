import os
from datetime import datetime

import hydra
import numpy as np
import omegaconf
import torch
import wandb
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

from train import Trainer
from utils import dataset_factory, model_factory, params_count, set_seed, use_deterministic

os.environ['NUMEXPR_MAX_THREADS'] = '8'
STARTING_TIME = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
DEVICE = "cuda:1"

CHECK_POINT_SAVE_DIR = "/storage/student1/c_wmy/model_output/model/"


def run_kfold(cfg: DictConfig):
    set_seed(seed=cfg.seed)
    use_deterministic(is_use=cfg.use_deterministic)

    dataset = dataset_factory(cfg)
    labels = np.array(dataset.labels)

    params_count(model_factory(cfg))

    save_dir = os.path.join(CHECK_POINT_SAVE_DIR, STARTING_TIME)
    os.makedirs(save_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=cfg.fold, shuffle=True)

    results = []
    for k_idx, (train_index, test_index) in enumerate(skf.split(dataset, labels)):
        trainer = Trainer(dataset, train_index, test_index, cfg, DEVICE, k_idx)

        test_metric, best_model_state = trainer.train()
        save_path = os.path.join(save_dir, "best_model_" + str(k_idx) + ".pth")
        torch.save(best_model_state, save_path)
        wandb.log(test_metric)
        print(test_metric)
        results.append(test_metric)
        print("*" * 100)
        del trainer


    print(results)
    print("*" * 100)
    results = get_mean_value(results)
    print(results)
    wandb.log(results)
    print("Done!")


def get_mean_value(results):
    test_measure = {}
    for k in results[0].keys():
        values = [cm[k] for cm in results]
        test_measure[k + "_mean"] = np.mean(values)
        test_measure[k + "_std"] = np.std(values)

    return test_measure


def main():
    with hydra.initialize(config_path="."):
        cfg = hydra.compose(config_name="config.yaml")

    group_name = f"{cfg.dataset.name}_{cfg.model.name}"

    run = wandb.init(
        project=cfg.project,
        group=group_name,
        name=STARTING_TIME,
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    run.log_code("./",
                 include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".yaml"))

    print(run.config)
    
    run_kfold(omegaconf.OmegaConf.create(run.config.as_dict()))
    run.finish()


if __name__ == '__main__':
    main()
