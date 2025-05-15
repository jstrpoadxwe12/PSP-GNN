import warnings

import torch
import wandb
from torch_geometric.loader import DataLoader
from torchmetrics import classification
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm

from utils import EarlyStopping, CheckPoint, model_factory, optimizers_factory

warnings.filterwarnings("ignore")

MAX_EPOCHS = 300


class Trainer:
    def __init__(self, dataset, train_index, test_index, cfg, device, k_idx=0):
        print(f'The {k_idx} fold')
        
        self.train_dataloader = DataLoader(dataset[train_index],
                                           shuffle=True,
                                           batch_size=cfg.batch_size
                                           , pin_memory=True
                                           , num_workers=1
                                           )
        self.test_dataloader = DataLoader(dataset[test_index],
                                          shuffle=False,
                                          batch_size=512
                                          , pin_memory=True
                                          , num_workers=1
                                          )

       
        self.model = model_factory(cfg).to(device)

        self.optimizer = optimizers_factory(self.model, cfg)

        self.device = device
        self.k_idx = k_idx

        self.early_stopping = EarlyStopping(monitor="loss", mode="min", patience=cfg.patience)
        self.check_point = CheckPoint(monitor="test_auroc", mode="max") # test_accuracy

        self.metrics = {
            # trian‘s metric
            "train_loss": MeanMetric().to(self.device),
            "train_acc": classification.BinaryAccuracy().to(self.device),
            "train_specificity": classification.BinarySpecificity().to(self.device),
            "train_recall": classification.BinaryRecall().to(self.device),
            "train_precision": classification.BinaryPrecision().to(self.device),
            "train_f1_score": classification.BinaryF1Score().to(self.device),
            ###
            "loss": MeanMetric().to(self.device),
            # test's metric
            "test_accuracy": classification.BinaryAccuracy().to(self.device),
            "test_specificity": classification.BinarySpecificity().to(self.device),
            "test_recall": classification.BinaryRecall().to(self.device),
            "test_precision": classification.BinaryPrecision().to(self.device),
            "test_f1_score": classification.BinaryF1Score().to(self.device),
            "test_auroc": classification.BinaryAUROC().to(self.device)
        }

    def reset_metrics(self):
        for item in self.metrics.values():
            item.reset()

    def compute_metrics(self):
        metrics_dict = {}
        for k, v in zip(self.metrics.keys(), self.metrics.values()):
            metrics_dict[k] = v.compute()
        return metrics_dict

    def train(self):
        test_metrics = {}

        best_model_state = self.model.state_dict()
        for epoch in range(MAX_EPOCHS):
            self.reset_metrics()
            # reference 
            self.train_per_epoch()
            # if epoch % 2 == 0: # 1 2 5 10
            self.test_per_epoch()

            metrics_dict = self.compute_metrics()

            metrics_dict_for_wandb = {}
            for k, v in zip(metrics_dict.keys(), metrics_dict.values()):
                metrics_dict_for_wandb[k + "/" + str(self.k_idx)] = v
            wandb.log(metrics_dict_for_wandb)

            if self.check_point(metrics_dict):
                best_model_state = self.model.state_dict()
                for k, v in zip(metrics_dict.keys(), metrics_dict.values()):
                    if "test_" in k:
                        test_metrics[k] = v.cpu().numpy().item()

            if self.early_stopping(metrics_dict):
                break

        return test_metrics, best_model_state

    def train_per_epoch(self):
        self.model.train()
        for batch_idx, data in enumerate(tqdm(self.train_dataloader, leave=False, smoothing=0)):
            data = data.to(self.device)

            out, loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = out.argmax(dim=-1)
            self.metrics["train_loss"].update(loss)
            self.metrics["train_acc"].update(pred, data.y)
            self.metrics["train_specificity"].update(pred, data.y)
            self.metrics["train_recall"].update(pred, data.y)
            self.metrics["train_precision"].update(pred, data.y)
            self.metrics["train_f1_score"].update(pred, data.y)

    def test_per_epoch(self):
        with torch.no_grad():
            self.model.eval()
            for batch_idx, data in enumerate(self.test_dataloader):
                data = data.to(self.device)
                out, loss = self.model(data)

                pred = out.argmax(dim=-1)
                self.metrics["loss"].update(loss)
                self.metrics["test_accuracy"].update(pred, data.y)
                self.metrics["test_specificity"].update(pred, data.y)
                self.metrics["test_recall"].update(pred, data.y)
                self.metrics["test_precision"].update(pred, data.y)
                self.metrics["test_f1_score"].update(pred, data.y)
                self.metrics["test_auroc"].update(torch.nn.functional.softmax(out, dim=-1)[:, 1], data.y.int())

