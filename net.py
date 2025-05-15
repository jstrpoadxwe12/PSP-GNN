import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import FeatureEncoder, ConnectionInteractionSensingModule, BernoulliRoiExtraction, Classifier


# 用作 Ablation
class PSPGnn(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if cfg.dataset.atlas == "aal":
            dim_in = 116
        elif cfg.dataset.atlas == "cc200":
            dim_in = 200
        else:
            raise "No args 'atlas'"

        dim_list = [128, 32, 8]

        self.r = cfg.model.r
        self.encoder = FeatureEncoder(input_dim=dim_in + 3, emb_dim=128)
        layers = []
        for i in range(len(dim_list) - 1):
            layers.append(ConnectionInteractionSensingModule(
                input_dim=dim_list[i],
                out_dim=dim_list[i + 1],
                dropout=cfg.model.dropout,
                num_heads=cfg.model.n_heads
            ))
        self.layers = torch.nn.Sequential(*layers)
        self.roi_extraction = BernoulliRoiExtraction(embed_dim=dim_list[-1])
        self.classifier = Classifier(node_input_dim=dim_list[-1],
                                     dropout=cfg.model.dropout,
                                     roi_num=dim_in
                                     )

        self.init_parameters()

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

    def loss_function(self, x, y, mask):
        if self.r == 0:
            bern_loss = 0
        else:
            bern_loss = (mask * torch.log(mask / self.r + 1e-6) + (1 - mask) * torch.log(
                (1 - mask) / (1 - self.r + 1e-6) + 1e-6)).mean()

        # LogitNorm
        t = 1
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        x = torch.div(x, norms) / t

        return F.cross_entropy(x, y, label_smoothing=0.2) + bern_loss

    def forward(self, batch):
        batch = self.encoder(batch)
        batch = self.layers(batch)
        batch.x, mask = self.roi_extraction(batch.x)
        x = self.classifier(batch)
        loss = self.loss_function(x, batch.y, mask)

        return x, loss
