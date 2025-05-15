from time import sleep

import torch
import torch.nn.functional as F
from math import sqrt
from torch import nn
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import to_dense_batch


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def __norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self.__norm(x.float()).type_as(x)
        return output * self.weight


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, emb_dim)
        self.norm = RMSNorm(dim=emb_dim)

    def forward(self, batch):
        batch.x = self.norm(self.encoder(batch.x))
        return batch


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, output_dim=32, dropout=0.0):
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.q_linear = nn.Linear(input_dim, output_dim)
        self.v_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim)

        self._norm_fact = 1 / sqrt(output_dim // num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, n, input_dim = x.shape
        assert input_dim == self.input_dim

        # Linear Projections
        query = self.q_linear(x)
        key = self.k_linear(x)
        value = self.v_linear(x)

        # Split into multiple heads
        query = query.reshape(batch_size, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.reshape(batch_size, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.reshape(batch_size, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.permute(0, 1, 3, 2)) * self._norm_fact
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        # Reshape and concatenate attention outputs
        attention_output = attention_output.permute(0, 2, 1, 3).reshape(batch_size, n, -1)
        attention_output = self.dropout(attention_output)

        # return attention_output, attention_weights
        return attention_output


class ConnectionInteractionSensingModule(nn.Module):
    def __init__(self, input_dim, out_dim, dropout, num_heads):
        super().__init__()
        self.skip_connection = nn.Linear(in_features=input_dim, out_features=out_dim)
        self.global_layer = MultiHeadSelfAttention(input_dim=input_dim,
                                                   output_dim=out_dim,
                                                   num_heads=num_heads,
                                                   dropout=dropout)
        self.local_layer = GCNConv(in_channels=input_dim, out_channels=out_dim)
        self.batch_norm = BatchNorm(in_channels=out_dim)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, batch):
        x_dense, mask = to_dense_batch(batch.x, batch.batch)
        x_global = self.global_layer(x_dense)
        x_global = x_global[mask]
        x_local = self.local_layer(batch.x, batch.edge_index, batch.edge_weight)
        x = self.skip_connection(batch.x)

        x = x + x_local + x_global

        x = self.batch_norm(x)
        batch.x = self.dropout(self.act_fn(x))
        return batch


class BernoulliRoiExtraction(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, emb):
        log_score = self.feature_extractor(emb)
        bern_mask = self.sampling(log_score)
        return emb * bern_mask, bern_mask

    def sampling(self, log_score, temp=1):
        if self.training:
            random_noise = torch.empty_like(log_score).uniform_(1e-10, 1 - 1e-10)
            log_random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            bern_mask = ((log_score + log_random_noise) / temp).sigmoid()
        else:
            bern_mask = log_score.sigmoid()
        return bern_mask


class Classifier(nn.Module):
    def __init__(self, node_input_dim, dropout, roi_num):
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Linear(node_input_dim * roi_num, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, batch):
        x, _ = to_dense_batch(batch.x, batch.batch)
        batch_size, _, _ = x.shape
        x = x.reshape(batch_size, -1)

        return self.fcn(x)
