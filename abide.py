import numpy as np
import torch
from sklearn.preprocessing import minmax_scale
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from structural_allocation import knn_graph


class Abide(Dataset):
    def __init__(self, cfg):
        super().__init__()

        if cfg.dataset.atlas == "aal":
            raw_path = "./rois_aal.npy"
        elif cfg.dataset.atlas == "cc200":
            raw_path = "./rois_cc200.npy"
        else:
            raise "No args 'atlas'"

        
        raw_data = np.load(raw_path, allow_pickle=True).item()
  
        phenotype = raw_data["Phenotype"]
        connectivity = raw_data["Connectivity"]
        k_list = raw_data["k_list"] 
        coordinate = minmax_scale(raw_data["ROIs_Center"]) 

        self.total_subjects = len(phenotype)
        print("subjects nums:", self.total_subjects)
        self.labels = []
        self.graphs = []

        for index in tqdm(range(self.total_subjects)):
            new_connectivity = connectivity[index] * 0.5 + 0.5

            if cfg.dataset.k == 0:
                knn = k_list[index]
            else:
                knn = cfg.dataset.k

            edge_index = knn_graph(knn, new_connectivity)
            edge_weight = torch.Tensor(new_connectivity[edge_index[0], edge_index[1]])

            x = np.hstack((connectivity[index], coordinate))
            x = torch.Tensor(x)  # Convert to tensor

            phenotype_item = phenotype[index]
            graph = Data(x=x
                         , edge_index=edge_index
                         , edge_weight=edge_weight
                         , y=torch.tensor(phenotype_item["DX_GROUP"] - 1, dtype=torch.long)
                         )

            self.labels.append(graph.y)
            self.graphs.append(graph)

    def len(self):
        return self.total_subjects

    def get(self, index):
        return self.graphs[index]
