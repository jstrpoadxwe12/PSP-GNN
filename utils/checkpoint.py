from typing import Any, Dict

import numpy as np
import torch


class CheckPoint:
    mode_dict = {"min": torch.lt, "max": torch.gt}
    order_dict = {"min": "<", "max": ">"}

    def __init__(
            self,
            monitor: str,
            mode: str = "min",
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode

        if self.mode not in self.mode_dict:
            raise f"`mode` can be {', '.join(self.mode_dict.keys())}, got {self.mode}"

        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    @property
    def monitor_op(self):
        return self.mode_dict[self.mode]

    def __call__(self, metrics: Dict[str, Any]) -> bool:
        current = metrics[self.monitor].squeeze()
        is_best_model = False
        if self.monitor_op(current, self.best_score.to(current.device)):
            is_best_model = True
            self.best_score = current

        return is_best_model


