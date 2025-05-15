from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


class EarlyStopping:
    mode_dict = {"min": torch.lt, "max": torch.gt}
    order_dict = {"min": "<", "max": ">"}

    def __init__(
            self,
            monitor: str,
            min_delta: float = 0.0,
            patience: int = 3,
            verbose: bool = False,
            mode: str = "min",
            check_finite: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.check_finite = check_finite
        self.wait_count = 0

        if self.mode not in self.mode_dict:
            raise f"`mode` can be {', '.join(self.mode_dict.keys())}, got {self.mode}"

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def __call__(self, metrics: Dict[str, Any]) -> bool:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        current = metrics[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)

        if reason and self.verbose:
            print(reason)

        return should_stop

    def _evaluate_stopping_criteria(self, current: Tensor) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )

        return should_stop, reason

    def _improvement_message(self, current: Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_score):
            msg = (
                f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f}"
        return msg
