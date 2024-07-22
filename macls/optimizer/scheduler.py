import math
from typing import List


class WarmupCosineSchedulerLR:
    def __init__(
            self,
            optimizer,
            min_lr,
            max_lr,
            warmup_epoch,
            fix_epoch,
            step_per_epoch
    ):
        self.optimizer = optimizer
        assert min_lr <= max_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_step = warmup_epoch * step_per_epoch
        self.fix_step = fix_epoch * step_per_epoch
        self.current_step = 0.0

    def set_lr(self, ):
        new_lr = self.clr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def step(self, step=None):
        if step is not None:
            self.current_step = step
        new_lr = self.set_lr()
        self.current_step += 1
        return new_lr

    def clr(self, step):
        if step < self.warmup_step:
            return self.min_lr + (self.max_lr - self.min_lr) * \
                (step / self.warmup_step)
        elif self.warmup_step <= step < self.fix_step:
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                (1 + math.cos(math.pi * (step - self.warmup_step) /
                              (self.fix_step - self.warmup_step)))
        else:
            return self.min_lr

    def get_last_lr(self) -> List[float]:
        return [self.clr(self.current_step)]
