# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



"""
MIT License

Copyright (c) 2022 Naoki Katsura

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# From https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup



import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CustomScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedule_type: str,
        **kwargs
    ):
        self.schedule_type = schedule_type
        self.kwargs = kwargs
        self.scheduler_func = self.get_scheduler(schedule_type, **kwargs)
        super(CustomScheduler, self).__init__(optimizer)

    def get_scheduler(self, schedule_type: str, **kwargs):
        return get_scheduler(schedule_type, **kwargs)

    def get_lr(self):
        return [self.scheduler_func(self.last_epoch) for _ in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
    def reset(self):
        self.last_epoch=-1

def get_scheduler(schedule_type: str, **kwargs):
    '''
    examples:
        get_schedule('constant', level=0.2))
        get_schedule('linear', max=0.2, hold_steps=100, anneal_steps=300, min=0.1))
        get_schedule('cosine', max=0.2, hold_steps=100, anneal_steps=300, min=0.1))
        
        get_schedule('constant_warmup', min=0.1, warmup_steps=100, max=0.2))
        get_schedule('linear_warmup', min=0.1, warmup_steps=100, max=0.2, hold_steps=50, anneal_steps=300))
        get_schedule('cosine_warmup', min=0.1, warmup_steps=100, max=0.2, hold_steps=50, anneal_steps=300))
    '''
    if schedule_type == 'constant':
        assert len(kwargs) == 1
        return lambda x: float(kwargs['level'])
    elif schedule_type == 'constant_warmup':
        assert len(kwargs) == 3
        eps_min, warmup, eps_max = float(kwargs['min']), int(kwargs['warmup_steps']), float(kwargs['max'])
        if warmup == 0:
            return lambda x: eps_max
        return lambda x: (eps_max-eps_min)/warmup * x + eps_min if x < warmup else eps_max
    elif schedule_type == 'linear':
        assert len(kwargs) == 4
        eps_min, hold_steps, eps_max, anneal_steps = float(kwargs['min']), int(kwargs['hold_steps']), float(kwargs['max']), int(kwargs['anneal_steps'])
        return lambda x: np.clip(eps_max - (eps_max - eps_min) * (x - hold_steps) / anneal_steps, eps_min, eps_max)
    elif schedule_type == 'linear_warmup':
        assert len(kwargs) == 5
        eps_min, warmup, eps_max, hold_steps, anneal_steps = float(kwargs['min']), int(kwargs['warmup_steps']), float(kwargs['max']), int(kwargs['hold_steps']), int(kwargs['anneal_steps'])
        if warmup == 0:
            return lambda x: np.clip(eps_max - (eps_max - eps_min) * (x - hold_steps) / anneal_steps, eps_min, eps_max)
        return lambda x: (eps_max-eps_min)/warmup * x + eps_min if x < warmup else np.clip(eps_max - (eps_max - eps_min) * (x - hold_steps - warmup) / anneal_steps, eps_min, eps_max)
    elif schedule_type == 'cosine':
        assert len(kwargs) == 4
        eps_max, hold_steps, anneal_steps, eps_min= float(kwargs['max']), int(kwargs['hold_steps']), int(kwargs['anneal_steps']), float(kwargs['min'])
        return lambda x: eps_min + (eps_max - eps_min) * 0.5 * (1 + np.cos(np.clip((x - hold_steps) / anneal_steps, 0, 1) * np.pi))
    elif schedule_type == 'cosine_warmup':
        assert len(kwargs) == 5
        eps_min, warmup, eps_max, hold_steps, anneal_steps = float(kwargs['min']), int(kwargs['warmup_steps']), float(kwargs['max']), int(kwargs['hold_steps']), int(kwargs['anneal_steps'])
        if warmup == 0:
            return lambda x: eps_min + (eps_max - eps_min) * 0.5 * (1 + np.cos(np.clip((x - hold_steps-warmup) / anneal_steps, 0, 1) * np.pi))
        return lambda x: (eps_max-eps_min)/warmup * x + eps_min if x < warmup else eps_min + (eps_max - eps_min) * 0.5 * (1 + np.cos(np.clip((x - hold_steps-warmup) / anneal_steps, 0, 1) * np.pi))
    else:
        raise ValueError('Unknown schedule: %s' % schedule_type)


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    # Define the schedulers
    schedulers = [
        ('Constant', get_scheduler('constant', level=0.08)),
        ('Linear', get_scheduler('linear', min=0.016, hold_steps=200, max=0.08, anneal_steps=800)),
        ('Cosine', get_scheduler('cosine', max=0.08, min=0.016, hold_steps=200, anneal_steps=800)),
        ('Constant Warmup', get_scheduler('constant_warmup', min=0.016, warmup_steps=100, max=0.08)),
        ('Linear Warmup', get_scheduler('linear_warmup', max=0.08, min=0.016, warmup_steps=50, hold_steps=250, anneal_steps=700)),
        ('Cosine Warmup', get_scheduler('cosine_warmup', max=0.08, min=0.016, warmup_steps=100, hold_steps=200, anneal_steps=700))
    ]

    # Create the plot
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Scheduler Visualizations', fontsize=16)

    # Generate x values
    x = np.arange(0, 1001)

    # Plot each scheduler
    for i, (name, scheduler) in enumerate(schedulers):
        row = i // 3
        col = i % 3
        
        y = [scheduler(t) for t in x]
        axs[row, col].plot(x, y)
        axs[row, col].set_title(name)
        axs[row, col].set_xlabel('Steps')
        axs[row, col].set_ylabel('Value')
        axs[row, col].grid(True)

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig('scheduler_visualization.png')
    print(f"figure saved to {'scheduler_visualization.png'}")
    
    


import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    optimizer (Optimizer): Wrapped optimizer.
    first_cycle_steps (int): First cycle step size.
    cycle_mult(float): Cycle steps magnification. Default: -1.
    max_lr(float): First cycle's max learning rate. Default: 0.1.
    min_lr(float): Min learning rate. Default: 0.001.
    warmup_steps(int): Linear warmup step size. Default: 0.
    max_lr_decrease_per_cycle(float): Decrease rate of max learning rate by cycle. Default: 1. self.max_lr = self.base_max_lr * (self.max_lr_decrease_per_cycle**self.cycle)
    last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        max_lr_decrease_per_cycle: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.max_lr_decrease_per_cycle = max_lr_decrease_per_cycle  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.max_lr_decrease_per_cycle**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class WarmupReduceLROnPlateau:
    def __init__(self, optimizer, warmup_steps, target_lr, mode, min_lr, patience, factor, threshold):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.scheduler = ReduceLROnPlateau(optimizer, 
                                           mode=mode, 
                                           factor=factor, 
                                            patience=patience, threshold=threshold,
                                            min_lr = min_lr, 
                                            verbose=True)
        self.current_step = 0
        
    def step(self, val_metric):
        if self.current_step < self.warmup_steps:
            # Warmup phase: Increase learning rate to target_lr
            warmup_lr = self.target_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            # After warmup, use ReduceLROnPlateau scheduler
            self.scheduler.step(val_metric)
        
        self.current_step += 1



