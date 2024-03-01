import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, lr_min=0.001, start_factor=0.1, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        lr_min (:obj:`float`, `optional`, defaults to 0.0001):
            The minimum value of the learning rate.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        start_factor (:obj:`float`, `optional`, defaults to 0.1):
            The maximum value of the learning rate scaling factor after the warmup (the lower the value, the harder
            we finetune the model).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return (1 - start_factor) / num_warmup_steps * current_step + start_factor
        else:
            return lr_min + (0.5 * (1 + math.cos(2 * math.pi * (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps) * num_cycles)) * (1 - lr_min))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def plot(lr_scheduler, num_training_steps):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(num_training_steps)
    y = []
    for i in range(num_training_steps):
        y.append(lr_scheduler.get_lr())
        lr_scheduler.step()

    # draw hlines for first and last lr and max lr, label them with their values
    plt.hlines(y[0], 0, num_training_steps, colors='r', linestyles='dashed', label=f'lr={y[0]}')
    plt.hlines(y[-1], 0, num_training_steps, colors='r', linestyles='dashed', label=f'lr={y[-1]}')
    plt.hlines(max(y), 0, num_training_steps, colors='r', linestyles='dashed', label=f'lr={max(y)}')

    plt.plot(x, y)

    plt.legend()

    plt.show()


if __name__ == '__main__':
    _opt = torch.optim.AdamW(torch.nn.ParameterList([torch.nn.Parameter(torch.randn(3, 3))]), lr=0.001)
    _sched = get_cosine_schedule_with_warmup(_opt, 100, 1000, num_cycles=0.5, lr_min=0.001, start_factor=0.1)
    plot(_sched, 1000)
