import random

import torch


class APR(torch.nn.Module):
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p

    def __call__(self, x):
        return mix_data(x, prob=self.p)

    def __str__(self):
        return f'APR(p={self.p})'


def mix_data(x, prob=0.6):
    """Returns mixed inputs, pairs of targets, and lambda"""

    p = random.uniform(0, 1)

    if p > prob:
        return x

    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)

    fft_1 = torch.fft.fftn(x, dim=(1, 2, 3))
    abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)

    fft_2 = torch.fft.fftn(x[index, :], dim=(1, 2, 3))
    abs_2, angle_2 = torch.abs(fft_2), torch.angle(fft_2)

    fft_1 = abs_2 * torch.exp(1j * angle_1)

    mixed_x = torch.fft.ifftn(fft_1, dim=(1, 2, 3)).float()

    return mixed_x
