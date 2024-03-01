import torch
import numpy as np

from einops import rearrange
from opt_einsum import contract


class GeneralFourierOnline(torch.nn.Module):

    def __init__(
            self, img_size, groups, phases, f_cut=1, phase_cut=1, min_str=0, mean_str=5, granularity=64
    ):
        super().__init__()

        _x = torch.linspace(- img_size / 2, img_size / 2, steps=img_size)
        self._x, self._y = torch.meshgrid(_x, _x, indexing='ij')

        self.groups = groups
        self.num_groups = len(groups)
        self.freqs = [f / img_size for f in groups]

        self.phase_range = phases
        self.num_phases = granularity
        self.phases = - np.pi * np.linspace(phases[0], phases[1], num=granularity)

        self.f_cut = f_cut
        self.phase_cut = phase_cut

        self.min_str = min_str
        self.mean_str = mean_str

        self.eps_scale = img_size / 32

    def sample_f_p(self, b, c, device):
        f_cut = self.f_cut
        p_cut = self.phase_cut

        freqs = torch.tensor(self.freqs, device=device, dtype=torch.float32)
        phases = torch.tensor(self.phases, device=device, dtype=torch.float32)

        f_s = freqs[
            torch.randint(0, self.num_groups, (b, c, f_cut, 1), device=device)
        ]

        p_s = phases[
            torch.randint(0, self.num_phases, (b, c, f_cut, p_cut), device=device)
        ]

        return f_s, p_s, f_cut, p_cut

    def forward(self, x):
        init_shape = x.shape
        if len(x.shape) < 4:
            x = rearrange(x, 'c h w -> () c h w')
        b, c, h, w = x.shape

        freqs, phases, num_f, num_p = self.sample_f_p(b, c, x.device)
        strengths = torch.empty_like(phases).exponential_(1 / self.mean_str) + self.min_str

        return self.apply_fourier_aug(freqs, phases, strengths, x).reshape(init_shape)

    def apply_fourier_aug(self, freqs, phases, strengths, x):
        aug = contract(
            'b c f p, b c f p h w -> b c h w',
            strengths,
            self.gen_planar_waves(freqs, phases, x.device)
        )
        aug *= 1 / (self.f_cut * self.phase_cut)
        return torch.clamp(x + aug, 0, 1)

    def gen_planar_waves(self, freqs, phases, device):
        _x, _y = self._x.to(device), self._y.to(device)
        freqs, phases = rearrange(freqs, 'b c f p -> b c f p () ()'), rearrange(phases, 'b c f p -> b c f p () ()')
        _waves = torch.sin(
            2 * torch.pi * freqs * (
                    _x * torch.cos(phases) + _y * torch.sin(phases)
            ) - torch.pi / 4
        )
        _waves.div_(_waves.norm(dim=(-2, -1), keepdim=True))

        return self.eps_scale * _waves

    def __str__(self):
        return f'GeneralFourierOnline(' \
               f'f={self.groups}, phases={self.phase_range}, ' \
               f'f_cut={self.f_cut}, p_cut={self.phase_cut}' \
               f', min_str={self.min_str}, max_str={self.mean_str}' \
               f')'
