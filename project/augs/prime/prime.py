import torch

from einops import rearrange, repeat
from torch.distributions import Dirichlet, Beta


class PRIMEAugModule(torch.nn.Module):
    def __init__(self, augmentations):
        super().__init__()
        self.augmentations = augmentations
        self.num_transforms = len(augmentations)

    def forward(self, x, mask_t):
        aug_x = torch.zeros_like(x)
        for i in range(self.num_transforms):
            aug_x += self.augmentations[i](x) * mask_t[:, i]
        return aug_x

    def __repr__(self):
        return f'PRIMEAugModule(\n' \
               f'\taugmentations={self.augmentations}\n' \
               f')'


class GeneralizedPRIMEModule(torch.nn.Module):
    def __init__(
            self, aug_module, mixture_width=3,
            mixture_depth=-1, max_depth=3
    ):
        """
        Wrapper to perform PRIME augmentation.

        :param mixture_width: Number of augmentation chains to mix per augmented example
        :param mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]
        """
        super().__init__()
        self.aug_module = aug_module
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth

        self.max_depth = max_depth
        self.depth = self.mixture_depth if self.mixture_depth > 0 else self.max_depth
        self.depth_combos = torch.tril(torch.ones((max_depth, max_depth)))

    @torch.no_grad()
    def forward(self, img):
        return self.aug(img)

    def aug(self, img):
        init_shape = img.shape
        if len(init_shape) == 3:
            img = img.unsqueeze(0)

        dirichlet = Dirichlet(concentration=torch.tensor([1.] * self.mixture_width, device=img.device))
        beta = Beta(concentration1=torch.ones(1, device=img.device, dtype=torch.float32),
                    concentration0=torch.ones(1, device=img.device, dtype=torch.float32))

        ws = dirichlet.sample([img.shape[0]])
        m = beta.sample([img.shape[0]])[..., None, None]

        img_repeat = repeat(img, 'b c h w -> m b c h w', m=self.mixture_width)
        img_repeat = rearrange(img_repeat, 'm b c h w -> (m b) c h w')

        trans_combos = torch.eye(self.aug_module.num_transforms, device=img_repeat.device)
        depth_mask = torch.zeros(img_repeat.shape[0], self.max_depth, 1, 1, 1, device=img_repeat.device)
        trans_mask = torch.zeros(img_repeat.shape[0], self.aug_module.num_transforms, 1, 1, 1, device=img_repeat.device)

        depth_idx = torch.randint(0, len(self.depth_combos), size=(img_repeat.shape[0],))
        depth_mask.data[:, :, 0, 0, 0] = self.depth_combos[depth_idx]

        image_aug = img_repeat.clone()

        for d in range(self.depth):
            trans_idx = torch.randint(0, len(trans_combos), size=(img_repeat.shape[0],))
            trans_mask.data[:, :, 0, 0, 0] = trans_combos[trans_idx]

            image_aug.data = depth_mask[:, d] * self.aug_module(image_aug, trans_mask) + (
                    1. - depth_mask[:, d]) * image_aug

        image_aug = rearrange(image_aug, '(m b) c h w -> m b c h w', m=self.mixture_width)

        mix = torch.einsum('bm, mbchw -> bchw', ws, image_aug)
        mixed = (1. - m) * img + m * mix

        return mixed.reshape(init_shape)

    def __repr__(self):
        return (f'GeneralizedPRIMEModule('
                f'mixture_width={self.mixture_width}, '
                f'mixture_depth={self.mixture_depth}, '
                f'max_depth={self.max_depth}, '
                f'aug_module={self.aug_module.__repr__()})')

    def __str__(self):
        return self.__repr__()
