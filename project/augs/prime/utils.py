from project.augs.prime.diffeomorphism import Diffeo
from project.augs.prime.rand_filter import RandomFilter
from project.augs.prime.color_jitter import RandomSmoothColor

CIFAR_PARAMS = {
    'diffeo': {
        'sT': 1., 'rT': 1.,
        'scut': 1., 'rcut': 1.,
        'cutmin': 2, 'cutmax': 100,
        'alpha': 1.0
    },
    'color_jit': {
        'cut': 100, 'T': 0.01, 'freq_bandwidth': None
    },
    'rand_filter': {
        'kernel_size': 3, 'sigma': 4.
    },
}

TIN_PARAMS = {
    'diffeo': {
        'sT': 1., 'rT': 1.,
        'scut': 1., 'rcut': 1.,
        'cutmin': 2, 'cutmax': 100,
        'alpha': 1.0
    },
    'color_jit': {
        'cut': 500, 'T': 0.05, 'freq_bandwidth': 20
    },
    'rand_filter': {
        'kernel_size': 3, 'sigma': 4.
    },
}

IMAGE_NET_PARAMS = {
    'diffeo': {
        'sT': 1., 'rT': 1.,
        'scut': 1., 'rcut': 1.,
        'cutmin': 2, 'cutmax': 500,
        'alpha': 1.0
    },
    'color_jit': {
        'cut': 500, 'T': 0.05, 'freq_bandwidth': 20
    },
    'rand_filter': {
        'kernel_size': 3, 'sigma': 4.
    },
}

ALL_PARAMS = {
    'c': CIFAR_PARAMS,
    'tin': TIN_PARAMS,
    'in': IMAGE_NET_PARAMS,
}


def make_original_prime_aug_config(dataset):
    if 'c' in dataset:
        params = CIFAR_PARAMS
    elif 'tin' in dataset:
        params = TIN_PARAMS
    elif 'in' in dataset or 'rn224' in dataset:
        params = IMAGE_NET_PARAMS
    else:
        raise NotImplementedError

    aug_config = [
        Diffeo(
            **params['diffeo'], stochastic=True
        ),
        RandomSmoothColor(
            **params['color_jit'], stochastic=True
        ),
        RandomFilter(
            kernel_size=3,
            sigma=4., stochastic=True
        )
    ]
    return aug_config
