import math
import os.path

import ml_collections


def make_config(
        devices, strategy, num_workers,
        epochs, grad_clip, batch_size, learning_rate,
        dset, model_arch,
        opt_type, opt_params_dict,
        lr_scheduler_type, lr_scheduler_params, lr_scheduler_interval,
        orthogonal_combination,
        use_prime, use_augmix, use_mix, premix, use_fourier, use_apr,
        in_mix, use_jsd, lambda_,
        use_attack, attack_type,
        run_name, batch_accumulation=1,
        turn_off_norm_weight_decay=False,
        min_str=None, mean_str=None, freq_cut=1, phase_cut=1, granularity=448
):
    config = ml_collections.ConfigDict()

    # training hardware
    config.devices = devices
    config.strategy = strategy
    config.num_workers = num_workers

    # training params
    config.epochs = epochs
    config.grad_clip = grad_clip
    config.batch_accumulation = batch_accumulation
    config.turn_off_norm_weight_decay = turn_off_norm_weight_decay

    config.batch_size = batch_size

    # dataset
    config.dataset = dset
    config.data_dir = os.path.join('.', 'data')

    # model
    config.model = model_arch

    # optimizer
    config.opt = ml_collections.ConfigDict()
    config.opt.type = opt_type
    config.opt.params = ml_collections.ConfigDict(opt_params_dict)
    config.opt.params.lr = learning_rate

    # lr scheduler
    config.lr_scheduler = ml_collections.ConfigDict()
    config.lr_scheduler.type = lr_scheduler_type
    config.lr_scheduler.params = lr_scheduler_params
    config.lr_scheduler.interval = lr_scheduler_interval

    # augmentations
    config.enable_aug = ml_collections.ConfigDict()
    config.enable_aug.use_prime = use_prime
    config.enable_aug.use_augmix = use_augmix
    config.enable_aug.use_mix = use_mix
    config.enable_aug.premix = premix
    config.enable_aug.general_fourier = use_fourier
    config.enable_aug.use_apr = use_apr

    config.orthogonal_combination = orthogonal_combination
    config.in_mix = in_mix
    config.use_jsd = use_jsd
    config.lambda_ = lambda_

    # attack
    config.enable_attack = use_attack

    if config.enable_attack:
        config.attack = ml_collections.ConfigDict()

        assert attack_type in ['afa', 'prime', 'apr']
        config.attack.type = attack_type

    if config.enable_aug.general_fourier or (config.enable_attack and config.attack.type == 'afa'):
        # general fourier
        config.general_fourier = ml_collections.ConfigDict()
        config.general_fourier.min_str = min_str
        config.general_fourier.mean_str = mean_str
        config.general_fourier.freq_cut = freq_cut
        config.general_fourier.phase_cut = phase_cut
        config.general_fourier.granularity = granularity

    config.run_name = run_name if run_name is not None else 'default'

    return config


class ConfigBuilder:
    JSD_PARAMS = {
        'C10': 10.,
        'C100': 1.,
        'IN': 12.,
        'TIN': 10.,
        'IN100': 12.,
    }

    COMMON_PARAMS = {
        'C10': {
            'devices': 1,
            'batch_accumulation': 1,
            'strategy': 'auto',
            'num_workers': 0,
            'epochs': 200,
            'grad_clip': 1.0,
            'turn_off_norm_weight_decay': True,
            'batch_size': 256,
            'learning_rate': 0.1,
            'opt_type': 'SGD',
            'opt_params_dict': {'momentum': 0.9, 'weight_decay': 5e-4, 'nesterov': True},
            'lr_scheduler_type': ['CosineAnnealingLR'],
            'lr_scheduler_params': [
                {'T_max': 200 * 196, 'eta_min': 0.00001}
            ],
            'lr_scheduler_interval': ['step'],
        },
        'C10_mix': {
            'devices': 1,
            'batch_accumulation': 1,
            'strategy': 'auto',
            'num_workers': 0,
            'epochs': 300,
            'grad_clip': 0.5,
            'turn_off_norm_weight_decay': True,
            'batch_size': 256,
            'learning_rate': 0.1,
            'opt_type': 'SGD',
            'opt_params_dict': {'momentum': 0.9, 'weight_decay': 1e-4, 'nesterov': True},
            'lr_scheduler_type': ['CosineAnnealingLR'],
            'lr_scheduler_params': [
                {'T_max': 300 * 196, 'eta_min': 0.00001}
            ],
            'lr_scheduler_interval': ['step'],
        },
        'cct_C10_mix': {
            'devices': 1,
            'batch_accumulation': 1,
            'strategy': 'auto',
            'num_workers': 0,
            'epochs': 300,
            'grad_clip': 0.0,
            'turn_off_norm_weight_decay': True,
            'batch_size': 128,
            'learning_rate': 6e-4,
            'opt_type': 'AdamW',
            'opt_params_dict': {'weight_decay': 6e-2},
            'lr_scheduler_type': ['get_cosine_schedule_with_warmup'],
            'lr_scheduler_params': [
                {
                    'num_warmup_steps': 10 * 391, 'num_training_steps': 300 * 391,
                }
            ],
            'lr_scheduler_interval': ['step'],
        },
        'C100': {
            'devices': 1,
            'batch_accumulation': 1,
            'strategy': 'auto',
            'num_workers': 0,
            'epochs': 100,
            'grad_clip': 0.5,
            'turn_off_norm_weight_decay': True,
            'batch_size': 128,
            'learning_rate': 0.2,
            'opt_type': 'SGD',
            'opt_params_dict': {'momentum': 0.9, 'weight_decay': 5e-4, 'nesterov': True},
            'lr_scheduler_type': ['CosineAnnealingLR'],
            'lr_scheduler_params': [
                {'T_max': 100 * 391, 'eta_min': 0.00001}
            ],
            'lr_scheduler_interval': ['step'],
        },
        'C100_mix': {
            'devices': 1,
            'batch_accumulation': 1,
            'strategy': 'auto',
            'num_workers': 0,
            'epochs': 200,
            'grad_clip': 0.5,
            'turn_off_norm_weight_decay': True,
            'batch_size': 128,
            'learning_rate': 0.2,
            'opt_type': 'SGD',
            'opt_params_dict': {'momentum': 0.9, 'weight_decay': 1e-4, 'nesterov': True},
            'lr_scheduler_type': ['CosineAnnealingLR'],
            'lr_scheduler_params': [
                {'T_max': 200 * 391, 'eta_min': 0.00001}
            ],
            'lr_scheduler_interval': ['step'],
        },
        'cct_C100_mix': {
            'devices': 1,
            'batch_accumulation': 1,
            'strategy': 'auto',
            'num_workers': 0,
            'epochs': 300,
            'grad_clip': 0.0,
            'turn_off_norm_weight_decay': True,
            'batch_size': 128,
            'learning_rate': 6e-4,
            'opt_type': 'AdamW',
            'opt_params_dict': {'weight_decay': 6e-2},
            'lr_scheduler_type': ['get_cosine_schedule_with_warmup'],
            'lr_scheduler_params': [
                {
                    'num_warmup_steps': 10 * 391, 'num_training_steps': 300 * 391,
                }
            ],
            'lr_scheduler_interval': ['step'],
        },
        'TIN': {
            'devices': 1,
            'batch_accumulation': 1,
            'strategy': 'auto',
            'num_workers': 0,
            'epochs': 100,
            'grad_clip': 0.5,
            'turn_off_norm_weight_decay': True,
            'batch_size': 128,
            'learning_rate': 0.2,
            'opt_type': 'SGD',
            'opt_params_dict': {'momentum': 0.9, 'weight_decay': 5e-4, 'nesterov': True},
            'lr_scheduler_type': ['CosineAnnealingLR'],
            'lr_scheduler_params': [
                {
                    'T_max': 100 * 782, 'eta_min': 0.00001
                }
            ],
            'lr_scheduler_interval': ['step'],
        },
        'IN': {
            'devices': 1,
            'batch_accumulation': 4,
            'strategy': 'auto',
            'num_workers': 14,
            'epochs': 90,
            'grad_clip': 1.0,
            'turn_off_norm_weight_decay': True,
            'batch_size': 128,
            'learning_rate': 0.2,
            'opt_type': 'SGD',
            'opt_params_dict': {'momentum': 0.9, 'weight_decay': 1e-4, 'nesterov': False},
            'lr_scheduler_type': [
                'LinearLR',
                'MultiStepLR'
            ],
            'lr_scheduler_params': [
                {'start_factor': 1 / 2, 'end_factor': 1., 'total_iters': 5},
                {'milestones': [30, 60], 'gamma': 0.1}
            ],
            'lr_scheduler_interval': ['epoch', 'epoch'],
        },
        'cct_IN_mix': {
            'devices': 8,
            'batch_accumulation': 1,
            'strategy': 'ddp',
            'num_workers': 14,
            'epochs': 300,
            'grad_clip': 5.0,
            'turn_off_norm_weight_decay': True,
            'batch_size': 128,
            'learning_rate': 5e-4,
            'opt_type': 'AdamW',
            'opt_params_dict': {'weight_decay': 0.05},
            'lr_scheduler_type': ['get_cosine_schedule_with_warmup'],
            'lr_scheduler_params': [
                {
                    'num_warmup_steps': 25 * 5005 // 4, 'num_training_steps': 300 * 5005 // 4,
                }
            ],
            'lr_scheduler_interval': ['step'],
        },
    }

    @classmethod
    def build(
            cls, ds, m, attack, use_prime, use_augmix, use_fourier, use_apr, in_mix, use_jsd,
            use_mix=False, mean_str=None, min_str=None, orth=True, premix='none'
    ):
        _lookup_ds = ds.upper()

        if use_mix:
            _lookup_ds = f'{_lookup_ds}_mix'

        if 'cct' in m or 'vit_lite' in m or 'cvt' in m:
            _lookup_ds = f'cct_{_lookup_ds}'

            if not use_mix:
                raise NotImplementedError('Training setup without using MixUp and CutMix is not supported for compact models')

        config = {**cls.COMMON_PARAMS[_lookup_ds]}

        if use_prime and ds == 'in' and ('cct' in m or 'vit_lite' in m or 'cvt' in m):
            config['batch_accumulation'] = 4
            config['batch_size'] = 64
            config['grad_clip'] = 0.5

        if use_prime and ds == 'in' and not ('cct' in m or 'vit_lite' in m or 'cvt' in m):
            config['grad_clip'] = 0.5

        config['dset'] = ds
        config['model_arch'] = m

        config['use_attack'] = attack != 'none' and attack is not None
        config['attack_type'] = attack

        config['use_prime'] = use_prime
        config['use_augmix'] = use_augmix
        config['use_mix'] = use_mix
        config['premix'] = premix
        config['use_fourier'] = use_fourier
        config['use_apr'] = use_apr

        config['in_mix'] = in_mix
        config['use_jsd'] = use_jsd
        config['lambda_'] = cls.JSD_PARAMS[ds.upper()]

        config['orthogonal_combination'] = orth

        config['min_str'] = min_str
        config['mean_str'] = mean_str

        aug_name = 'prime' if use_prime else 'augmix' if use_augmix else 'afa' if use_fourier else 'apr' if use_apr else 'none'
        aug_name = f'{aug_name}-comb' if not orth else aug_name
        aug_name = f'mix-{aug_name}' if use_mix else aug_name
        aug_name = f'{premix}-{aug_name}' if premix != 'none' else aug_name
        aug_name = f'in-mix-{aug_name}' if in_mix else aug_name

        m_n = ''.join(m.split('_'))

        if min_str or mean_str:
            config['run_name'] = f'{aug_name}_{attack}-{"jsd" if use_jsd else "ce"}-{int(min_str)}-{int(mean_str)}_{m_n}_{ds}'
        else:
            config['run_name'] = f'{aug_name}_{attack}-{"jsd" if use_jsd else "ce"}_{m_n}_{ds}'

        config['freq_cut'] = 1
        config['phase_cut'] = 1
        config['granularity'] = 448

        return make_config(**config)
