from torchvision import transforms as T
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from project.augs import GeneralFourierOnline, GeneralizedPRIMEModule, PRIMEAugModule, AugMixDataset
from project.augs.apr import APR
from project.augs.prime import make_original_prime_aug_config

try:
    import wandb
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run
except ModuleNotFoundError:
    # needed for test mocks, these tests shall be updated
    wandb, Run, RunDisabled = None, None, None


class MyWandBLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        if self._log_model:
            if checkpoint_callback.best_model_path:
                # get the best model path
                best_model_path = checkpoint_callback.best_model_path
                # log the best model
                wandb.save(best_model_path, base_path=wandb.run.dir)


def get_standard_transforms(dataset, img_sz, premix='none'):
    if dataset in ['c10', 'c100', 'tin', 'tin10']:
        train_transform = [
            T.RandomCrop(img_sz, padding=4),
            T.RandomHorizontalFlip(),
        ]

        if premix == 'ta':
            train_transform.append(T.TrivialAugmentWide())

        train_transform.append(T.ToTensor())

        train_transform = T.Compose(train_transform)

        test_transform = T.Compose([
            T.ToTensor(),
        ])
    elif dataset in ['in', 'in100']:
        train_transform = [
            T.RandomResizedCrop(img_sz, antialias=True), T.RandomHorizontalFlip(),
        ]

        if premix == 'ta':
            train_transform.append(T.TrivialAugmentWide())

        train_transform.append(T.ToTensor())

        train_transform = T.Compose(train_transform)

        test_transform = T.Compose([
            T.Resize(256, antialias=True), T.CenterCrop(img_sz),
            T.ToTensor(),
        ])
    else:
        raise NotImplementedError
    return test_transform, train_transform


def build_augmentations(training_dataset, config, image_size, train_transform):
    augmentations = []

    if config.enable_aug.use_prime:
        aug_config = make_original_prime_aug_config(config.dataset)

        if config.in_mix:
            if config.enable_aug.general_fourier:
                _gen_fourier_aug = gen_afa_from_config(config, image_size)
                aug_config.append(_gen_fourier_aug)

        prime_module = GeneralizedPRIMEModule(
            mixture_width=3,
            mixture_depth=-1,
            max_depth=3,
            aug_module=PRIMEAugModule(aug_config),
        )

        augmentations.append(prime_module)

    elif config.enable_aug.use_augmix:

        def make_augmix_op(order):
            def augmax_op(image, *args, **kwargs):
                return order(image)

            return augmax_op

        pipeline = lambda aug: T.Compose([
            T.PILToTensor(), aug, T.ToPILImage(),
        ])

        extra_ops = []
        if config.enable_aug.general_fourier and config.in_mix:
            extra_ops.append(
                make_augmix_op(
                    pipeline(
                        gen_afa_from_config(config, image_size)
                    )
                )
            )

        if config.enable_aug.use_augmix:
            dataset_wrapper = AugMixDataset
        else:
            raise NotImplementedError

        training_dataset = dataset_wrapper(
            training_dataset,
            all_ops=True,
            extra_ops=extra_ops,
            preprocess=train_transform,
            aug_severity=3,
            max_depth=3,
            mixture_width=3,
            mixture_depth=-1,
            no_jsd=not config.use_jsd,
            retain_clean=config.enable_attack,
            img_sz=image_size,
        )
        augmentations = []

    if config.enable_aug.use_apr:
        augmentations.append(
            APR(p=0.6)
        )

    if not config.in_mix:
        if config.enable_aug.general_fourier:
            augmentations.append(
                gen_afa_from_config(config, image_size)
            )

    return training_dataset, T.Compose(augmentations), None


def make_attack(config, dataset_class):
    if config.enable_attack:
        if config.attack.type == 'prime':
            aug_config = make_original_prime_aug_config(config.dataset)

            if config.in_mix:
                if config.enable_aug.general_fourier:
                    _gen_fourier_aug = gen_afa_from_config(config, dataset_class.image_size)
                    aug_config.append(_gen_fourier_aug)

            return GeneralizedPRIMEModule(
                mixture_width=3,
                mixture_depth=-1,
                max_depth=3,
                aug_module=PRIMEAugModule(
                    aug_config
                ),
            )
        elif config.attack.type == 'apr':
            return APR(p=1.0)
        elif config.attack.type == 'afa':
            return gen_afa_from_config(config, dataset_class.image_size)
        else:
            raise NotImplementedError
    else:
        return None


def gen_afa_from_config(config, image_size):
    min_str, mean_str, freq_cut, phase_cut, granularity = (
        config.general_fourier.min_str, config.general_fourier.mean_str,
        config.general_fourier.freq_cut, config.general_fourier.phase_cut,
        config.general_fourier.granularity
    )
    _gen_fourier_aug = GeneralFourierOnline(
        img_size=image_size, groups=range(1, image_size + 1), phases=(0., 1.),
        f_cut=freq_cut, phase_cut=phase_cut, min_str=min_str, mean_str=mean_str,
        granularity=granularity
    )
    return _gen_fourier_aug


def get_best_ckpt(run_id):
    if run_id is not None:
        ckpt_path = wandb.restore('last.ckpt', root=wandb.run.dir, replace=True).name
    else:
        ckpt_path = None
    return ckpt_path
