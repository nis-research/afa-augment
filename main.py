import os
import warnings

from functools import partial

import torch
import wandb
import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from project.trainers import get_module_class
from project.dsets import get_dataset, get_c_dataset
from project.models.image_classification import get_model
from utils import MyWandBLogger, get_standard_transforms, build_augmentations, make_attack, get_best_ckpt

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Casting complex values to real discards.*")


def main(config, weights=None):
    run_id = None if 'run_id' not in config else config.run_id

    strategy = config.strategy
    devices = config.devices

    # create a PyTorch Lightning trainer with the generation callback
    logger = MyWandBLogger(
        name=config.run_name, project=config.project, log_model=True, id=run_id, allow_val_change=True
    )
    logger.experiment
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=wandb.run.dir,
                filename='model_best',
                monitor='val_acc', mode='max', save_top_k=1, save_last=True,
            ),
            LearningRateMonitor(logging_interval='step', log_momentum=True),
        ],
        accelerator="gpu",
        precision="32-true",
        strategy=strategy,
        devices=devices,
        max_epochs=config.epochs,
        num_sanity_val_steps=0,
        gradient_clip_val=config.grad_clip,
        accumulate_grad_batches=config.batch_accumulation,
        benchmark=True,
    )

    wandb.config.update(config.to_dict())

    dataset_class = get_dataset(config.dataset)
    img_sz = dataset_class.image_size
    num_classes = dataset_class.num_classes

    # dataset
    normalise_transform = T.Normalize(mean=dataset_class.mean, std=dataset_class.std)

    test_transform, train_transform = get_standard_transforms(config.dataset, img_sz, config.enable_aug.premix)
    using_wrapper = config.enable_aug.use_augmix or config.enable_aug.use_augmax

    model_class = get_model(config.dataset, config.model)
    if config.dataset == 'tin':
        model_class = partial(model_class, init_stride=2)
    model = model_class(num_classes=num_classes)

    t_dset, v_dset = [
        dataset_class(root=config.data_dir, train=train, transform=transform)
        for train, transform in [
            (True, None if using_wrapper else train_transform), (False, test_transform)
        ]
    ]

    t_dset, train_aug, val_aug = build_augmentations(t_dset, config, img_sz, train_transform)
    attack = make_attack(config, dataset_class)

    t_dl, v_dl = [
        DataLoader(
            dset, batch_size=config.batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
            persistent_workers=True
        )
        for dset, num_workers, shuffle in [(t_dset, 12, True), (v_dset, 4, False)]
    ]

    # create a training_module
    training_module = get_module_class(config)(
        config=config,
        num_classes=num_classes, model=model,
        train_aug=train_aug, val_aug=val_aug, normalisation=normalise_transform,
        attack=attack,
    )

    pprint(
        {
            'config': config.to_dict(),
            'train_transform': train_transform,
            'test_transform': test_transform,
            'train_aug': train_aug,
            'val_aug': val_aug,
            'normalise_transform': normalise_transform,
            'attack': attack,
            'using_mix': config.enable_aug.use_mix,
        }
    )

    # if run_id is specified this already downloaded the best model checkpoint locally for future use
    # otherwise a new model is being trained and the best model checkpoint will be saved locally and logged to wandb
    if weights:
        ckpt_path = weights
    else:
        ckpt_path = get_best_ckpt(run_id)

    # train the training_module
    trainer.fit(training_module, train_dataloaders=t_dl, val_dataloaders=v_dl, ckpt_path=ckpt_path)
    test_accs = cc_test(training_module, config, test_transform, v_dl, severities=(1, 2, 3, 4, 5))

    # take the average of the test nested test accuracies dictionary except for the clean test accuracy
    avg_test_acc = sum(
        [
            sum([acc for acc in test_accs[corruption].values()]) / len(test_accs[corruption])
            for corruption in test_accs if corruption != 'clean'
        ]
    ) / (len(test_accs) - 1)
    wandb.log({'corr_acc': avg_test_acc})

    wandb.finish()


def cc_test(training_module, config, test_transform, val_loader, severities=(4,)):
    test_accs = {}

    # requires the best model checkpoint to be saved locally already
    ckpt_path = os.path.join(wandb.run.dir, 'model_best.ckpt')

    training_module.load_state_dict(torch.load(ckpt_path)['state_dict'])

    clean_test_log = training_module.manual_test(val_loader, name='clean'.ljust(25))
    my_table = wandb.Table(columns=["corruption", "severity", "accuracy"])

    my_table.add_data("clean", 0, clean_test_log['val_acc'])
    test_accs['clean'] = {
        0: clean_test_log['val_acc']
    }

    c_dataset_class = get_c_dataset(config.dataset)
    for corruption in c_dataset_class.corruptions:
        for severity in severities:
            c_s_dst = c_dataset_class(
                config.data_dir, transform=test_transform, severity=severity, corruption=corruption
            )

            c_s_loader = DataLoader(
                c_s_dst,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )

            c_s_log = training_module.manual_test(c_s_loader, name=f'{corruption}_{severity}'.ljust(25))

            my_table.add_data(corruption, severity, c_s_log['val_acc'])
            test_accs[corruption] = test_accs.get(corruption, {})
            test_accs[corruption][severity] = c_s_log['val_acc']

    wandb.log({'corruption_table': my_table})

    return test_accs


if __name__ == '__main__':
    os.environ['WANDB_MODE'] = 'offline'

    from pprint import pprint
    from config_utils import ConfigBuilder

    _config_params = [
        #  Space for config params as shown in the example snippet
    ]

    for i, _config_param in enumerate(_config_params):
        _config = ConfigBuilder.build(**_config_param)
        _config.num_workers = 12
        _config.project = _config.dataset

        _weights = None

        pprint(_config.to_dict())
        main(_config, _weights)
