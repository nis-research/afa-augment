import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F

from tqdm.auto import tqdm

from project.augs import get_mixup_cutmix
from project.trainers.utils import init_optims_from_config


class Identity(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args):
        if len(args) > 0:
            return x, *args
        else:
            return x


def make_cutmix_mixup(config, num_classes):
    if config.enable_aug.use_mix:
        print("Using Mixup and Cutmix")
        if config.dataset in ['in', 'in100']:
            mix_up_alpha = 0.2
        else:
            if 'cct' in config.model:
                mix_up_alpha = 0.8
            else:
                mix_up_alpha = 1.0
        return get_mixup_cutmix(
            mixup_alpha=mix_up_alpha, cutmix_alpha=1.0, num_categories=num_classes
        )
    else:
        return Identity()


class JSDLoss(torch.nn.Module):

    def __init__(self, lambda_=10.):
        super().__init__()
        self.lambda_ = lambda_
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, logits_clean, logits_aug, logits_adv, target):
        loss_clean = self.criterion(logits_clean, target)
        p_clean, p_aug1, p_aug2 = (
            F.softmax(logits_clean, dim=1),
            F.softmax(logits_adv, dim=1),
            F.softmax(logits_aug, dim=1)
        )
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        loss_cst = self.lambda_ * (
                F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug2, reduction='batchmean')
        ) / 3.

        return loss_clean, loss_cst


class BaseModule(pl.LightningModule):

    def __init__(self, config, num_classes, model, train_aug, val_aug, normalisation, attack=None, **kwargs):
        super().__init__()
        self.config = config

        self.model = model
        self.train_criterion = torch.nn.CrossEntropyLoss()
        self.val_criterion = torch.nn.CrossEntropyLoss()

        self.mixer = make_cutmix_mixup(config, num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

        if train_aug is None:
            train_aug = Identity()
        self.train_aug = train_aug

        if val_aug is None:
            val_aug = Identity()
        self.val_aug = val_aug

        self.normalisation = normalisation

        self.attack = attack

        self.test_accs = []

    def training_step(self, batch, batch_idx):
        x, y = batch

        x, y = self.mixer(self.train_aug(x), y)

        # forward pass
        y_hat = self.model(self.normalisation(x))

        # calculate loss
        loss = self.train_criterion(y_hat, y)

        # log loss
        self.log_train_metrics(loss, y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # forward pass
        y_hat = self.model(self.normalisation(self.val_aug(x)))

        # calculate loss
        loss = self.val_criterion(y_hat, y)

        # log loss
        self.log('val_loss', loss, on_epoch=True, on_step=False, logger=True, sync_dist=True, prog_bar=True)
        self.val_acc(y_hat, y)
        self.log('val_acc', self.val_acc, on_epoch=True, on_step=False, logger=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_end(self):
        print()

    def log_train_metrics(self, loss, y_hat, y):
        self.log(
            f'train_loss', loss,
            on_epoch=True, logger=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        return init_optims_from_config(self.config, self.model)

    def manual_test(self, test_loader=None, name='nunya'):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """

        self.model.to('cuda')
        device = next(self.model.parameters()).device
        self.test_acc.to(device)

        test_loader_wrapped = tqdm(enumerate(test_loader), desc=f'Testing... {name}', total=len(test_loader))

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in test_loader_wrapped:
                data, target = data.to(device), target.to(device)

                data = self.normalisation(data)
                output = self.model(data)

                self.test_acc.update(output, target)

                test_loader_wrapped.set_postfix(test_acc=f'{self.test_acc.compute().item():.3f}')

        log = {
            'val_acc': self.test_acc.compute(),
        }

        self.test_acc.reset()

        return log


class NormalJSDModule(BaseModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_criterion = JSDLoss(lambda_=self.config.lambda_)
        using_augmix = self.config.enable_aug.use_augmix

        if using_augmix:
            print('Using AugMix')
            self.training_step = self.training_step_augmix
        else:
            print('Not using AugMix')
            self.training_step = self.training_step_noaugmix

    def training_step_augmix(self, batch, batch_idx):
        (x1, x2, x3), y = batch
        x2, x3 = self.train_aug(x2), self.train_aug(x3)
        return self.do_rest(x1, x2, x3, y)

    def training_step_noaugmix(self, batch, batch_idx):
        x1, y = batch
        x2, x3 = self.train_aug(x1), self.train_aug(x1)
        return self.do_rest(x1, x2, x3, y)

    def do_rest(self, x1, x2, x3, y):
        # forward pass
        y1 = self.model(self.normalisation(x1))
        y2 = self.model(self.normalisation(x2))
        y3 = self.model(self.normalisation(x3))

        # calculate loss
        loss_clean, loss_cst = self.train_criterion(y1, y2, y3, y)

        # log loss
        self.log_train_metrics(loss_clean, y1, y)
        return loss_clean + loss_cst


class AdvJSDModule(BaseModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_criterion = JSDLoss(lambda_=self.config.lambda_)
        using_augmix = self.config.enable_aug.use_augmix

        if using_augmix:
            print('Using AugMix')
            self.training_step = self.training_step_augmix
        else:
            print('Not using AugMix')
            self.training_step = self.training_step_noaugmix

    def training_step_augmix(self, batch, batch_idx):
        (x1, x2), y = batch

        y1, y2 = self.model(self.normalisation(x1)), self.model(self.normalisation(self.train_aug(x2)))
        return self.do_rest(x1, y1, y2, y)

    def training_step_noaugmix(self, batch, batch_idx):
        x1, y = batch

        x2 = self.train_aug(x1)

        y1, y2 = self.model(self.normalisation(x1)), self.model(self.normalisation(x2))
        return self.do_rest(x1, y1, y2, y)

    def do_rest(self, x1, y1, y2, y):
        self.model.apply(lambda m: setattr(m, 'route', 'A'))
        y3 = self.model(self.normalisation(self.attack(x1)))
        self.model.apply(lambda m: setattr(m, 'route', 'M'))
        loss_clean, loss_cst = self.train_criterion(y1, y2, y3, y)
        self.log_train_metrics(loss_clean, y1, y)
        return loss_clean + loss_cst


class AdvCombinationJSDModule(BaseModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_criterion = JSDLoss(lambda_=self.config.lambda_)
        using_augmix = self.config.enable_aug.use_augmix

        if using_augmix:
            print('Using AugMix')
            self.training_step = self.training_step_augmix
        else:
            print('Not using AugMix')
            self.training_step = self.training_step_noaugmix

    def training_step_augmix(self, batch, batch_idx):
        (x1, x2), y = batch
        x2 = self.train_aug(x2)
        return self.do_rest(x1, x2, y)

    def training_step_noaugmix(self, batch, batch_idx):
        x1, y = batch
        x2 = self.train_aug(x1)
        return self.do_rest(x1, x2, y)

    def do_rest(self, x1, x2, y):
        y1 = self.model(self.normalisation(x1))

        self.model.apply(lambda m: setattr(m, 'route', 'A'))
        y2 = self.model(self.normalisation(x2))
        y3 = self.model(self.normalisation(self.attack(x1)))
        self.model.apply(lambda m: setattr(m, 'route', 'M'))

        loss_clean, loss_cst = self.train_criterion(y1, y2, y3, y)
        self.log_train_metrics(loss_clean, y1, y)
        return loss_clean + loss_cst


class AdvModule(BaseModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        using_augmix = self.config.enable_aug.use_augmix

        if using_augmix:
            print('Using AugMix')
            if self.config.enable_aug.use_mix:
                print('Using Mixup and Cutmix Step')
                self.training_step = self.mixing_training_step_augmix
            else:
                print('Using Normal Step')
                self.training_step = self.training_step_augmix
        else:
            print('Not using AugMix')
            if self.config.enable_aug.use_mix:
                print('Using Mixup and Cutmix Step')
                self.training_step = self.mixing_training_step
            else:
                print('Using Normal Step')
                self.training_step = self.training_step_noaugmix

    def mixing_training_step(self, batch, batch_idx):
        x, y = batch
        x_aug_mix, y_mix = self.mixer(self.train_aug(x), y)
        x_mix, y_adv = self.mixer(x, y)

        y_pred = self.model(self.normalisation(x_aug_mix))
        self.model.apply(lambda m: setattr(m, 'route', 'A'))
        y_adv_pred = self.model(self.normalisation(self.attack(x_mix)))
        self.model.apply(lambda m: setattr(m, 'route', 'M'))

        loss_clean, loss_adv = self.train_criterion(y_pred, y_mix), self.train_criterion(y_adv_pred, y_adv)
        self.log_train_metrics(loss_clean, y_pred, y_mix)
        return (loss_clean + loss_adv) / 2.

    def mixing_training_step_augmix(self, batch, batch_idx):
        (x1, x2), y = batch
        x_aug_mix, y_mix = self.mixer(self.train_aug(x2), y)
        x_adv_mix, y_adv = self.mixer(x1, y)

        y_pred = self.model(self.normalisation(x_aug_mix))
        self.model.apply(lambda m: setattr(m, 'route', 'A'))
        y_adv_pred = self.model(self.normalisation(self.attack(x_adv_mix)))
        self.model.apply(lambda m: setattr(m, 'route', 'M'))

        loss_clean, loss_adv = self.train_criterion(y_pred, y_mix), self.train_criterion(y_adv_pred, y_adv)
        self.log_train_metrics(loss_clean, y_pred, y_mix)
        return (loss_clean + loss_adv) / 2.

    def training_step_augmix(self, batch, batch_idx):
        (x1, x2), y = batch

        y2 = self.model(self.normalisation(self.train_aug(x2)))
        loss = self.do_rest(x1, y2, y)
        return loss

    def training_step_noaugmix(self, batch, batch_idx):
        x1, y = batch
        y2 = self.model(self.normalisation(self.train_aug(x1)))

        loss = self.do_rest(x1, y2, y)
        return loss

    def do_rest(self, x1, y2, y):
        self.model.apply(lambda m: setattr(m, 'route', 'A'))
        y1 = self.model(self.normalisation(self.attack(x1)))
        self.model.apply(lambda m: setattr(m, 'route', 'M'))
        loss_clean, loss_adv = self.train_criterion(y1, y), self.train_criterion(y2, y)
        self.log_train_metrics(loss_clean, y1, y)
        return (loss_clean + loss_adv) / 2.


def get_module_class(config):
    if config.enable_attack:
        if config.use_jsd:
            if config.orthogonal_combination:
                print('Using AdvJSDModule - Orthogonal')
                return AdvJSDModule
            else:
                print('Using AdvJSDModule - Combination')
                return AdvCombinationJSDModule
        else:
            print('Using AdvModule')
            return AdvModule
    else:
        if config.use_jsd:
            print('Using NormalJSDModule')
            return NormalJSDModule
        else:
            print('Using BaseModule')
            return BaseModule
