import torchvision.transforms as transforms

from project.augs.mix.cutmix import RandomCutMix
from project.augs.mix.mixup import RandomMixUp


def get_mixup_cutmix(*, mixup_alpha, cutmix_alpha, num_categories):
    mixup_cutmix = []

    if mixup_alpha > 0:
        mixup_cutmix.append(
            RandomMixUp(num_classes=num_categories, p=1.0, alpha=mixup_alpha)
        )

    if cutmix_alpha > 0:
        mixup_cutmix.append(
            RandomCutMix(num_classes=num_categories, p=1.0, alpha=cutmix_alpha)
        )

    if not mixup_cutmix:
        return None

    return transforms.RandomChoice(mixup_cutmix)
