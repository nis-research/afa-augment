from .cifar import ResNet18, ResNeXt29
from .dubn import (
    WideResNetDuBIN40x2,
    WideResNetDuBIN40x4,
    ResNeXt29DuBIN,
    ResNet18DuBN,
    ResNet18DuBIN
)
from .wrn import WideResNet40x2, WideResNet40x4, WideResNet28x10
from .imagenet import ResNet18DuBIN, ResNet50DuBIN
from .compact_transformers import cct_7_3x1_32, cct_14_7x2_224, cvt_7_4_32, vit_lite_7_4_32

from .utils import get_model, _MODELS, benchmark_model
