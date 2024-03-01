from typing import Optional, Type, Union, List

import torch
import torch.nn as nn

import torchvision.models as tvm

from project.models.image_classification import utils

BN_choices = ['M', 'A']

utils.register_model(
    cls=tvm.resnet18,
    dataset='in', name='rn18'
)

utils.register_model(
    cls=tvm.resnet50,
    dataset='in', name='rn50'
)

utils.register_model(
    cls=tvm.wide_resnet50_2,
    dataset='in', name='wrn50x2'
)

utils.register_model(
    cls=tvm.resnext50_32x4d,
    dataset='in', name='rnxt50'
)


class DualBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(DualBatchNorm2d, self).__init__()
        self.bn = nn.ModuleList([nn.BatchNorm2d(num_features), nn.BatchNorm2d(num_features)])
        self.num_features = num_features
        self.ignore_model_profiling = True

        self.route = 'M'  # route images to main BN or aux BN

    def forward(self, x):
        idx = BN_choices.index(self.route)
        y = self.bn[idx](x)
        return y


class DuBIN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio=0.5):
        super(DuBIN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = DualBatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockDuBIN(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            ibn: str = None
    ) -> None:
        super(BasicBlockDuBIN, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if ibn == 'a':
            self.bn1 = DuBIN(planes)
        else:
            self.bn1 = DualBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = DualBatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckDuBIN(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            ibn: str = None
    ) -> None:
        super(BottleneckDuBIN, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        if ibn == 'a':
            self.bn1 = DuBIN(width)
        else:
            self.bn1 = DualBatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = DualBatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = DualBatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetDuBIN(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlockDuBIN, BottleneckDuBIN]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            ibn_cfg: tuple = ('a', 'a', 'a', None),
            **kwargs
    ) -> None:
        super(ResNetDuBIN, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = DualBatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       ibn=ibn_cfg[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block: Type[Union[BasicBlockDuBIN, BottleneckDuBIN]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, ibn: str = None) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                DualBatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, ibn=ibn)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                      ibn=ibn))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


@utils.register_model(dataset='in', name='rn18_dubin')
class ResNet18DuBIN(ResNetDuBIN):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    def __init__(self, **kwargs):
        super().__init__(BasicBlockDuBIN, [2, 2, 2, 2], **kwargs)

        if 'pretrained' in kwargs and kwargs['pretrained']:
            print('Loading pretrained weights...')
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
            a, b = self.load_state_dict(
                weights,
                strict=False
            )
            print(f'Loaded {len(a)} keys, failed to load {len(b)} keys.')


@utils.register_model(dataset='in', name='rn50_dubin')
class ResNet50DuBIN(ResNetDuBIN):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    def __init__(self, **kwargs):
        super().__init__(BottleneckDuBIN, [3, 4, 6, 3], **kwargs)

        if 'pretrained' in kwargs and kwargs['pretrained']:
            print('Loading pretrained weights...')
            weights = tvm.ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
            a, b = self.load_state_dict(
                weights,
                strict=False
            )
            print(f'Loaded {len(a)} keys, failed to load {len(b)} keys.')


if __name__ == '__main__':
    model_classes = [
        tvm.resnet18, tvm.resnet50,
        ResNet18DuBIN, ResNet50DuBIN
    ]
    for _model_class in model_classes:
        print(f'Checking {_model_class.__name__}...')
        print(utils.benchmark_model(_model_class(num_classes=1000, pretrained=True), (3, 224, 224)))
