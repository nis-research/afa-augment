import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from project.models.image_classification import utils

BN_choices = ['M', 'A']


class DualBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(DualBatchNorm2d, self).__init__()
        self.bn = nn.ModuleList([nn.BatchNorm2d(num_features), nn.BatchNorm2d(num_features)])
        self.num_features = num_features
        self.ignore_model_profiling = True

        self.route = 'M'  # route images to main BN or aux BN

    def forward(self, input):
        idx = BN_choices.index(self.route)
        y = self.bn[idx](input)
        return y


class BasicBlockDuBN(nn.Module):

    def __init__(self, in_planes, mid_planes, out_planes, stride=1):
        super(BasicBlockDuBN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = DualBatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = DualBatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                DualBatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        # print(out.size())
        return out


class ResNetDuBN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init_stride=1):
        super(ResNetDuBN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=init_stride, padding=1, bias=False)
        self.bn1 = DualBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


@utils.register_model(dataset='c', name='rn18_dubn')
@utils.register_model(dataset='tin', name='rn18_dubn')
class ResNet18DuBN(ResNetDuBN):

    def __init__(self, num_classes=10):
        super(ResNet18DuBN, self).__init__(BasicBlockDuBN, [2, 2, 2, 2], num_classes=num_classes, init_stride=1)


class ResNeXtBottleneckDuBN(nn.Module):
    """ResNeXt Bottleneck Block type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)."""
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None, ibn=None):
        super(ResNeXtBottleneckDuBN, self).__init__()

        dim = int(math.floor(planes * (base_width / 64.0)))

        self.conv_reduce = nn.Conv2d(inplanes, dim * cardinality, kernel_size=1, stride=1, padding=0, bias=False)
        if ibn == 'a':
            self.bn_reduce = DuBIN(dim * cardinality)
        else:
            self.bn_reduce = DualBatchNorm2d(dim * cardinality)

        self.conv_conv = nn.Conv2d(dim * cardinality, dim * cardinality,
                                   kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = DualBatchNorm2d(dim * cardinality)

        self.conv_expand = nn.Conv2d(dim * cardinality, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = DualBatchNorm2d(planes * 4)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXtDuBIN(nn.Module):
    """ResNext optimized for the Cifar dataset, as specified in https://arxiv.org/pdf/1611.05431.pdf."""

    def __init__(self, block, depth, cardinality, base_width, num_classes, init_stride=1, ibn_cfg=('a', 'a', None)):
        super(CifarResNeXtDuBIN, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        layer_blocks = (depth - 2) // 9

        self.cardinality = cardinality
        self.base_width = base_width
        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(3, 64, kernel_size=3, stride=init_stride, padding=1, bias=False)
        self.bn_1 = DualBatchNorm2d(64)

        self.inplanes = 64
        self.stage_1 = self._make_layer(block, 64, layer_blocks, 1, ibn=ibn_cfg[0])
        self.stage_2 = self._make_layer(block, 128, layer_blocks, 2, ibn=ibn_cfg[1])
        self.stage_3 = self._make_layer(block, 256, layer_blocks, 2, ibn=ibn_cfg[2])
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                DualBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample, ibn=ibn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, ibn=ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@utils.register_model(dataset='c', name='rnxt29_dubin')
@utils.register_model(dataset='tin', name='rnxt29_dubin')
class ResNeXt29DuBIN(CifarResNeXtDuBIN):

    def __init__(self, num_classes=10, cardinality=4, base_width=32, init_stride=1):
        super(ResNeXt29DuBIN, self).__init__(ResNeXtBottleneckDuBN, 29, cardinality, base_width, num_classes,
                                             init_stride=init_stride)


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


class BasicBlockDuBIN(nn.Module):

    def __init__(self, in_planes, mid_planes, out_planes, stride=1, ibn=None):
        super(BasicBlockDuBIN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if ibn == 'a':
            self.bn1 = DuBIN(mid_planes)
        else:
            self.bn1 = DualBatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = DualBatchNorm2d(out_planes)

        self.IN = nn.InstanceNorm2d(out_planes, affine=True) if ibn == 'b' else None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                DualBatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.IN is not None:
            out = self.IN(out)
        out = F.relu(out)
        # print(out.size())
        return out


class ResNetDuBIN(nn.Module):
    def __init__(self, block, num_blocks, ibn_cfg=('a', 'a', 'a', None), num_classes=10, init_stride=1):
        '''
        For c (32*32) images, init_stride=1, num_classes=10/100;
        For Tiny ImageNet (64*64) images, init_stride=2, num_classes=200;
        See https://github.com/snu-mllab/PuzzleMix/blob/b7a795c1917a075a185aa7ea078bb1453636c2c7/models/prern.py#L65.
        '''
        super(ResNetDuBIN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=init_stride, padding=1, bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = DualBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, ibn=ibn_cfg[3])
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, ibn=None):
        layers = [block(self.in_planes, planes, planes, stride, None if ibn == 'b' else ibn)]
        self.in_planes = planes

        for i in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, planes, 1, None if (ibn == 'b' and i < num_blocks - 1) else ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


@utils.register_model(dataset='c', name='rn18_dubin')
@utils.register_model(dataset='tin', name='rn18_dubin')
class ResNet18DuBIN(ResNetDuBIN):
    def __init__(self, num_classes=10, init_stride=1, ibn_cfg=('a', 'a', 'a', None)):
        super(ResNet18DuBIN, self).__init__(BasicBlockDuBIN, [2, 2, 2, 2], ibn_cfg=ibn_cfg, num_classes=num_classes,
                                            init_stride=init_stride)


class BasicBlockWRNDuBIN(nn.Module):

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, ibn=None):
        super(BasicBlockWRNDuBIN, self).__init__()
        if ibn == 'a':
            self.bn1 = DuBIN(in_planes)
        else:
            self.bn1 = DualBatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = DualBatchNorm2d(out_planes)

        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.is_in_equal_out = (in_planes == out_planes)
        self.conv_shortcut = (not self.is_in_equal_out) and nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                                                      stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.is_in_equal_out:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.is_in_equal_out:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        if not self.is_in_equal_out:
            return torch.add(self.conv_shortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlockWRNDuBIN(nn.Module):
    """Layer container for blocks."""

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, ibn=None):
        super(NetworkBlockWRNDuBIN, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate, ibn)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, ibn):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate=drop_rate,
                      ibn=ibn)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetDuBIN(nn.Module):

    def __init__(self, depth, num_classes=10, widen_factor=1, drop_rate=0.0, init_stride=1):
        super(WideResNetDuBIN, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlockWRNDuBIN

        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=init_stride, padding=1, bias=False)

        self.block1 = NetworkBlockWRNDuBIN(n, n_channels[0], n_channels[1], block, 1, drop_rate, ibn='a')
        self.block2 = NetworkBlockWRNDuBIN(n, n_channels[1], n_channels[2], block, 2, drop_rate, ibn='a')
        self.block3 = NetworkBlockWRNDuBIN(n, n_channels[2], n_channels[3], block, 2, drop_rate, ibn=None)

        self.bn1 = DualBatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.n_channels = n_channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.n_channels)
        return self.fc(out)


@utils.register_model(dataset='c', name='wrn_dubin40x2')
@utils.register_model(dataset='tin', name='wrn_dubin40x2')
class WideResNetDuBIN40x2(WideResNetDuBIN):
    def __init__(self, num_classes=10, init_stride=1):
        super(WideResNetDuBIN40x2, self).__init__(40, num_classes=num_classes, widen_factor=2, init_stride=init_stride)


@utils.register_model(dataset='c', name='wrn40x4_dubin')
@utils.register_model(dataset='tin', name='wrn40x4_dubin')
class WideResNetDuBIN40x4(WideResNetDuBIN):
    def __init__(self, num_classes=10, init_stride=1):
        super(WideResNetDuBIN40x4, self).__init__(40, num_classes=num_classes, widen_factor=4, init_stride=init_stride)


if __name__ == '__main__':
    model_classes = [
        WideResNetDuBIN40x2,
        WideResNetDuBIN40x4,
        ResNeXt29DuBIN,
        ResNet18DuBN,
        ResNet18DuBIN
    ]
    for _model_class in model_classes:
        print(f'Checking {_model_class.__name__}...')
        print(utils.benchmark_model(_model_class(num_classes=10), (3, 32, 32)))
