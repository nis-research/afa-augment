import torch.nn as nn

from project.models.image_classification.compact_transformers.utils.transformers import TransformerClassifier
from project.models.image_classification.compact_transformers.utils.tokenizer import Tokenizer

from project.models.image_classification.utils import register_model


class CVT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 kernel_size=16,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 *args, **kwargs):
        super(CVT, self).__init__()
        assert img_size % kernel_size == 0, f"Image size ({img_size}) has to be" \
                                            f"divisible by patch size ({kernel_size})"
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=kernel_size,
                                   padding=0,
                                   max_pool=False,
                                   activation=None,
                                   n_conv_layers=1,
                                   conv_bias=True)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def _cvt(arch, pretrained, progress,
         num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=4, positional_embedding='learnable',
         *args, **kwargs):
    model = CVT(num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                *args, **kwargs)

    assert not pretrained, "Pretrained models not available for CVT"

    return model


def cvt_2(*args, **kwargs):
    return _cvt(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cvt_4(*args, **kwargs):
    return _cvt(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cvt_6(*args, **kwargs):
    return _cvt(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cvt_7(*args, **kwargs):
    return _cvt(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cvt_8(*args, **kwargs):
    return _cvt(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cvt_14(*args, **kwargs):
    return _cvt(num_layers=14, num_heads=6, mlp_ratio=4, embedding_dim=768,
                *args, **kwargs)


@register_model(dataset='c', name='cvt_7_4_32')
def cvt_7_4_32(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return cvt_7('cvt_7_4_32', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model(dataset='in', name='cvt_14_16_224')
def cvt_14_16_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=1000,
                  *args, **kwargs):
    return cvt_14('cvt_14_16_224', pretrained, progress,
                  kernel_size=16,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)
