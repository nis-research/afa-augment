from project.models.image_classification import utils

from torchvision.models import vit_b_16

utils.register_model(
    cls=vit_b_16,
    dataset='in',
    name='vit_b_16',
)


if __name__ == '__main__':
    model_classes = [
        vit_b_16,
    ]
    for _model_class in model_classes:
        print(f'Checking {_model_class.__name__}...')
        print(utils.benchmark_model(_model_class(num_classes=1000, weights=None), (3, 224, 224)))
