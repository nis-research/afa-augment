import os

import torch
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

from project.models.image_classification.imagenet import ResNet18DuBIN, ResNet50DuBIN
from project.models.image_classification.compact_transformers import cct_14_7x2_224

WEIGHTS_DIR = 'weights'
WEIGHTS = {
    'rn18dubin': [
        'augmax',
        'augmix_afa_jsd',
        'augmix_afa_ace',
        'afa_only',
        'prime_afa',
        'ta_afa'
    ],
    'rn18': [
        'baseline',
        'prime',
        'augmix',
        'ta'
    ],
    'rn50': [
        'baseline',
        'augmix_jsd_90_epochs',
        'prime_not_fine_tuned',
        'ta',
        'apr-sp'
    ],
    'rn50dubin': [
        'afa_only',
        'prime_afa_not_fine_tuned',
        'ta_afa',
        'augmix_afa_jsd',
        'augmix_afa_ace',
        'apr-sp_afa'
    ],
    'cct': [
        'baseline',
        'augmix',
        'prime',
        'afa_only',
        'augmix_afa',
        'prime_afa',
        'ta',
        'ta_afa',
        # there is no combination of fourier with Augmix since we did not want to use JSD for cct experiments
    ]
}


class LightningWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args):
        return self.model(*args)


def load_weights(weights_dir, model_name, experiment_name, weights_file_name='model_best.ckpt'):
    """
    Load the weights of a model for the ImageNet dataset
    :param weights_dir: for example: weights
    :param model_name: for example: rn18
    :param experiment_name:  for example: baseline
    :param weights_file_name: for example: model_best.ckpt
    :return: the model with the weights loaded wrapped in a LightningWrapper
    """
    path = os.path.join(weights_dir, model_name, experiment_name, weights_file_name)
    if model_name == 'rn18':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if experiment_name == 'baseline' else None)
    elif model_name == 'rn50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if experiment_name == 'baseline' else None)
    elif model_name == 'rn18dubin':
        model = ResNet18DuBIN()
    elif model_name == 'rn50dubin':
        model = ResNet50DuBIN()
    elif model_name == 'cct':
        model = cct_14_7x2_224()
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    model = LightningWrapper(model)

    if not (model_name in ['rn18', 'rn50'] and experiment_name == 'baseline'):

        # filter the key name of the state dict to remove all model. and module. prefixes
        if experiment_name == 'augmax':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path)['state_dict']

        state_dict = {
            k.replace('module.', '') if 'model.module.' in k else
            k.replace('module', 'model') if 'module.' in k else
            k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)

    return model


if __name__ == '__main__':
    for _model_name in WEIGHTS:
        for _experiment_name in WEIGHTS[_model_name]:
            print(f'Loading {_model_name} {_experiment_name}...')
            _model = load_weights(WEIGHTS_DIR, _model_name, _experiment_name)
