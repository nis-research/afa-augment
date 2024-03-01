_MODELS = {}


def register_model(cls=None, *, dataset=None, name=None):
    """
    A decorator for registering model classes.
    """

    def _register(model_class):
        if dataset in _MODELS and name in _MODELS[dataset]:
            return model_class
            # raise ValueError(f'Already registered model with name: {name}')
        if dataset not in _MODELS:
            _MODELS[dataset] = {}
        _MODELS[dataset][name] = model_class
        return model_class

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(dataset, name):
    if 'c' in dataset:
        _dataset = 'c'
    elif 'tin' in dataset:
        _dataset = 'tin'
    elif 'in' in dataset or 'rn224' in dataset:
        _dataset = 'in'
    else:
        raise NotImplementedError(f'Unknown dataset: {dataset}')

    if _dataset not in _MODELS:
        raise NotImplementedError(f'Unknown dataset shortform: {_dataset}')

    if name not in _MODELS[_dataset]:
        raise NotImplementedError(f'Unknown model name: {name}')

    return _MODELS[_dataset][name]


def benchmark_model(model, input_shape):
    import torch
    from thop import profile

    in_img = torch.randn(1, *input_shape)
    flops, params = profile(model, inputs=(in_img,), verbose=True)
    out_res = model(in_img)
    return f'FLOPs: {flops:.4f}, Params: {params:.4f}, Output: {out_res.shape}'
