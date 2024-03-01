_DATASET = {}

_C_DATASET = {}

_C_BAR_DATASET = {}

_P_DATASET = {}


def register_dataset(cls=None, *, dataset=None, is_c=False, is_c_bar=False, is_p=False):
    """
    A decorator for registering model classes.
    """

    def _register(dataset_class):
        if (not is_c) and (not is_c_bar) and (not is_p):
            if dataset not in _DATASET:
                _DATASET[dataset] = dataset_class
            return dataset_class
        elif is_c:
            if dataset not in _C_DATASET:
                _C_DATASET[dataset] = dataset_class
            return dataset_class
        elif is_c_bar:
            if dataset not in _C_BAR_DATASET:
                _C_BAR_DATASET[dataset] = dataset_class
            return dataset_class
        elif is_p:
            if dataset not in _P_DATASET:
                _P_DATASET[dataset] = dataset_class
            return dataset_class

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_dataset(dataset):
    return _DATASET[dataset]


def get_c_dataset(dataset):
    return _C_DATASET[dataset]
