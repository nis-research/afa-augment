import os

import numpy as np

from torchvision.datasets import CIFAR10, CIFAR100, DatasetFolder
from torchvision.datasets import VisionDataset, ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, extract_archive

from project.dsets.utils import register_dataset


@register_dataset(dataset='c10')
class CIFAR10Dataset(VisionDataset):
    mean = (0.4915, 0.4823, 0.4468)
    std = (0.2470, 0.2435, 0.2616)
    num_classes = 10
    image_size = 32

    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        super().__init__(root, transform, target_transform)
        self.dataset = CIFAR10(root, train, transform, target_transform, download)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


@register_dataset(dataset='c100')
class CIFAR100Dataset(VisionDataset):
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    num_classes = 100
    image_size = 32

    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        super().__init__(root, transform, target_transform)
        self.dataset = CIFAR100(root, train, transform, target_transform, download)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


@register_dataset(dataset='c10', is_c=True)
class CIFAR10CDataset(VisionDataset):
    mean = CIFAR10Dataset.mean
    std = CIFAR10Dataset.std
    num_classes = CIFAR10Dataset.num_classes
    image_size = CIFAR10Dataset.image_size

    url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1'
    filename = 'CIFAR-10-C.tar'
    base_folder = 'CIFAR-10-C'
    md5 = '56bf5dcef84df0e2308c6dcbcbbd8499'
    per_severity = 10000

    severities = [1, 2, 3, 4, 5]
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur',
        'snow', 'frost', 'fog', 'spatter',
        'brightness', 'contrast', 'saturate',
        'elastic_transform', 'pixelate', 'jpeg_compression'
    ]

    def __init__(self, root, download=False, extract_only=True,
                 severity=1, corruption='gaussian_noise',
                 transform=None, target_transform=None):
        assert severity in self.severities
        assert corruption in self.corruptions

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.slice = slice((severity - 1) * self.per_severity, severity * self.per_severity)

        if download:
            download_and_extract_archive(self.url, os.path.join(root, self.filename), root)
        if extract_only:
            if not os.path.exists(os.path.join(root, self.base_folder)):
                print(f'Extracting {self.__class__.__name__}')
                extract_archive(os.path.join(root, self.filename), root)

        # now load the picked numpy arrays
        images_file_path = os.path.join(self.root, self.base_folder, f'{corruption}.npy')
        self.data = np.load(images_file_path)[self.slice]
        labels_file_path = os.path.join(self.root, self.base_folder, f'labels.npy')
        self.targets = np.load(labels_file_path)[self.slice]

    def download(self):
        download_and_extract_archive(self.url, self.root, filename=self.base_folder, md5=self.md5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


@register_dataset(dataset='c100', is_c=True)
class CIFAR100CDataset(CIFAR10CDataset):
    mean = CIFAR100Dataset.mean
    std = CIFAR100Dataset.std
    num_classes = CIFAR100Dataset.num_classes
    image_size = CIFAR100Dataset.image_size

    url = 'https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1'
    filename = 'CIFAR-100-C.tar'
    base_folder = 'CIFAR-100-C'
    md5 = '11f0ed0f1191edbf9fa23466ae6021d3'
    per_severity = 10000

    severities = [1, 2, 3, 4, 5]
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur',
        'snow', 'frost', 'fog', 'spatter',
        'brightness', 'contrast', 'saturate',
        'elastic_transform', 'pixelate', 'jpeg_compression'
    ]


@register_dataset(dataset='c10', is_c_bar=True)
class CIFAR10CBarDataset(VisionDataset):
    mean = (0.4915, 0.4823, 0.4468)
    std = (0.2470, 0.2435, 0.2616)
    num_classes = 10
    image_size = 32

    filename = 'CIFAR-10-C-Bar.tar.gz'
    base_folder = 'CIFAR-10-C-Bar'
    per_severity = 10000

    severities = [1, 2, 3, 4, 5]
    corruptions = [
        'blue_noise_sample', 'brownish_noise', 'checkerboard_cutout',
        'inverse_sparkles', 'pinch_and_twirl', 'ripple', 'circular_motion_blur',
        'lines', 'sparkles', 'transverse_chromatic_abberation'
    ]

    def __init__(self, root, download=False, extract_only=True,
                 severity=1, corruption='blue_noise_sample',
                 transform=None, target_transform=None):
        assert severity in self.severities
        assert corruption in self.corruptions

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.slice = slice((severity - 1) * self.per_severity, severity * self.per_severity)

        if download:
            raise NotImplementedError('CIFAR C-Bar dataset(s) cannot be downloaded automatically')
        if extract_only:
            if not os.path.exists(os.path.join(root, self.base_folder)):
                print(f'Extracting {self.__class__.__name__}')
                extract_archive(os.path.join(root, self.filename), root)

        # now load the picked numpy arrays
        images_file_path = os.path.join(self.root, self.base_folder, f'{corruption}.npy')
        self.data = np.load(images_file_path)[self.slice]
        labels_file_path = os.path.join(self.root, self.base_folder, f'labels.npy')
        self.targets = np.load(labels_file_path)[self.slice]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


@register_dataset(dataset='c100', is_c_bar=True)
class CIFAR100CBarDataset(CIFAR10CBarDataset):
    mean = CIFAR100Dataset.mean
    std = CIFAR100Dataset.std
    num_classes = 100
    image_size = 32

    filename = 'CIFAR-100-C-Bar.tar.gz'
    base_folder = 'CIFAR-100-C-Bar'


def convert_val_images_to_image_folder(val_image_parent, annotation_file):
    import shutil
    import os

    print("Converting validation images to ImageFolder format...")
    for i, line in enumerate(map(lambda s: s.strip(), open(os.path.join(val_image_parent, annotation_file)))):
        name, wnid = line.split("\t")[0:2]
        origin_path_file = os.path.join(val_image_parent, "images", name)
        destination_path = os.path.join(val_image_parent, wnid)
        destination_path_file = os.path.join(destination_path, name)
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        shutil.move(origin_path_file, destination_path_file)

    os.rmdir(os.path.join(val_image_parent, "images"))
    print("Done!")


@register_dataset(dataset='tin')
class TinyImageNetDataset(ImageFolder):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    num_classes = 200
    image_size = 64

    url = 'https://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    base_folder = 'tiny-imagenet-200'

    val_folder = 'val'
    val_image_folder = 'images'
    val_annotation_file = 'val_annotations.txt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, extract_only=True):
        # extract the dataset if it does not exist yet from the zip file in the root folder
        if download:
            print('Downloading Tiny ImageNet')
            download_and_extract_archive(self.url, os.path.join(root, self.filename), root)
        if extract_only:
            if not os.path.exists(os.path.join(root, self.base_folder)):
                print('Extracting Tiny ImageNet')
                extract_archive(os.path.join(root, self.filename), root)
                print('Done!')

        if train:
            root = os.path.join(root, self.base_folder, 'train')
        else:
            # check if the validation images are already in ImageFolder format
            if os.path.exists(os.path.join(root, self.base_folder, self.val_folder, self.val_image_folder)):
                convert_val_images_to_image_folder(
                    os.path.join(root, self.base_folder, self.val_folder), self.val_annotation_file
                )
            root = os.path.join(root, self.base_folder, 'val')
        super().__init__(root, transform, target_transform)


@register_dataset(dataset='tin', is_c=True)
class TinyImageNetCDataset(ImageFolder):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    num_classes = 200
    image_size = 64

    url = 'https://zenodo.org/record/2536630/files/Tiny-ImageNet-C.tar?download=1'
    filename = 'Tiny-ImageNet-C.tar'
    base_folder = 'Tiny-ImageNet-C'

    severities = [1, 2, 3, 4, 5]
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]

    def __init__(self, root, extract_only=True,
                 severity=1, corruption='gaussian_noise',
                 transform=None, target_transform=None):

        assert severity in self.severities
        assert corruption in self.corruptions

        if extract_only:
            if not os.path.exists(os.path.join(root, self.base_folder)):
                print(f'Extracting {self.__class__.__name__}')
                extract_archive(os.path.join(root, self.filename), root)

        path_to_folder = os.path.join(root, self.base_folder, corruption, str(severity))

        super().__init__(path_to_folder, transform=transform, target_transform=target_transform)


@register_dataset(dataset='tin10')
class TinyImageNet10(TinyImageNetDataset):
    num_classes = 10

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, extract_only=True):
        super().__init__(root, train, transform, target_transform, download, extract_only)

        _targets = [i for i in range(0, 200, 20)]
        _indices = [i for i, label in enumerate(self.targets) if label % 20 == 0]

        # remap the labels
        self.targets = [_targets.index(label) for label in self.targets if label % 20 == 0]
        self.samples = [self.samples[i] for i in _indices]

        # remap the class_to_idx
        self.class_to_idx = {label: i for i, label in enumerate(_targets)}
        self.classes = [self.classes[i] for i in _targets]

        # remap the samples
        self.samples = [(path, self.class_to_idx[label]) for path, label in self.samples]

        # remap the targets
        self.targets = [label for _, label in self.samples]


@register_dataset(dataset='tin10', is_c=True)
class TinyImageNet10C(TinyImageNetCDataset):
    num_classes = 10

    def __init__(self, root, extract_only=True,
                 severity=1, corruption='gaussian_noise',
                 transform=None, target_transform=None):
        super().__init__(root, extract_only, severity, corruption, transform, target_transform)

        _targets = [i for i in range(0, 200, 20)]
        _indices = [i for i, label in enumerate(self.targets) if label % 20 == 0]

        # remap the labels
        self.targets = [_targets.index(label) for label in self.targets if label % 20 == 0]
        self.samples = [self.samples[i] for i in _indices]

        # remap the class_to_idx
        self.class_to_idx = {label: i for i, label in enumerate(_targets)}
        self.classes = [self.classes[i] for i in _targets]

        # remap the samples
        self.samples = [(path, self.class_to_idx[label]) for path, label in self.samples]

        # remap the targets
        self.targets = [label for _, label in self.samples]


@register_dataset(dataset='in')
class ImageNet(ImageFolder):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    num_classes = 1000
    image_size = 224

    train_filename = 'imagenet-train.tar.gz'
    test_filename = 'imagenet-val.tar.gz'

    train_folder = 'train'
    test_folder = 'val'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, extract_only=True):
        # extract the dataset if it does not exist yet from the zip file in the root folder
        if download:
            raise NotImplementedError('ImageNet dataset cannot be downloaded automatically')

        if extract_only:
            if train and not os.path.exists(os.path.join(root, self.train_folder)):
                print('Extracting ImageNet Training')
                extract_archive(os.path.join(root, self.train_filename), root, remove_finished=True)
                print('Done!')

            if not train and not os.path.exists(os.path.join(root, self.test_folder)):
                print('Extracting ImageNet Validation')
                extract_archive(os.path.join(root, self.test_filename), root, remove_finished=True)
                print('Done!')

        if train:
            root = os.path.join(root, self.train_folder)
        else:
            root = os.path.join(root, self.test_folder)

        super().__init__(root, transform, target_transform)


@register_dataset(dataset='in', is_c=True)
class ImageNetC(ImageFolder):
    mean = ImageNet.mean
    std = ImageNet.std
    num_classes = ImageNet.num_classes
    image_size = ImageNet.image_size

    severities = [1, 2, 3, 4, 5]
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]

    filename = 'ImageNet-C.tar.gz'
    base_folder = 'ImageNet_C'

    def __init__(
            self,
            root, extract_only=True,
            severity=1, corruption='gaussian_noise',
            transform=None, target_transform=None
    ):

        assert severity in self.severities
        assert corruption in self.corruptions

        if extract_only:
            if not os.path.exists(os.path.join(root, self.base_folder)):
                print('Extracting ImageNet-C')
                extract_archive(os.path.join(root, self.filename), root, remove_finished=True)

        path_to_folder = os.path.join(root, self.base_folder, corruption, str(severity))

        super().__init__(path_to_folder, transform=transform, target_transform=target_transform)


@register_dataset(dataset='in', is_c_bar=True)
class ImageNetCBar(ImageFolder):
    mean = ImageNet.mean
    std = ImageNet.std
    num_classes = ImageNet.num_classes
    image_size = ImageNet.image_size

    severities = [1, 2, 3, 4, 5]
    corruptions = [
        'blue_noise_sample', 'brownish_noise', 'caustic_refraction', 'checkerboard_cutout', 'cocentric_sine_waves',
        'inverse_sparkles', 'perlin_noise', 'plasma_noise', 'single_frequency_greyscale', 'sparkles'
    ]

    root_folder = ''
    base_folder = 'ImageNet_C_Bar'

    def __init__(
            self,
            severity=1, corruption='blue_noise_sample',
            transform=None, target_transform=None
    ):
        assert severity in self.severities
        assert corruption in self.corruptions

        root = os.path.join(self.root_folder, self.base_folder, corruption, str(severity))
        if not os.path.exists(root):
            raise RuntimeError(f'ImageNet-C-Bar dataset not found in {root}')

        super().__init__(root, transform=transform, target_transform=target_transform)


@register_dataset(dataset='in', is_p=True)
class ImageNetP(DatasetFolder):
    mean = ImageNet.mean
    std = ImageNet.std
    num_classes = ImageNet.num_classes
    image_size = ImageNet.image_size

    corruptions = [
        'gaussian_noise', 'shot_noise',
        'motion_blur', 'zoom_blur',
        'snow',
        'brightness',
        'rotate', 'scale', 'translate', 'tilt'
    ]

    root_folder = ''
    base_folder = 'ImageNet_P'

    def __init__(self, corruption='gaussian_noise', transform=None, target_transform=None):
        assert corruption in self.corruptions

        root = os.path.join(self.root_folder, self.base_folder, corruption)
        if not os.path.exists(root):
            raise RuntimeError(f'ImageNet-P dataset not found in {root}')

        super().__init__(
            root,
            loader=self.video_loader, extensions=('.mp4',),
            transform=transform, target_transform=target_transform
        )

    @staticmethod
    def video_loader(path):
        import cv2
        import torch

        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            frames.append(frame)
        cap.release()
        return torch.stack(frames)


@register_dataset(dataset='in100')
class ImageNet100(ImageNet):
    num_classes = 100

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, extract_only=True):
        super().__init__(root, train, transform, target_transform, download, extract_only)

        _targets = [i for i in range(0, 1000, 10)]
        _indices = [i for i, label in enumerate(self.targets) if label % 10 == 0]

        # remap the labels
        self.targets = [_targets.index(label) for label in self.targets if label % 10 == 0]
        self.samples = [self.samples[i] for i in _indices]

        # remap the class_to_idx
        self.class_to_idx = {label: i for i, label in enumerate(_targets)}
        self.classes = [self.classes[i] for i in _targets]

        # remap the samples
        self.samples = [(path, self.class_to_idx[label]) for path, label in self.samples]

        # remap the targets
        self.targets = [label for _, label in self.samples]


@register_dataset(dataset='in100', is_c=True)
class ImageNet100C(ImageNetC):
    num_classes = 100

    def __init__(
            self,
            root, extract_only=True,
            severity=1, corruption='gaussian_noise',
            transform=None, target_transform=None
    ):
        super().__init__(root, extract_only, severity, corruption, transform, target_transform)

        _targets = [i for i in range(0, 1000, 10)]
        _indices = [i for i, label in enumerate(self.targets) if label % 10 == 0]

        # remap the labels
        self.targets = [_targets.index(label) for label in self.targets if label % 10 == 0]
        self.samples = [self.samples[i] for i in _indices]

        # remap the class_to_idx
        self.class_to_idx = {label: i for i, label in enumerate(_targets)}
        self.classes = [self.classes[i] for i in _targets]

        # remap the samples
        self.samples = [(path, self.class_to_idx[label]) for path, label in self.samples]

        # remap the targets
        self.targets = [label for _, label in self.samples]


if __name__ == '__main__':
    _dataset_t = CIFAR10CBarDataset(
        root=os.path.join('..', '..', '..', 'data'),
        extract_only=True,
        severity=5,
        corruption='blue_noise_sample'
    )

    _dataset_c = CIFAR100CBarDataset(
        root=os.path.join('..', '..', '..', 'data'),
        extract_only=True,
        severity=5,
        corruption='blue_noise_sample'
    )

    print(len(_dataset_t))
    print(len(_dataset_c))
