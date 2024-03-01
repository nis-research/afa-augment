import os

import torchvision.transforms as T
from PIL import Image

IMAGENET_SAMPLES_DIR = "imagenet-sample-images"

PIPELINE = lambda img_sz: T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.Resize((img_sz, img_sz)),
])


def feature_extracted(img_size, augmentation, feature_extractor, repetitions=100):
    path_to_dir = os.path.join(os.path.dirname(__file__), IMAGENET_SAMPLES_DIR)

    image_paths = [
        'n04592741_wing.JPEG',
        'n02422699_impala.JPEG',
        'n02509815_lesser_panda.JPEG',
    ]

    mean = (0.485, 0.456, 0.406)  # these are ImageNet mean per channel
    std = (0.229, 0.224, 0.225)  # these are ImageNet std per channel

    all_features = []

    for i, image_path in enumerate(image_paths):
        image = Image.open(os.path.join(path_to_dir, image_path))

        resized_image = PIPELINE(img_size)(image)
        feature_extractor.eval()
        im_features = []

        for _ in range(repetitions):
            normalised_image = T.Normalize(mean, std)(augmentation(T.ToTensor()(resized_image))).unsqueeze(0)
            features = feature_extractor(normalised_image)
            im_features.append(features)

        all_features.append(im_features)

    return all_features
