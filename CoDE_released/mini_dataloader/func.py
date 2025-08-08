import cv2
import numpy as np
from .flip import horizontal_flip, vertical_flip
from .rotate import rotate
from .gaussian_blur import gaussian_blur
from .gaussian_noise import gaussian_noise
from .jpeg_compression import jpeg_compression
from .resize import resize

def read(forgery_path, mask_path):
    forgery = cv2.imread(forgery_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return forgery, mask

def train_process(forgery, mask, augment_prob=0.5, input_size=512, enable_aug_types=[0, 1, 2],
            intensity={'rates': None, 'qfs': None, 'sds': None, 'ksizes': None, 'ratios': None}):
    """

    :param forgery: Forgery image-
    :param mask:    Binary image. White area represents tampering.
    :param augment: Data augmentation, default is True. Type:{Flip, Rotate}
    :param enable_aug_types: Types of allowed data augmentation.
    :param intensity:        The intensity of data augmentation.
    :return:
    """

    # _/_/_/ Data Augmentation _/_/_/
    num_aug_types = np.random.choice(4, 1)
    aug_types = np.random.choice([0, 1, 2], num_aug_types, replace=False)
    for aug_type in aug_types:
        if aug_type == 0:
            forgery, mask = horizontal_flip(forgery, mask, p=augment_prob)
        elif aug_type == 1:
            forgery, mask = vertical_flip(forgery, mask, p=augment_prob)
        elif aug_type == 2:
            forgery, mask = rotate(forgery, mask, p=augment_prob)

    if enable_aug_types is not None:
        aug_types = np.random.choice(enable_aug_types, 1, replace=False)
        for aug_type in aug_types:
            if aug_type == 3:
                forgery = resize(forgery, rates=intensity['rates'], p=augment_prob)
            elif aug_type == 4:
                forgery = jpeg_compression(forgery, qfs=intensity['qfs'], p=augment_prob)
            elif aug_type == 5:
                forgery = gaussian_noise(forgery, sds=intensity['sds'], p=augment_prob)
            elif aug_type == 6:
                forgery = gaussian_blur(forgery, ksizes=intensity['ksizes'], p=augment_prob)

    # _/_/_/ Normalization _/_/_/
    if input_size is not None:
        forgery = cv2.resize(forgery, (input_size, input_size), cv2.INTER_AREA)
        mask = cv2.resize(mask, (input_size, input_size), cv2.INTER_NEAREST)
    mask = np.where(mask > 127, 255, 0)
    forgery, mask = forgery.astype(np.float32) / 255., mask.astype(np.float32) / 255.

    # _/_/_/ To fit the model inputs _/_/_/
    forgery = forgery[:, :, ::-1]
    mask = np.expand_dims(mask, axis=-1)
    forgery, mask = np.transpose(forgery, (2, 0, 1)), np.transpose(mask, (2, 0, 1))

    return forgery, mask


def test_process(forgery, mask, input_size=512, augment_prob=0.2,
            enable_aug_types=[0, 1, 2], intensity={'rates': None, 'qfs': None, 'sds': None, 'ksizes': None,
                                                   'ratios': None}):
    """

    :param forgery:          Forgery image.
    :param mask:             Binary mask. White area represents tampering.
    :param rescale_size:     Input size.
    :param augment_prob:     The probability of data augmentation.
    :param enable_aug_types: Types of allowed data augmentation.
    :param intensity:        The intensity of data augmentation.
    :return:
    """

    # _/_/_/ Data Augmentation _/_/_/

    # The type of data enhancement mixed in a single data
    if enable_aug_types is not None:
        print('aug')
        for aug_type in enable_aug_types:
            if aug_type == 0:
                forgery, mask = horizontal_flip(forgery, mask, p=augment_prob)
            elif aug_type == 1:
                forgery, mask = vertical_flip(forgery, mask, p=augment_prob)
            elif aug_type == 2:
                forgery, mask = rotate(forgery, mask, p=augment_prob)
            elif aug_type == 3:
                forgery = resize(forgery, rates=intensity['rates'], p=augment_prob)
            elif aug_type == 4:
                forgery = jpeg_compression(forgery, qfs=intensity['qfs'], p=augment_prob)
            elif aug_type == 5:
                forgery = gaussian_noise(forgery, sds=intensity['sds'], p=augment_prob)
            elif aug_type == 6:
                forgery = gaussian_blur(forgery, ksizes=intensity['ksizes'], p=augment_prob)

    # _/_/_/ Rescale size and Normalization _/_/_/
    if input_size is not None:
        forgery = cv2.resize(forgery, (input_size, input_size), cv2.INTER_AREA)
        mask = cv2.resize(mask, (input_size, input_size), cv2.INTER_NEAREST)
    mask = np.where(mask > 127, 255, 0)
    forgery, mask = forgery.astype(np.float32) / 255., mask.astype(np.float32) / 255.

    # _/_/_/ To adapt the model inputs _/_/_/
    forgery = forgery[:, :, ::-1]
    mask = np.expand_dims(mask, axis=-1)
    forgery, mask = np.transpose(forgery, (2, 0, 1)), np.transpose(mask, (2, 0, 1))
    return forgery, mask
