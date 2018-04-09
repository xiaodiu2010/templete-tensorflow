import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
sys.path.append('../')
import cv2
from img_aug import imgaug as ia
from img_aug import augmenters as iaa


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class Resize(object):
    def __init__(self, img_size, mask_size):
        self.img_size = tuple(img_size)
        self.mask_size = tuple(mask_size)

    def __call__(self, image, mask):
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
        mask  = cv2.resize(mask, self.mask_size)
        return image.astype(np.float32), mask


class ConvertFromInts(object):
    def __call__(self, image, mask):
        return image.astype(np.float32), mask


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, mask):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), mask


class Nomalize(object):
    def __init__(self, nomalizer):
        self.nomalizer = np.array(nomalizer, dtype=np.float32)

    def __call__(self, image, mask):
        image = image.astype(np.float32)
        image /= self.nomalizer
        image = 2*image - 1
        return image.astype(np.float32), mask


class augment_all(object):
    def __init__(self, ):
        self.st = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.seq_img = iaa.Sequential([
            self.st(iaa.GaussianBlur((0, 1.), name='GaussianBlur')),  # blur images with a sigma between 0 and 3.0
            self.st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02), per_channel=0.5, name='AdditiveGaussianNoise')),
            # add gaussian noise to images
            self.st(iaa.Dropout((0.0, 0.05), per_channel=0.5, name="Dropout")),
            # randomly remove up to 10% of the pixels
            self.st(iaa.Add((-20, 20), per_channel=0.5, name="Add")),
            # change brightness of images (by -10 to 10 of original value)
            self.st(iaa.Multiply((0.8, 1.2), per_channel=0.5, name="Multiply")),
            # change brightness of images (50-150% of original value)
            self.st(iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5, name="ContrastNormalization")),
            # improve or worsen the contrast
            self.st(iaa.ElasticTransformation(alpha=(0.8, 1.2), sigma=0.02, name="ElasticTransformation")),
            # apply elastic transformations with random strengths

        ],
            random_order=True  # do all of the above in random order
        )

        self.seq_affine = iaa.Sequential([
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_px={"x": (-50, 50), "y": (-50, 50)}, # translate by -16 to +16 pixels (per axis)
                rotate=(-30, 30), # rotate by -45 to +45 degrees
                shear=(-20, 20), # shear by -16 to +16 degrees
                order=1, # use any of scikit-image's interpolation methods
                cval=0., # if mode is constant, use a cval between 0 and 1.0
                mode="constant" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ),
        ],
            random_order=True  # do all of the above in random order
        )

        self.seq_transform = iaa.Sequential([
                iaa.Fliplr(0.5, name="Fliplr"),  # horizontally flip 50% of all images
                iaa.Flipud(0.5, name="Flipud"),  # vertically flip 50% of all images
                self.st(iaa.Crop(percent=(0, 0.1), name="Crop")),  # crop images by 0-10% of their height/width
        ],
            random_order=True  # do all of the above in random order
        )

        self.seq_det =None
        self.hooks = ia.HooksImages(activator=self.activator_heatmaps)

    def activator_heatmaps(self, images, augmenter, parents, default):
        if augmenter.name in ["GaussianBlur", "Dropout",
                              "AdditiveGaussianNoise", "Add", "Multiply",
                              "ContrastNormalization", "ElasticTransformation"]:
            return False
        else:
            # default value for all other augmenters
            return default

    def __call__(self, image, mask):
        ## augment images
        image = self.seq_img.augment_images([image])[0]
        dim = image.shape[2]

        ## affine images and masks
        mask = np.expand_dims(mask, -1)
        image, mask = self.seq_affine.augment_images([image, mask])

        ## transform images and masks
        new_image = np.concatenate([image, mask], -1)
        new_image = self.seq_transform.augment_images([new_image])[0]
        image = new_image[:, :, : dim]
        mask = new_image[:, :, dim]
        #mask = mask[:,:,0]
        return image, mask


class Augmentation(object):
    def __init__(self, config, is_train=True):
        self.mean = config.mean
        self.is_train = is_train
        self.img_size = (config.img_out_shape[0], config.img_out_shape[1])
        self.mask_size = config.mask_out_shape
        self.augment_train = Compose([
            augment_all(),
            Resize(self.img_size, self.mask_size),
            ConvertFromInts(),
            #SubtractMeans(self.mean),
            Nomalize(255.)
        ])
        self.augment_test = Compose([
            Resize(self.img_size, self.mask_size),
            ConvertFromInts(),
            #SubtractMeans(self.mean),
            Nomalize(255.)
        ])

    def __call__(self, img, mask):
        if self.is_train:
            return self.augment_train(img, mask)
        else:
            return self.augment_test(img, mask)