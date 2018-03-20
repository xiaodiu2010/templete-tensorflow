import cv2
import numpy as np
from numpy import random


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class ConvertFromInts(object):
    def __call__(self, image, mask):
        return image.astype(np.float32), mask


class Resize(object):
    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, image, mask):
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_CUBIC)
        mask  = cv2.resize(mask, self.size)
        return image.astype(np.float32), mask


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, mask):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), mask


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, mask):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, mask


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, mask):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, mask


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, mask):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            image = image[:, :, swap]
        return image, mask



class ConvertColor(object):
    def __init__(self, current='RGB', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, mask):
        if self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, mask


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, mask):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, mask


class RandomBrightness(object):
    def __init__(self, delta=32):
        self.delta = delta

    def __call__(self, image, mask):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, mask


class RandomMirror(object):
    def __call__(self, image, mask):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            mask  = mask[:, ::-1]
        return image, mask


class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image, mask):
        image = image[:, :, self.swaps]
        return image, mask


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, mask):
        im = image.copy()
        im, mask = self.rand_brightness(im, mask)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, mask = distort(im, mask)
        return self.rand_light_noise(image=im, mask=mask)


class Augmentation(object):
    def __init__(self, config, is_train=True):
        self.mean = config.mean
        self.is_train = is_train
        self.size = config.out_shape
        self.augment_train = Compose([
            ConvertFromInts(),
            Resize(self.size),
            PhotometricDistort(),
            RandomMirror(),
            #SubtractMeans(self.mean)
        ])
        self.augment_test = Compose([
            ConvertFromInts(),
            Resize(self.size),
            #SubtractMeans(self.mean)
        ])

    def __call__(self, img, mask):
        if self.is_train:
            return self.augment_train(img, mask)
        else:
            return self.augment_test(img, mask)