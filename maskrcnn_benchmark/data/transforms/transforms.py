# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, human_boxes, object_boxes):
        for t in self.transforms:
            image, human_boxes, object_boxes = t(image, human_boxes, object_boxes)
        return image, human_boxes, object_boxes

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, human_boxes=None, object_boxes=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if human_boxes is None and object_boxes is None:
            return image
        elif object_boxes is None:
            return image, human_boxes
        elif human_boxes is None:
            return image, {}, object_boxes
        human_boxes = human_boxes.resize(image.size)
        object_boxes = object_boxes.resize(image.size)
        return image, human_boxes, object_boxes


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, human_boxes, object_boxes):
        if random.random() < self.prob:
            image = F.hflip(image)
            human_boxes = human_boxes.transpose(0)
            object_boxes = object_boxes.transpose(0)
        return image, human_boxes, object_boxes

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, human_boxes, object_boxes):
        if random.random() < self.prob:
            image = F.vflip(image)
            #@todo check why after flip, bbox have negative indexes
            human_boxes = human_boxes.transpose(1)
            object_boxes = object_boxes.transpose(1)
        return image, human_boxes, object_boxes

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, human_boxes, object_boxes):
        image = self.color_jitter(image)
        return image, human_boxes, object_boxes


class ToTensor(object):
    def __call__(self, image, human_boxes=None, object_boxes=None):
        if human_boxes is None and object_boxes is None:
            return F.to_tensor(image)
        elif human_boxes is None:
            return F.to_tensor(image), {}, object_boxes
        elif object_boxes is None:
            return F.to_tensor(image), human_boxes
        return F.to_tensor(image), human_boxes, object_boxes


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, human_boxes=None, object_boxes=None):
        # If loading images with OpenCV, we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format,
        # or flip the channels if we want it to be in RGB in [0-1] range.
        # If loading images with PIL, then we need to convert it to BGR
        # and normalize by 255
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if human_boxes is None and object_boxes is None:
            return image
        elif object_boxes is None:
            return image, human_boxes
        elif human_boxes is None:
            return image, {}, object_boxes
        return image, human_boxes, object_boxes
