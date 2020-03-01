from __future__ import absolute_import

from augmentations.SpecAugment import time_warping, time_masking, freq_masking


def no_aug(features):
    return features


class Augmentation:
    def __init__(self, func, is_post=True, **kwargs):
        self.func = func
        # Whether postaugmentation or preaugmentation of feature extraction
        self.is_post = is_post
        self.kwargs = kwargs  # Save parameters in config

    def __call__(self, *args, **kwargs):
        if self.kwargs:
            return self.func(*args, **self.kwargs)
        return self.func(*args, **kwargs)


class FreqMasking(Augmentation):
    def __init__(self, **kwargs):
        super().__init__(func=freq_masking, is_post=True, **kwargs)


class TimeMasking(Augmentation):
    def __init__(self, **kwargs):
        super().__init__(func=time_masking, is_post=True, **kwargs)


class TimeWarping(Augmentation):
    def __init__(self, **kwargs):
        super().__init__(func=time_warping, is_post=True, **kwargs)
