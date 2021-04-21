# Augmentations

```yaml
augmentations:
    prob: 0.5 # a number between 0.0 and 1.0, this number indicates the randomness for signal_augment and feature_augment
    signal_augment: ... # augmentation on signal
    feature_augment: ... # augmentation on feature extracted from signal
```

## Methods

See [methods](./methods)

Currently we have:
- SpecAugment: Time Masking and Frequency Masking

Custom augmentation methods is inherited from class `AugmentationMethod` with the function `augment` must be defined.