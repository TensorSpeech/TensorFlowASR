# Augmentations

Augmentations use `nlpaug`, for futher information, see [nlpaug.readthedocs.io](nlpaug.readthedocs.io)

**YAML Config Structure**

```yaml
augmentations:
    before: ...
    after: ...
```

Where `include_original` is whether to use only augmented dataset or both augmented and original datasets, `before` and `after` are augmentation methods to use before and after features extraction, **sometimes** is whether to randomly apply augmentation or apply all augmentations

**Supported Methods**

All methods that supported my `nlpaug` for audio and spectrogram are supported.

## Changes

-   remove `include_original` augmentation on probability
-   `noise_augment.py` file is deprecated
