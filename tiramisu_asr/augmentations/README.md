# Augmentations

Augmentations use `nlpaug`, for futher information, see [nlpaug.readthedocs.io](nlpaug.readthedocs.io)

**YAML Config Structure**

```yaml
augmentations:
    include_original: False
    before:
        methods: ...
        sometimes: True
    after:
        methods: ...
        sometimes: False
```

Where `include_original` is whether to use only augmented dataset or both augmented and original datasets, `before` and `after` are augmentation methods to use before and after features extraction, **sometimes** is whether to randomly apply augmentation or apply all augmentations

**Supported Methods**

All methods that supported my `nlpaug` for audio and spectrogram are supported.
