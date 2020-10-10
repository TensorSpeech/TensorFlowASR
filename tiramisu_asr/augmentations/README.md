# Augmentations

Augmentations use `nlpaug`, for futher information, see [nlpaug.readthedocs.io](nlpaug.readthedocs.io)

**YAML Config Structure**

```yaml
augmentations:
    before: ...
    after: ...
```

Where `before` and `after` are augmentation methods to use before and after features extraction.

**Supported Methods**

All methods that supported my `nlpaug` for audio and spectrogram are supported.