# Augmentations

**YAML Config Structure**

```yaml
augmentations:
    include_original: False
    before: ...
    after: ...
```

Where `include_original` is whether to use only augmented dataset or both augmented and original datasets, `before` and `after` are augmentation methods to use before and after features extraction.

**Supported Methods**

-   _White Noise_ with SNR chosen randomly from a list
-   _Real World Noises_ with SNR chosen randomly from a list and noise audio file chosen randomly from a directory
-   _Time Stretch_ to slow down or speed up signal with a rate chosen randomly from a to b
-   _Time Masking_ makes a part of spectrogram in time dimension zeros
-   _Frequency Masking_ makes a part of spectrogram in frequency dimension zeros
-   _Pretrained TFLite Segan_ for model to learn segan's generated distribution
