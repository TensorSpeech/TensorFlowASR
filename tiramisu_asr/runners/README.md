# Runners :wink:

## Trainers

The trainers use `BaseTrainer` for training models. To create a custom trainer, define these following methods:

1. Set metrics (`train_metrics` and `eval_metrics`)
2. `_train_step` to process batch of data when training
3. `_eval_step` to process batch of data when validating
4. `compile` for loading built models, optimizers and any custom variables
5. Optionally define `save_model_weights` for save latest model weights every checkpoint. 

## Testers

_Testers only run in 1 GPU_ because we have to use `tf.py_function` or `tf.numpy_function` to calculate WER, CER, Semetrics, ...

The testers for **acoustic** models are combined into single class `BaseTester`. Therefore you don't need to define any custom tester for **acoustic** models.

The `BaseTester` do the steps as follows:

1. Load test dataset.
2. Run testing with `greedy` decoding, `beamsearch` decoding and if you provide an `Scorer` in `TextFeaturizer.scorer`, it will decode `beamsearch_with_lm`, otherwise another `beamsearch`.
3. The result of `greedy`, `beamsearch` and `beamsearch_with_lm` are written to the `test.tsv` file in the `outdir` configured in `.yml` config file.
4. Finish testing by reloading whole `test.tsv` and calculate `WER, CER` from it, the results are printed to stdout.
