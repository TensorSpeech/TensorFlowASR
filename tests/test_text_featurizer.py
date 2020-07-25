from tiramisu_asr.featurizers.text_featurizers import TextFeaturizer

txf = TextFeaturizer(None, blank_at_zero=True)

a = txf.extract("fkaff aksfbfnak kcjhoiu")

print(a)
