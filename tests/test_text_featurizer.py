from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer

txf = CharFeaturizer(None, blank_at_zero=True)

a = txf.extract("fkaff aksfbfnak kcjhoiu")

print(a)
