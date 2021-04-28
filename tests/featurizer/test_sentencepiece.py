import os
import re
import sentencepiece as spm
from tensorflow_asr.datasets.asr_dataset import ASRSliceDataset, ASRSliceTestDataset
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import SentencePieceFeaturizer, SubwordFeaturizer, TextFeaturizer


def test_encoder():
    # Load model
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, "vocabularies")
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(model_path, "sentencepiece_librispeech_960_8000.model"))

    # Encode a dummy sentence
    sentence = 'this is a test'
    embeding_int = sp.encode_as_ids(sentence)
    embeding_str = sp.encode_as_pieces(sentence)
    decoded_int = sp.decode_ids(embeding_int)
    decoded_str = sp.decode_pieces(embeding_str)

    # Assertions
    assert sentence == decoded_int, f"Decoded {decoded_int}, expected {sentence}"
    assert sentence == decoded_str, f"Decoded {decoded_str}, expected {sentence}"


def test_featurizer():
    config = {
        "output_path_prefix": "/data/models/asr/conformer_sentencepiece_subword",
        "model_type": "unigram",
        "target_vocab_size": 8000,
        "blank_at_zero": True,
        "beam_width": 5,
        "norm_score": True,
        "corpus_files": [
            "/data/datasets/LibriSpeech/train-clean-100/transcripts.tsv"
            "/data/datasets/LibriSpeech/train-clean-360/transcripts.tsv"
            "/data/datasets/LibriSpeech/train-other-500/transcripts.tsv"]}

    config_speech = {
        "sample_rate": 16000,
        "frame_ms": 25,
        "stride_ms": 10,
        "num_feature_bins": 80,
        'feature_type': "log_mel_spectrogram",
        "preemphasis": 0.97,
        "normalize_signal": True,
        "normalize_feature": True,
        "normalize_per_frame": False}

    text_featurizer_sentencepiece = SentencePieceFeaturizer.load_from_file(config, None)
    subwords_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 os.pardir,
                                 os.pardir,
                                 "vocabularies",
                                 "librispeech_train_4_1030.subwords")
    text_featurizer_subwords = SubwordFeaturizer.load_from_file(config, subwords_path)
    speech_featurizer = TFSpeechFeaturizer(config_speech)
    data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "transcripts_librispeech_train_clean_100.tsv")

    def get_data(featurizer: TextFeaturizer):
        train_dataset = ASRSliceDataset(
            data_paths=[data_path],
            speech_featurizer=speech_featurizer,
            text_featurizer=featurizer,
            stage="train",
            shuffle=False
        )
        train_data = train_dataset.create(1)
        return next(iter(train_data))

    data_sentencepiece = get_data(text_featurizer_sentencepiece)
    data_subwords = get_data(text_featurizer_subwords)

    assert len(data_sentencepiece) == len(data_subwords)
    assert data_sentencepiece[0].shape == data_subwords[0].shape
    assert data_sentencepiece[0].dtype == data_subwords[0].dtype


def test_iextract():
    config = {
        "output_path_prefix": "/data/models/asr/conformer_sentencepiece_subword",
        "model_type": "unigram",
        "target_vocab_size": 8000,
        "blank_at_zero": True,
        "beam_width": 5,
        "norm_score": True,
        "corpus_files": [
            "/data/datasets/LibriSpeech/train-clean-100/transcripts.tsv"
            "/data/datasets/LibriSpeech/train-clean-360/transcripts.tsv"
            "/data/datasets/LibriSpeech/train-other-500/transcripts.tsv"]}

    config_speech = {
        "sample_rate": 16000,
        "frame_ms": 25,
        "stride_ms": 10,
        "num_feature_bins": 80,
        'feature_type': "log_mel_spectrogram",
        "preemphasis": 0.97,
        "normalize_signal": True,
        "normalize_feature": True,
        "normalize_per_frame": False}

    text_featurizer_sentencepiece = SentencePieceFeaturizer.load_from_file(config, None)
    speech_featurizer = TFSpeechFeaturizer(config_speech)
    data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "transcripts_librispeech_train_clean_100.tsv")

    train_dataset = ASRSliceTestDataset(
        data_paths=[data_path],
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer_sentencepiece,
        stage="train",
        shuffle=False
    )
    train_data = train_dataset.create(1)
    batch = next(iter(train_data))
    file_paths, features, input_length, labels = batch
    labels = text_featurizer_sentencepiece.iextract(labels)
    labels = labels.numpy()[0].decode("utf-8")

    # Open transcript
    file_path = file_paths[0].numpy().decode("utf-8")
    file_path = re.sub(r"(?<!\s)-[0-9]{4}.flac", ".trans.txt", file_path)
    print(file_path)
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
    m = re.search(r"[0-9]+-[0-9]+-[0-9]+\s+([\w\s]+)", lines[0])
    transcript = m.groups(1)[0].lower()

    assert(labels == transcript)
