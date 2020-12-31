import os
import pytest
import sentencepiece as spm


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
