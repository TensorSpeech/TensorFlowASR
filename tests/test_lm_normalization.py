"""
`LMDataset.normalize_corpus_text`: turning written text into something the transducer could say.

A corpus is written, not spoken. It carries punctuation and digits; an ASR vocabulary carries
neither. Every token the language model knows but the acoustic model cannot emit is probability
mass spent on nothing, so an un-normalised corpus makes fusion *worse* rather than better. The
LibriSpeech LM corpus is normalised before OpenSLR ships it; no such file exists for most
languages, which is why this runs here.

The sweep in `test_vietnamese_le_is_only_used_when_the_tens_digit_is_zero` is the reason this file
exists. `num2words` reads a Vietnamese group with a zero hundreds digit as "lẻ" whatever the tens
digit is, so 2018 comes back "hai nghìn lẻ mười tám" where Vietnamese says "hai nghìn không trăm
mười tám". Every year of this century is wrong out of the box, and in a news corpus a year is the
most common number there is.
"""

import re

import pytest

from tensorflow_asr.datasets import LMDataset


def normalizer(language=None, keep_punctuation=None):
    """
    A bare `LMDataset` carrying only what normalisation reads.

    Built without `__init__` on purpose: the real one resolves data paths and loads metadata, none
    of which this pure text function touches, and requiring a corpus on disk to test a regex would
    make these tests slow and fragile for no gain.
    """
    dataset = LMDataset.__new__(LMDataset)
    dataset.language = language
    dataset.keep_punctuation = keep_punctuation
    return dataset


def normalize(text, language=None, keep_punctuation=None):
    return normalizer(language, keep_punctuation).normalize_corpus_text(text)


# ------------------------------------------------------------------ off by default


def test_no_language_means_no_normalisation():
    """
    There is no language-independent reading of "2018", so an undeclared corpus is left alone.

    Silently applying English rules to Vietnamese text would be worse than doing nothing: the text
    would look processed while teaching the language model the wrong words.
    """
    text = "Năm 2018, giá 1.000.000 đồng."

    assert normalize(text, language=None) == text


def test_an_unknown_language_still_strips_and_reads():
    """
    A language `num2words` knows but this file has no rules for must not crash or silently no-op.

    It falls back to stripping every punctuation mark and reading numbers through `num2words`, which
    is the safe half of the job; only the keep-set and the decimal word are missing.
    """
    out = normalize("Hallo, wereld! 21", language="nl")

    assert "," not in out and "!" not in out
    assert not re.search(r"\d", out), "digits must not survive"


# ------------------------------------------------------------------ numbers


@pytest.mark.parametrize(
    "number,expected",
    [
        (2018, "hai nghìn không trăm mười tám"),  # zero hundreds, non-zero tens -> "không trăm"
        (2010, "hai nghìn không trăm mười"),
        (2025, "hai nghìn không trăm hai mươi lăm"),
        (2005, "hai nghìn lẻ năm"),  # zero hundreds AND zero tens -> "lẻ" is correct
        (1115, "một nghìn một trăm mười lăm"),  # non-zero hundreds, untouched
        (15, "mười lăm"),  # lăm not năm after mười
        (21, "hai mươi mốt"),  # mốt not một after mươi
        (1000000, "một triệu"),
    ],
)
def test_vietnamese_numbers_read_correctly(number, expected):
    assert normalize(str(number), language="vi") == expected


def test_vietnamese_le_is_only_used_when_the_tens_digit_is_zero():
    """
    Sweep the rule rather than trusting the handful of examples above.

    Two invariants, which together are the rule: "lẻ" is never followed by a tens expression, and a
    number whose hundreds digit is zero while its tens digit is not must say "không trăm".
    """
    lm = normalizer("vi")
    misplaced_le, missing_khong_tram = [], []

    for number in range(100000):
        words = lm._read_number(str(number))
        if re.search(r"\blẻ (mười|mươi)\b", words):
            misplaced_le.append(number)
        hundreds, tens = (number % 1000) // 100, (number % 100) // 10
        if number >= 1000 and hundreds == 0 and tens != 0 and "không trăm" not in words:
            missing_khong_tram.append(number)

    assert not misplaced_le, f"'lẻ' before a tens word, e.g. {misplaced_le[:5]}"
    assert not missing_khong_tram, f"missing 'không trăm', e.g. {missing_khong_tram[:5]}"


def test_group_separators_are_only_separators_when_they_separate_groups_of_three():
    """
    "1.000.000" is a million; "1.2.3" is a version number, and reading it as one would be a lie.

    With the dots gone the two are the same string, so the grouping is what tells them apart. A
    version falls back to digit-by-digit, which is roughly how it is spoken and leaves no digits.
    """
    assert normalize("1.000.000", language="vi") == "một triệu"
    assert normalize("1.2.3", language="vi") == "một hai ba"
    assert normalize("$1,000,000", language="en") == "one million"


def test_decimals_are_read_digit_by_digit_after_the_separator():
    """ "3,14" is "ba phẩy một bốn", not "ba phẩy mười bốn" -- a decimal is spoken as digits."""
    assert normalize("3,14", language="vi") == "ba phẩy một bốn"
    assert normalize("3.14", language="en") == "three point one four"


def test_no_digit_ever_survives():
    """
    The one guarantee that matters: a digit reaching the tokenizer is a token the model cannot emit.

    Whatever the shape -- quantity, version, date, code, decimal -- something pronounceable comes
    out the other side.
    """
    corpus = [
        "Năm 2018 tăng 3,14%",
        "Phiên bản 1.2.3 ra mắt ngày 12.05.2018",
        "Mã ABC-123 và COVID-19",
        "0 1 2 3 4 5 6 7 8 9",
        "1.000.000.000 đồng",
    ]

    for line in corpus:
        assert not re.search(r"\d", normalize(line, language="vi")), f"digits survived in {line!r}"


# ------------------------------------------------------------------ punctuation


def test_vietnamese_strips_every_punctuation_mark():
    """Vietnamese transcripts carry no apostrophes or hyphens, so nothing is worth keeping."""
    assert normalize('"Hà Nội" (thủ đô) — xin chào, bạn!', language="vi") == "Hà Nội thủ đô xin chào bạn"


def test_english_keeps_what_a_transcript_would_contain():
    """
    LibriSpeech writes "don't" and "well-known", so stripping `'` and `-` would invent words.

    The same characters standing alone are ordinary punctuation and must still go, which is why the
    rule is about what they join rather than which characters they are.
    """
    assert normalize("Don't go -- wait! A well-known fact (really).", language="en") == "Don't go wait A well-known fact really"


def test_keep_punctuation_overrides_the_language_default():
    """A corpus whose transcripts were written without apostrophes can say so."""
    assert normalize("Don't go!", language="en", keep_punctuation="") == "Don t go"


def test_punctuation_becomes_a_space_rather_than_nothing():
    """Deleting it would weld two words together; "one,two" is two words, not one."""
    assert normalize("một,hai", language="vi") == "một hai"


def test_symbols_are_stripped_not_spoken():
    """
    "%" becomes a space, not "phần trăm".

    Speaking symbols needs a word per symbol per language, and a wrong reading is worse than a
    missing one. Documented here because it is a deliberate limit rather than an oversight.
    """
    out = normalize("tăng 5% và 3€", language="vi")

    assert "%" not in out and "€" not in out
    assert "phần trăm" not in out


# ------------------------------------------------------------------ what it does not touch


def test_case_and_unicode_are_left_to_the_tokenizer():
    """
    `Tokenizer.normalize_text` already lowercases and NFKC-normalises every line it tokenizes.

    Doing it here too would be wasted work on a multi-GB corpus, and NFKC there is what folds the
    two Vietnamese tone placements (`hoà` / `hòa`) onto one spelling.
    """
    assert normalize("Xin Chào", language="vi") == "Xin Chào"


# ------------------------------------------------------------------ wiring


def test_the_corpus_is_normalised_and_transcripts_are_not(tmp_path):
    """
    The end-to-end path: `language` reaches the dataset from config, and only `.txt` is rewritten.

    Transcripts are what the transducer trained on, and the internal LM's job is to approximate
    exactly what it picked up from them, so rewriting a `.tsv` here would change what LODR
    subtracts. `_iter_texts` is the one place both file kinds pass through, which is why the
    distinction has to be made there rather than in the normaliser.
    """
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Năm 2018, giá 1.000.000 đồng.\n", encoding="utf-8")
    transcripts = tmp_path / "transcripts.tsv"
    transcripts.write_text("PATH\tDURATION\tTRANSCRIPT\na.wav\t1.0\tNăm 2018, giá 1.000.000 đồng.\n", encoding="utf-8")

    dataset = LMDataset(
        stage="train",
        tokenizer=None,
        data_paths=[str(corpus), str(transcripts)],
        language="vi",  # arrives through `lm_dataset_config`, which LMDataset spreads over kwargs
        enabled=True,
    )
    corpus_line, transcript_line = list(dataset._iter_texts())

    assert corpus_line == "Năm hai nghìn không trăm mười tám giá một triệu đồng"
    assert transcript_line == "Năm 2018, giá 1.000.000 đồng.", "a transcript must be passed through untouched"


def test_language_is_case_insensitive_and_blank_means_off(tmp_path):
    """`language: VI` in a config is the same as `vi`, and an empty string is the same as unset."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Năm 2018\n", encoding="utf-8")

    def first_line(language):
        dataset = LMDataset(stage="train", tokenizer=None, data_paths=[str(corpus)], language=language, enabled=True)
        return next(iter(dataset._iter_texts()))

    assert first_line("VI") == "Năm hai nghìn không trăm mười tám"
    assert first_line("") == "Năm 2018"
