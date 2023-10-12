# Tokenizers

- [Tokenizers](#tokenizers)
  - [1. Character Tokenizer](#1-character-tokenizer)
  - [2. Wordpiece Tokenizer](#2-wordpiece-tokenizer)
  - [3. Sentencepiece Tokenizer](#3-sentencepiece-tokenizer)


## 1. Character Tokenizer

See [librespeech config](../examples/configs/librispeech/characters/char.yml.j2)

This splits the text into characters and then maps each character to an index. The index starts from 1 and 0 is reserved for blank token. This tokenizer only used for languages that have a small number of characters and each character is not a combination of other characters. For example, English, Vietnamese, etc.

## 2. Wordpiece Tokenizer

See [librespeech config](../examples/configs/librispeech/wordpiece/wp.yml.j2) for wordpiece splitted by whitespace

See [librespeech config](../examples/configs/librispeech/wordpiece/wp_whitespace.yml.j2) for wordpiece that whitespace is a separate token

This splits the text into words and then splits each word into subwords. The subwords are then mapped to indices. Blank token can be set to <unk> as index 0. This tokenizer is used for languages that have a large number of words and each word can be a combination of other words, therefore it can be applied to any language.

## 3. Sentencepiece Tokenizer

See [librespeech config](../examples/configs/librispeech/sentencepiece/sp.yml.j2)

This splits the whole sentence into subwords and then maps each subword to an index. Blank token can be set to <unk> as index 0. This tokenizer is used for languages that have a large number of words and each word can be a combination of other words, therefore it can be applied to any language.