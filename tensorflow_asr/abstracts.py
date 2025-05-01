import typing
from abc import ABC, abstractmethod

from tensorflow_asr import tf


class AbstractTokenizer(ABC):
    initialized: bool

    @abstractmethod
    def make(self):
        pass

    @abstractmethod
    def tokenize(self, text: str) -> tf.Tensor:
        pass

    @abstractmethod
    def detokenize(self, indices: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def prepand_blank(self, text: tf.Tensor) -> tf.Tensor:
        pass


class AbstractDataset(ABC):
    name: str

    @abstractmethod
    def generator(self) -> typing.Generator:
        pass

    @abstractmethod
    def vocab_generator(self) -> typing.Generator:
        pass
