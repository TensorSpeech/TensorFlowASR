import numpy as np
import tensorflow as tf
from typing import List
from functools import partial
from multiprocessing import Pool, cpu_count
from Levenshtein import editops


class WER:
    def __call__(self, sequences: List[tf.Tensor], label: tf.Tensor) -> List[tf.Tensor]:
        sequences_arr = [np.array(s) for s in sequences]
        func = partial(self._wer, destination=np.array(label))

        with Pool(cpu_count() - 1) as pool:
            wers = pool.map(func, sequences_arr)

        return tf.stack(wers)


    def _wer(self, source: np.ndarray, destination: np.ndarray) -> float:
        source_str = "".join([chr(i) for i in source])
        dest_str = "".join([chr(i) for i in destination])

        edits = editops(source_str, dest_str)
        wer = len(edits) / len(dest_str)

        return wer
