import numpy as np
import tensorflow as tf
from multiprocessing import Pool, cpu_count
from jiwer import wer

class WER:
    def __call__(self,
                 sequences: tf.Tensor,  # [batch_size * beam_size]
                 labels: tf.Tensor  # [batch_size * beam_size]
                 ) -> tf.Tensor:
        labels = tf.unstack(labels)
        sequences = tf.unstack(sequences)

        sequences_arr = [np.array(s) for s in sequences]
        labels_arr = [np.array(s) for s in labels]

        with Pool(cpu_count() - 1) as pool:
            wers = pool.starmap(self._wer, zip(sequences_arr, labels_arr))

        return tf.constant(wers)


    def _wer(self, source, destination) -> float:
        source_str = str(source)
        dest_str = str(destination)

        return wer(dest_str, source_str)
