import tensorflow as tf


def reduce_per_replica(values, strategy, reduction):
    if reduction == "auto":
        if isinstance(strategy, tf.distribute.TPUStrategy):
            reduction = "first"
        else:
            reduction = "mean"

    def _reduce(v):
        """Reduce a single `PerReplica` object."""
        if reduction == "first":
            return strategy.experimental_local_results(v)[0]
        if reduction == "sum":
            return strategy.reduce("SUM", v, axis=None)
        if reduction == "mean":
            return strategy.reduce("MEAN", v, axis=None)
        raise ValueError("`reduction` must be one of " '"first", "mean", "sum", or "auto". ' f"Received: reduction={reduction}.")

    return tf.nest.map_structure(_reduce, values)
