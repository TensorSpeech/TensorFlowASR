import tensorflow as tf


def _is_per_replica_instance(obj):
    return isinstance(obj, tf.distribute.DistributedValues) and isinstance(obj, tf.__internal__.CompositeTensor)


def _collective_all_reduce_multi_worker(strategy):
    return (isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy)) and strategy.extended._in_multi_worker_mode()


def reduce_per_replica(values, strategy, reduction):
    """Attempt to reduce the structure `values` to single values.

    Given `values` (a `tf.Tensor` or a `PerReplica` structure),
    which represents the values across all the replicas, `reduce_per_replica`
    attempts to "reduce" those values and returns the corresponding structure
    that represents only single values.

    Currently, `reduce_per_replica` is only used for reducing the metric results
    from `tf.distribute.Strategy.run()`. Depending on the underlying
    `Strategy` implementation, `values` may be a `PerReplica` object,
    which can be thought of as a collection of values across the replicas,
    or a `tf.Tensor`, if the strategy has already conducted the reduction
    for the downstream library.

    There are five possible outcomes of reduction:

    1) if the `values` is a structure of simple `tf.Tensor`s, meaning that
       reduction is not actually needed, `reduce_per_replica` returns the
       structure as-is.
    2) else, if `reduction="auto"`, then the best reduction strategy is
       chosen based on the current environment. This should only be used
       for training cases (`fit()`).
    3) else, if `reduction="first"`, then `reduce_per_replica`
       returns the values of the first replica. This is used in the case of
       training and evaluation, where `values` is expected to hold the same
       value across the replicas as a result of `Strategy`'s synchronization
       across the replicas.
       `reduce_per_replica` does not synchronize the values.
    4) else, if `reduction="sum"`, then `reduce_per_replica` returns the sum
       of values for all replicas. This may be used in the custom training loop
       case, where each replica contain different values which are not
       synchronized.
    5) else, if `reduction="concat"`, then `reduce_per_replica`
       returns the concatenation of the values across the replicas, along the
       axis of dimension 0. This is used in the inference case (`predict()`).

    Args:
        values: Structure of `PerReplica` objects or `tf.Tensor`s.
            `tf.Tensor`s are returned as-is.
        strategy: `tf.distribute.Strategy` object.
        reduction: One of `"auto"`, `"first"`, `"concat"`, `"mean"`, or `"sum"`.
            `"auto"` will select `"first"` when used under a TPUStrategy, or
            `"mean"` otherwise.

    Returns:
        Structure of `Tensor`s, representing the result of reduction.
    """

    if reduction == "auto":
        if isinstance(strategy, tf.distribute.TPUStrategy):
            reduction = "first"
        else:
            reduction = "mean"

    def _reduce(v):
        """Reduce a single `PerReplica` object."""
        if _collective_all_reduce_multi_worker(strategy):
            if reduction == "sum":
                return strategy.reduce("SUM", v)
            if reduction == "mean":
                return strategy.reduce("MEAN", v, axis=0)
        if not _is_per_replica_instance(v):
            return v
        if reduction == "first":
            return strategy.experimental_local_results(v)[0]
        if reduction == "sum":
            return tf.reduce_sum(strategy.experimental_local_results(v))
        if reduction == "mean":
            return tf.reduce_mean(strategy.experimental_local_results(v), axis=0)
        raise ValueError("`reduction` must be one of " '"first", "mean", "sum", or "auto". ' f"Received: reduction={reduction}.")

    return tf.nest.map_structure(_reduce, values)
