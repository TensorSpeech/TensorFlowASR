from __future__ import absolute_import

import numpy as np
from nltk.metrics import distance


def levenshtein(a, b):
    """
    Calculate Levenshtein distance between a and b
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n
    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]


def wer(decode, target):
    words = set(decode.split() + target.split())
    word2char = dict(zip(words, range(len(words))))

    new_decode = [chr(word2char[w]) for w in decode.split()]
    new_target = [chr(word2char[w]) for w in target.split()]

    return distance.edit_distance(''.join(new_decode),
                                  ''.join(new_target)), len(target.split())


def mywer(decode, target):
    dist = levenshtein(target.lower().split(), decode.lower().split())
    return dist, len(target.split())


def cer(decode, target):
    return distance.edit_distance(decode, target), len(target)
