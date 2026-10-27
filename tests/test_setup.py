"""M1 placeholder tests: the package imports and seeding is reproducible."""

import random

import numpy as np

from src.utils import set_seed


def test_same_seed_gives_same_numbers():
    set_seed(123)
    first = (random.random(), np.random.rand(3).tolist())
    set_seed(123)
    second = (random.random(), np.random.rand(3).tolist())
    assert first == second


def test_different_seed_gives_different_numbers():
    set_seed(1)
    a = np.random.rand(3).tolist()
    set_seed(2)
    b = np.random.rand(3).tolist()
    assert a != b
