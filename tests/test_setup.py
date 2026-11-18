"""M1 placeholder tests: the package imports and seeding is reproducible."""

import random

import numpy as np

from src.utils import make_run_dir, set_seed


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


def test_run_dirs_started_in_the_same_second_do_not_clash(tmp_path):
    # Fast runs (e.g. 1-epoch tests on a GPU) can start within one second; each must still get its own folder
    dirs = [make_run_dir("same", tmp_path) for _ in range(3)]
    assert len(set(dirs)) == 3 and all(d.is_dir() for d in dirs)
