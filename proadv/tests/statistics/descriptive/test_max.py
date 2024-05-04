# test_max.py
import pytest
import numpy as np
from proadv.statistics.descriptive import max


def test_max_with_list():
    assert max([3, 5, 7, 9, 1]) == 9
    assert max([25.5, 10.5, 56.5, 23.5]) == 56.5
    assert max([-1, -2, -3, -4]) == -1
    assert max([-1.5, -2.5, -3.5, -4.5]) == -1.5


def test_max_with_tuple():
    assert max((5, 3, 7, 1)) == 7
    assert max((21.5, 5.5, 10.5, 8.5)) == 21.5
    assert max((-1, -2, -3, -4)) == -1
    assert max((-1.5, -2.5, -3.5, -4.5)) == -1.5


def test_max_with_numpy_array():
    assert max(np.array([5, 3, 7, 1])) == 7
    assert max(np.array([21.5, 5.5, 10.5, 8.5])) == 21.5
    assert max(np.array([-1, -2, -3, -4])) == -1
    assert max(np.array([-1.5, -2.5, -3.5, -4.5])) == -1.5


def test_max_with_empty_data():
    with pytest.raises(ValueError):
        max(())
        max([])


def test_max_with_non_numerical():
    with pytest.raises(TypeError):
        max("proadv")
