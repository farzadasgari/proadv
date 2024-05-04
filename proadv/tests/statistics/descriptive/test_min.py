# test_min.py
import pytest
import numpy as np
from proadv.statistics.descriptive import min


def test_min_with_list():
    assert min([3, 5, 7, 9, 1]) == 1
    assert min([25.5, 10.5, 56.5, 23.5]) == 10.5
    assert min([-1, -2, -3, -4]) == -4
    assert min([-1.5, -2.5, -3.5, -4.5]) == -4.5


def test_min_with_tuple():
    assert min((5, 3, 7, 1)) == 1
    assert min((21.5, 5.5, 10.5, 8.5)) == 5.5
    assert min((-1, -2, -3, -4)) == -4
    assert min((-1.5, -2.5, -3.5, -4.5)) == -4.5


def test_min_with_numpy_array():
    assert min(np.array([5, 3, 7, 1])) == 1
    assert min(np.array([21.5, 5.5, 10.5, 8.5])) == 5.5
    assert min(np.array([-1, -2, -3, -4])) == -4
    assert min(np.array([-1.5, -2.5, -3.5, -4.5])) == -4.5


def test_min_with_empty_data():
    with pytest.raises(ValueError):
        min(())
        min([])


def test_min_with_non_numerical():
    with pytest.raises(TypeError):
        min("proadv")
