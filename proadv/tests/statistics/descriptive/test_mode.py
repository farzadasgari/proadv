# test_mode.py
import pytest
import numpy as np
from proadv.statistics.descriptive import mode

def test_mode_with_list():
    assert mode([1.5, 5.5, 5.5, 1.5, 7.5, 1.5, 8.5, 5.5, 7.5, 8.5, 5.5]) == (5.5, 4)
    assert mode([-2.5, -4.5, -6.5, -6.5, -6.5, -8.5, -8.5]) == (-6.5, 3)
    assert mode([1, 3, 5, 7, 9, 7, 5, 9, 5, 7, 9, 7]) == (7,4)
    assert mode([-1, -2, -4, -3, -4, -5, -6]) == (-4, 2)
    assert mode([5]) == (5, 1)

def test_mode_with_tuple():
    assert mode((1.5, 5.5, 5.5, 1.5, 7.5, 1.5, 8.5, 5.5, 7.5, 8.5, 5.5)) == (5.5, 4)
    assert mode((-2.5, -4.5, -6.5, -6.5, -6.5, -8.5, -8.5)) == (-6.5, 3)
    assert mode((1, 3, 5, 7, 9, 7, 5, 9, 5, 7, 9, 7)) == (7, 4)
    assert mode((-1, -2, -4, -3, -4, -5, -6)) == (-4, 2)
    assert mode((5)) == (5, 1)

def test_mode_with_numpy_array():
    assert mode(np.array([1.5, 5.5, 5.5, 1.5, 7.5, 1.5, 8.5, 5.5, 7.5, 8.5, 5.5])) == (5.5, 4)
    assert mode(np.array([-2.5, -4.5, -6.5, -6.5, -6.5, -8.5, -8.5])) == (-6.5, 3)
    assert mode(np.array([1, 3, 5, 7, 9, 7, 5, 9, 5, 7, 9, 7])) == (7, 4)
    assert mode(np.array([-1, -2, -4, -3, -4, -5, -6])) == (-4, 2)
    assert mode(np.array([5])) == (5, 1)

def test_mode_with_non_numerical():
    with pytest.raises(TypeError):
        mode(np.array["proadv"])

def test_mode_with_empty_data():
    with pytest.raises(ValueError):
        mode(())
        mode([])