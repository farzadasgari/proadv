# test_median.py
import pytest
import numpy as np
from proadv.statistics.descriptive import median


def test_median_with_numpy_array():
    assert round(median(np.array([2, 4, 6, 8])), 2) == 5.00
    assert round(median(np.array(sorted([2, 4, 10, 8]))), 2) == 6.00
    assert round(median(np.array(sorted([2.5, 4.5, 8.5, 5.5]))), 2) == 5.00
    assert round(median(np.array([2.5, 4.5, 6.5, 8.5])), 2) == 5.50
    assert round(median(np.array([2, 4, 6, 8, 10])), 2) == 6.00
    assert round(median(np.array([2.5, 4.5, 6.5, 8.5, 10.5])), 2) == 6.50
    assert round(median(np.array([-1, -2, -3, -4])), 2) == -2.5
    assert round(median(np.array([-1.5, -2.5, -3.5, -4.5])), 2) == -3.00
    assert round(median(np.array([-1.5, -2.5, -3.5, -4.5, -5.5])), 2) == -3.50


def test_median_with_empty_data():
    with pytest.raises(ValueError):
        median(np.array([]))


def test_median_with_non_numerical():
    with pytest.raises(TypeError):
        median(np.array["proadv"])
