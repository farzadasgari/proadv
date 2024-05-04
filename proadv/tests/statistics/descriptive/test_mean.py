# test_mean.py
import pytest
import numpy as np
from proadv.statistics.descriptive import mean


def test_mean_with_list():
    assert round(mean([2, 4, 6, 8]), 2) == 5.00
    assert round(mean([2.5, 4.5, 6.5, 8.5]), 2) == 5.50
    assert round(mean([2, 4, 6, 8, 10]), 2) == 6.00
    assert round(mean([2.5, 4.5, 6.5, 8.5, 10.5]), 2) == 6.50
    assert round(mean([-1, -2, -3, -4]), 2) == -2.50
    assert round(mean([-1, -2, -3, -4, -5]), 2) == -3.00
    assert round(mean([-1.5, -2.5, -3.5, -4.5]), 2) == -3.00
    assert round(mean([-1.5, -2.5, -3.5, -4.5, -5.5]), 2) == -3.50


def test_mean_with_tuple():
    assert round(mean((2, 4, 6, 8)), 2) == 5.00
    assert round(mean((2.5, 4.5, 6.5, 8.5)), 2) == 5.50
    assert round(mean((2, 4, 6, 8, 10)), 2) == 6.00
    assert round(mean((2.5, 4.5, 6.5, 8.5, 10.5)), 2) == 6.50
    assert round(mean((-1, -2, -3, -4)), 2) == -2.50
    assert round(mean((-1.5, -2.5, -3.5, -4.5)), 2) == -3.00
    assert round(mean((-1.5, -2.5, -3.5, -4.5)), 2) == -3.00
    assert round(mean((-1.5, -2.5, -3.5, -4.5, -5.5)), 2) == -3.50


def test_mean_with_numpy_array():
    assert round(mean(np.array([2, 4, 6, 8])), 2) == 5.00
    assert round(mean(np.array([2.5, 4.5, 6.5, 8.5])), 2) == 5.50
    assert round(mean(np.array([2, 4, 6, 8, 10])), 2) == 6.00
    assert round(mean(np.array([2.5, 4.5, 6.5, 8.5, 10.5])), 2) == 6.50
    assert round(mean(np.array([-1, -2, -3, -4])), 2) == -2.50
    assert round(mean(np.array([-1, -2, -3, -4, -5])), 2) == -3.00
    assert round(mean(np.array([-1.5, -2.5, -3.5, -4.5])), 2) == -3.00
    assert round(mean(np.array([-1.5, -2.5, -3.5, -4.5, -5.5])), 2) == -3.50
