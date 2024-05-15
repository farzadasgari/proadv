# test_moving_average.py
import pytest
import numpy as np
from proadv.statistics.series import moving_average

def test_moving_average_with_single_value():
    with pytest.raises(ValueError):
        moving_average(np.array([5]))

def test_moving_average_window_size_greater_than_data_size():
    with pytest.raises(ValueError):
        moving_average(np.array([1, 2, 3, 4, 5]))

def test_moving_average_with_posetive_values():
    with pytest.raises(ValueError):
        moving_average(np.array([1, 2, 3]))

def test_moving_average_with_negative_values():
    with pytest.raises(ValueError):
        moving_average(np.array([-1, -2, -3]))

def test_moving_average_with_large_array():
    with pytest.raises(ValueError):
        moving_average(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

def test_moving_average_with_zero_window_size():
    with pytest.raises(ValueError):
        moving_average(np.array([1, 2, 3]), window_size=0)

def test_moving_average_with_negative_window_size():
    with pytest.raises(ValueError):
        moving_average(np.array([1, 2, 3]), window_size=-5)