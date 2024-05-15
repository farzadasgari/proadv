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