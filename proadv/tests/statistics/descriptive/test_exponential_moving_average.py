# test_exponential_moving_average.py
import pytest
import numpy as np
from proadv.statistics.series import exponential_moving_average

def test_exponential_moving_average_with_negative_alpha():
    with pytest.raises(ValueError):
        exponential_moving_average([1, 2, 3, 4, 5], -1)

def test_exponential_moving_average_with_greater_than_1_alpha():
    with pytest.raises(ValueError):
        exponential_moving_average([1, 2, 3, 4, 5], 2)