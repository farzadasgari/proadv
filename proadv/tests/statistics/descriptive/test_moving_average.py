# test_moving_average.py
import pytest
import numpy as np
from proadv.statistics.series import moving_average

def test_moving_average_with_single_value():
    with pytest.raises(ValueError):
        moving_average(np.array([5]))