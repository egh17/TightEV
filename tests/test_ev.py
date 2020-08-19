"""Test unit functionality."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import evxtb.xtb_ev as evxtb


def test_scale():
    """Test the scaling function."""
    test_array = np.eye(3)
    test_sf = 8
    expected = np.eye(3) * 2

    assert (evxtb.resize_lat(test_array, test_sf) == expected).all()


@pytest.mark.parametrize(
    "array,expected",
    [
        (np.eye(3), ("1.0\t0.0\t0.0\n" "0.0\t1.0\t0.0\n" "0.0\t0.0\t1.0")),
        (2 * np.eye(3), ("2.0\t0.0\t0.0\n" "0.0\t2.0\t0.0\n" "0.0\t0.0\t2.0")),
    ],
)
def test_format(array, expected):
    """Test that an array is correctly formatted."""
    assert evxtb.format_array(array) == expected


rot45 = R.from_euler("z", 45, degrees=True)


@pytest.mark.parametrize(
    "array,expected", [(np.eye(3), 1), (2 * np.eye(3), 8), (rot45.apply(np.eye(3)), 1)]
)
def test_volume(array, expected):
    """Test that a volume is correctly calculated."""
    assert evxtb.unit_vol(array) == expected
