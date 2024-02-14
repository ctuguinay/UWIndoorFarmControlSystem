import numpy as np
import pytest

from src.linear_control_system.linear_control_system import LinearControlSystem


# Test cases for initialization
def test_initialization_with_valid_dimensions():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5], [6]])
    x = np.array([7, 8])
    system = LinearControlSystem(A, B, x)
    assert np.array_equal(system.A, A)
    assert np.array_equal(system.B, B)
    assert np.array_equal(system.x, x)


def test_initialization_with_invalid_dimensions():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7], [8]])
    x = np.array([9, 10])
    with pytest.raises(ValueError):
        LinearControlSystem(A, B, x)


# Test cases for calculate_next_state
def test_calculate_next_state_with_valid_dimensions():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5], [6]])
    x = np.array([7, 8])
    system = LinearControlSystem(A, B, x)
    u = np.array([9])
    next_x = system.calculate_next_state(u)
    assert np.array_equal(next_x, np.array([68, 107]))


def test_calculate_next_state_with_invalid_dimensions():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5], [6]])
    x = np.array([7, 8])
    system = LinearControlSystem(A, B, x)
    u = np.array([9, 10])  # incorrect dimensions
    with pytest.raises(ValueError):
        system.calculate_next_state(u)
