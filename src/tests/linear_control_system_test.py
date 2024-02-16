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


# Test cases for compute_eigen
def test_compute_A_eigen():
    A = np.array([[-0.01, 0, 0], [-0.01, -0.05, 0], [-0.01, 0, -0.05]])
    B = np.array(
        [
            [0.2, -0.2, 0.01, 0.01, 0.01],
            [-0.01, 0.01, 0.1, -0.1, 0.05],
            [-0.01, 0.01, 0.05, -0.05, 0.1],
        ]
    )
    system = LinearControlSystem(A, B, np.zeros(A.shape[0]))
    eigenvalues, eigenvectors = system.compute_A_eigen()
    expected_eigenvalues = np.array([-0.05, -0.05, -0.01])
    expected_eigenvectors = np.array(
        [[0, 0, 0.94280904], [0, 1, -0.23570226], [1, 0, -0.23570226]]
    )
    assert np.allclose(eigenvalues, expected_eigenvalues)
    assert np.allclose(eigenvectors, expected_eigenvectors)


def test_controllability_matrix():
    A = np.array([[-0.01, 0, 0], [-0.01, -0.05, 0], [-0.01, 0, -0.05]])
    B = np.array(
        [
            [0.2, -0.2, 0.01, 0.01, 0.01],
            [-0.01, 0.01, 0.1, -0.1, 0.05],
            [-0.01, 0.01, 0.05, -0.05, 0.1],
        ]
    )
    system = LinearControlSystem(A, B, np.zeros(A.shape[0]))
    controllability_matrix, rank, controllable = system.compute_controllability_matrix()
    expected_rank = 3
    expected_controllable = True
    expected_controllability_matrix = np.array(
        [
            [
                2.00e-01,
                -2.00e-01,
                1.00e-02,
                1.00e-02,
                1.00e-02,
                -2.00e-03,
                2.00e-03,
                -1.00e-04,
                -1.00e-04,
                -1.00e-04,
                2.00e-05,
                -2.00e-05,
                1.00e-06,
                1.00e-06,
                1.00e-06,
            ],
            [
                -1.00e-02,
                1.00e-02,
                1.00e-01,
                -1.00e-01,
                5.00e-02,
                -1.50e-03,
                1.50e-03,
                -5.10e-03,
                4.90e-03,
                -2.60e-03,
                9.50e-05,
                -9.50e-05,
                2.56e-04,
                -2.44e-04,
                1.31e-04,
            ],
            [
                -1.00e-02,
                1.00e-02,
                5.00e-02,
                -5.00e-02,
                1.00e-01,
                -1.50e-03,
                1.50e-03,
                -2.60e-03,
                2.40e-03,
                -5.10e-03,
                9.50e-05,
                -9.50e-05,
                1.31e-04,
                -1.19e-04,
                2.56e-04,
            ],
        ]
    )
    assert np.allclose(rank, expected_rank)
    assert controllable == expected_controllable
    assert np.allclose(controllability_matrix, expected_controllability_matrix)
