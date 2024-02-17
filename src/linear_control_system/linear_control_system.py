import numpy as np


class LinearControlSystem:
    def __init__(self, A=None, B=None, x=None):
        self.A = A
        self.B = B
        self.x = x

        # Check dimensions
        if B.shape[0] != A.shape[0]:
            raise ValueError("Number of rows in B must match number of rows in A")
        if A.shape[1] != x.shape[0]:
            raise ValueError("Number of columns in A must match number of rows in x")

    def calculate_next_state(self, u):
        if not hasattr(self, "A") or not hasattr(self, "B") or not hasattr(self, "x"):
            raise ValueError(
                "A, B, and x must be initialized before calling calculate_next_state"
            )

        if self.B.shape[1] != u.shape[0]:
            raise ValueError("Number of columns of B must match number of rows in u")

        next_x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.x = next_x
        return next_x

    def compute_controllability_matrix(self):
        if not hasattr(self, "A") or not hasattr(self, "B"):
            raise ValueError("Matrices A and B must be defined.")

        n = self.A.shape[0]
        controllability_matrix = self.B
        for i in range(1, n):
            controllability_matrix = np.hstack(
                (controllability_matrix, np.linalg.matrix_power(self.A, i) @ self.B)
            )
        rank = np.linalg.matrix_rank(controllability_matrix)
        controllable = rank == n
        return controllability_matrix, rank, controllable

    def compute_A_eigen(self):
        if not hasattr(self, "A"):
            raise ValueError("Matrices A must be defined.")

        eigenvalues, eigenvectors = np.linalg.eig(self.A)
        return eigenvalues, eigenvectors

    def compute_observability_matrix(self):
        if not hasattr(self, "C"):
            raise ValueError("Matrix C must be defined.")

        n = self.A.shape[0]
        observability_matrix = self.C
        for i in range(1, n):
            observability_matrix = np.vstack(
                (
                    observability_matrix,
                    np.dot(np.linalg.matrix_power(self.A, i), self.C),
                )
            )
        rank = np.linalg.matrix_rank(observability_matrix)
        observable = rank == n
        return observability_matrix, rank, observable
