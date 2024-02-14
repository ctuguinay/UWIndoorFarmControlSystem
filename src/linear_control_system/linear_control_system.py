import numpy as np


class LinearControlSystem:
    def __init__(self, A, B, x):
        self.A = A
        self.B = B
        self.x = x

        # Check dimensions
        if B.shape[0] != A.shape[0]:
            raise ValueError("Number of rows in B must match number of rows in A")
        if A.shape[1] != x.shape[0]:
            raise ValueError("Number of columns in A must match number of rows in x")

    def calculate_next_state(self, u):
        if self.B.shape[1] != u.shape[0]:
            raise ValueError("Number of columns of B must match number of rows in u")

        next_x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.x = next_x
        return next_x
