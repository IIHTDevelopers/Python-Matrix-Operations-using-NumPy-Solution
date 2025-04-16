import numpy as np

class MatrixOperations:
    def __init__(self, matrix_a, matrix_b):
        """Initialize matrices as NumPy arrays"""
        self.matrix_a = np.array(matrix_a)
        self.matrix_b = np.array(matrix_b)

    def dot_product(self):
        """Compute the dot product of two matrices"""
        return np.dot(self.matrix_a, self.matrix_b)

    def matrix_multiplication(self):
        """Perform matrix multiplication of two matrices"""
        if self.matrix_a.shape[1] != self.matrix_b.shape[0]:
            raise ValueError("Matrix A columns must match Matrix B rows for multiplication")
        return np.matmul(self.matrix_a, self.matrix_b)

    def transpose_matrix(self, matrix="A"):
        """Return the transpose of matrix A or matrix B"""
        if matrix == "A":
            return np.transpose(self.matrix_a)
        elif matrix == "B":
            return np.transpose(self.matrix_b)
        else:
            raise ValueError("Invalid matrix selection")

    def determinant(self, matrix="A"):
        """Calculate the determinant of matrix A or B"""
        if matrix == "A":
            return np.linalg.det(self.matrix_a)
        elif matrix == "B":
            return np.linalg.det(self.matrix_b)
        else:
            raise ValueError("Invalid matrix selection")

