import numpy as np

if __name__ == "__main__":
    sizes = [5,3,2]

    A = np.random.randn(3, 5)
    x = np.random.randn(5, 1)

    b = np.random.randn(3, 1)
    rjes = np.dot(A, x) + b
