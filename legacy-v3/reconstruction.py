import numpy as np


def gaussian_elimination(A, b, tolerance=1e-12):
    """Solve Ax=b using Gaussian elimination with partial pivoting."""
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    n = len(b)

    if A.shape != (n, n):
        raise ValueError("Gaussian elimination requires a square matrix.")

    aug = np.column_stack((A, b))

    for col in range(n):
        pivot_row = col + np.argmax(np.abs(aug[col:, col]))
        pivot = aug[pivot_row, col]

        if abs(pivot) < tolerance:
            raise ValueError("The normal-equation matrix is singular or rank-deficient.")

        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]

        # Vectorized elimination below the pivot.
        if col + 1 < n:
            factors = aug[col + 1:, col] / aug[col, col]
            aug[col + 1:, col:] -= factors[:, None] * aug[col, col:]

    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        rhs = aug[i, -1]
        if i + 1 < n:
            rhs -= np.dot(aug[i, i + 1:n], x[i + 1:])
        x[i] = rhs / aug[i, i]

    return x


def least_squares_reconstruction(A, b):
    """Solve min ||Ax-b||^2 using normal equations + Gaussian elimination."""
    AT = A.T
    ATA = AT @ A
    ATb = AT @ b
    x = gaussian_elimination(ATA, ATb)
    return x, ATA, ATb


def reconstruct_volume(A, volume):
    """Reconstruct every z-slice independently and stack them into 3D."""
    reconstructed = np.zeros_like(volume, dtype=float)
    matrices = []

    for z in range(volume.shape[0]):
        b = A @ volume[z].flatten()
        x, ATA, ATb = least_squares_reconstruction(A, b)
        reconstructed[z] = x.reshape(volume.shape[1:])
        matrices.append((ATA, ATb))

    return reconstructed, matrices


def reconstruction_metrics(original, reconstructed):
    error = original - reconstructed
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    return mse, rmse, float(np.max(np.abs(error)))


def svd_analysis(A):
    _, singular_values, _ = np.linalg.svd(A, full_matrices=False)
    return singular_values, int(np.linalg.matrix_rank(A))
