import numpy as np


def create_phantom(n: int = 10) -> np.ndarray:
    """Create a synthetic 3D attenuation volume for the demo."""
    z, y, x = np.indices((n, n, n), dtype=float)
    c = (n - 1) / 2.0

    # Main roughly elliptical body.
    body = (
        ((x - c) / (n * 0.43)) ** 2
        + ((y - c) / (n * 0.37)) ** 2
        + ((z - c) / (n * 0.43)) ** 2
    ) <= 1.0

    volume = np.zeros((n, n, n), dtype=float)
    volume[body] = 0.75

    # Three denser structures at different depths.
    structures = [
        (c + 2.0, c - 1.4, c + 1.0, n * 0.095, 1.8),
        (c - 2.0, c + 1.5, c - 1.0, n * 0.075, 1.45),
        (c + 0.2, c + 2.1, c - 2.1, n * 0.06, 1.2),
    ]

    for cx, cy, cz, r, value in structures:
        mask = (
            (x - cx) ** 2
            + (y - cy) ** 2
            + (z - cz) ** 2
        ) <= r ** 2
        volume[mask] = value

    return volume
