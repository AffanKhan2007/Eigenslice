import numpy as np


DEFAULT_ANGLES = np.linspace(0, 180, 12, endpoint=False)


def create_2d_projection_matrix(n: int, angles=None):
    """Build a simple 2D ray/path matrix for one CT slice.

    Each row represents a detector ray. A 1 indicates that a pixel
    lies on that simplified ray path. This is an educational model,
    not clinical CT acquisition geometry.
    """
    if angles is None:
        angles = DEFAULT_ANGLES

    y, x = np.indices((n, n), dtype=float)
    center = (n - 1) / 2.0
    points = np.column_stack(((x - center).ravel(), (y - center).ravel()))

    rows = []
    ray_labels = []

    for angle in angles:
        theta = np.deg2rad(angle)
        # Detector coordinate perpendicular to the ray direction.
        detector_coordinate = (
            points[:, 0] * np.cos(theta)
            + points[:, 1] * np.sin(theta)
        )
        detector_coordinate = np.round(detector_coordinate, 6)

        for detector_value in np.unique(detector_coordinate):
            row = (detector_coordinate == detector_value).astype(float)
            rows.append(row)
            ray_labels.append((float(angle), float(detector_value)))

    return np.asarray(rows, dtype=float), ray_labels


def projection_image(slice_2d, angle, output_size=None):
    """Return a detector-image representation of one projection angle."""
    n = slice_2d.shape[0]
    if output_size is None:
        output_size = n

    y, x = np.indices((n, n), dtype=float)
    center = (n - 1) / 2.0
    points = np.column_stack(((x - center).ravel(), (y - center).ravel()))
    values = slice_2d.ravel()

    theta = np.deg2rad(angle)
    detector_coordinate = (
        points[:, 0] * np.cos(theta)
        + points[:, 1] * np.sin(theta)
    )
    detector_coordinate = np.round(detector_coordinate, 6)

    unique_values = np.sort(np.unique(detector_coordinate))
    detector_values = np.array([
        values[detector_coordinate == detector_value].sum()
        for detector_value in unique_values
    ], dtype=float)

    # Different angles can produce different numbers of detector bins.
    # Resample every projection to the same number of detector pixels so
    # that X-ray images and the sinogram always have a rectangular shape.
    target_width = int(output_size)
    if len(unique_values) == 1:
        resampled = np.full(target_width, detector_values[0], dtype=float)
    else:
        target_coordinates = np.linspace(
            unique_values[0],
            unique_values[-1],
            target_width,
        )
        resampled = np.interp(
            target_coordinates,
            unique_values,
            detector_values,
        )

    # Make a visually useful 2D X-ray panel by repeating the detector row.
    return np.repeat(resampled[None, :], target_width, axis=0)


def simulate_measurements(A, slice_2d):
    """b = A x for one 2D slice."""
    return A @ slice_2d.flatten()
