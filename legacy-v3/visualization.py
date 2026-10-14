from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt


def figure_to_png(fig, dpi=180):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def slice_figure(volume, index=None, title="CT Slice"):
    if index is None:
        index = volume.shape[0] // 2
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    im = ax.imshow(volume[index], cmap="gray", interpolation="nearest")
    ax.set_title(f"{title} — slice {index + 1}/{volume.shape[0]}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.colorbar(im, ax=ax, label="Voxel value")
    fig.tight_layout()
    return fig


def comparison_figure(original, reconstructed, index=None):
    if index is None:
        index = original.shape[0] // 2
    a = original[index]
    b = reconstructed[index]
    e = np.abs(a - b)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(a, cmap="gray", interpolation="nearest")
    axes[0].set_title("Original")
    axes[1].imshow(b, cmap="gray", interpolation="nearest")
    axes[1].set_title("Reconstructed")
    axes[2].imshow(e, cmap="hot", interpolation="nearest")
    axes[2].set_title("Absolute Error")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Slice {index + 1}/{original.shape[0]}")
    fig.tight_layout()
    return fig


def xray_figure(image, angle):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(image, cmap="gray", interpolation="nearest", aspect="auto")
    ax.set_title(f"Simulated X-ray projection — {angle:.0f}°")
    ax.set_xlabel("Detector position")
    ax.set_ylabel("Projection")
    fig.colorbar(im, ax=ax, label="Attenuation sum")
    fig.tight_layout()
    return fig


def sinogram_figure(volume, angles):
    """Create a fixed-width sinogram from the central z-slice."""
    from projection import projection_image

    slice_2d = volume[volume.shape[0] // 2]
    detector_width = slice_2d.shape[0]
    rows = []

    for angle in angles:
        img = projection_image(
            slice_2d,
            float(angle),
            output_size=detector_width,
        )
        # projection_image repeats the detector row for display; only
        # one row is needed for the sinogram.
        rows.append(img[0])

    sino = np.vstack(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        sino,
        cmap="gray",
        aspect="auto",
        interpolation="nearest",
        extent=[0, detector_width, float(angles[-1]), float(angles[0])],
    )
    ax.set_title("Sinogram — central slice")
    ax.set_xlabel("Detector position")
    ax.set_ylabel("Projection angle (degrees)")
    fig.colorbar(im, ax=ax, label="Projection value")
    fig.tight_layout()
    return fig


def volume_3d_figure(volume, threshold=0.35):
    """Render the reconstructed volume as actual voxels, not a point cloud."""
    filled = volume > threshold
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(filled, facecolors="lightgray", edgecolor="dimgray", linewidth=0.25, alpha=0.9)
    ax.set_title("Reconstructed 3D CT Volume")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    return fig
