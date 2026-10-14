import io
import zipfile

import numpy as np
import streamlit as st

from phantom import create_phantom
from projection import DEFAULT_ANGLES, create_2d_projection_matrix, projection_image
from reconstruction import reconstruct_volume, reconstruction_metrics, svd_analysis
from visualization import (
    comparison_figure,
    figure_to_png,
    slice_figure,
    sinogram_figure,
    volume_3d_figure,
    xray_figure,
)

st.set_page_config(page_title="Matrix CT Reconstruction", page_icon="🩻", layout="wide")

st.title("🩻 3D CT Reconstruction Using Matrix Operations")
st.markdown(
    """
**Educational model:** multiple simulated 2D X-ray projections are converted into a
3D voxel volume using a projection matrix and the syllabus topics of
least-squares approximation and Gaussian elimination.

\[
Ax=b,\qquad \min_x\|Ax-b\|^2,\qquad A^TAx=A^Tb
\]
"""
)
st.warning(
    "This is a mathematical simulation, not a clinical CT reconstruction system. "
    "The X-rays are generated from a synthetic phantom and the ray geometry is simplified."
)

st.sidebar.header("Settings")
n = st.sidebar.selectbox("Voxel resolution", [8, 10, 12], index=1,
                         help="10 means a 10×10×10 volume (1000 voxels).")
threshold = st.sidebar.slider("3D display threshold", 0.10, 1.20, 0.35, 0.05)
show_svd = st.sidebar.checkbox("Show optional SVD analysis", True)
run = st.sidebar.button("🚀 Generate / Reconstruct", type="primary")

@st.cache_data(show_spinner=False)
def run_simulation(n):
    volume = create_phantom(n)
    A, ray_labels = create_2d_projection_matrix(n, DEFAULT_ANGLES)
    reconstructed, matrices = reconstruct_volume(A, volume)
    b = A @ volume[volume.shape[0] // 2].flatten()
    metrics = reconstruction_metrics(volume, reconstructed)
    return volume, A, ray_labels, reconstructed, matrices, b, metrics

if run or "results" not in st.session_state or st.session_state.get("n") != n:
    with st.spinner(f"Building the {n}×{n}×{n} volume and reconstructing it..."):
        try:
            st.session_state.results = run_simulation(n)
            st.session_state.n = n
        except Exception as exc:
            st.error(f"Reconstruction failed: {exc}")
            st.stop()

original, A, ray_labels, reconstructed, matrices, b, metrics = st.session_state.results
mse, rmse, max_error = metrics

c1, c2, c3, c4 = st.columns(4)
c1.metric("Voxels", f"{n**3:,}")
c2.metric("Projection matrix A", f"{A.shape[0]} × {A.shape[1]}")
c3.metric("Rank(A)", f"{np.linalg.matrix_rank(A)}")
c4.metric("MSE", f"{mse:.3e}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🩻 X-rays", "📐 Matrices", "🔄 Reconstruction", "🧊 3D CT", "📚 Mathematics"])

with tab1:
    st.header("Generated 2D X-ray Projections")
    st.write("Each selected projection is a simulated detector image from a different direction.")
    angle_index = st.selectbox("Projection angle", range(len(DEFAULT_ANGLES)),
                               format_func=lambda i: f"{DEFAULT_ANGLES[i]:.0f}°")
    angle = float(DEFAULT_ANGLES[angle_index])
    image = projection_image(original[original.shape[0] // 2], angle, output_size=n)
    fig = xray_figure(image, angle)
    st.pyplot(fig, width="stretch")
    st.download_button("⬇️ Download selected X-ray", figure_to_png(fig),
                       file_name=f"xray_{angle:.0f}_degrees.png", mime="image/png")

    st.subheader("Sinogram")
    sino_fig = sinogram_figure(original, DEFAULT_ANGLES)
    st.pyplot(sino_fig, width="stretch")
    st.download_button("⬇️ Download sinogram", figure_to_png(sino_fig),
                       file_name="sinogram.png", mime="image/png")

    st.subheader("Download all generated X-ray projections")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for a in DEFAULT_ANGLES:
            f = xray_figure(projection_image(original[original.shape[0] // 2], float(a), output_size=n), float(a))
            z.writestr(f"xray_{a:.0f}_degrees.png", figure_to_png(f))
    st.download_button("📦 Download all X-rays as ZIP", zip_buffer.getvalue(),
                       file_name="generated_xrays.zip", mime="application/zip")

with tab2:
    st.header("Projection Matrix")
    st.latex(r"Ax=b")
    st.write(f"A has **{A.shape[0]} rows** (ray measurements) and **{A.shape[1]} columns** (unknown pixels in one slice).")
    st.write(f"Rank(A) = **{np.linalg.matrix_rank(A)}**")
    st.dataframe(A, height=420, use_container_width=True)
    st.subheader("Normal equations")
    st.latex(r"A^TAx=A^Tb")
    with st.expander("Show AᵀA"):
        st.dataframe(matrices[0][0], height=420, use_container_width=True)
    with st.expander("Show Aᵀb"):
        st.dataframe(matrices[0][1].reshape(-1, 1), height=420, use_container_width=True)

with tab3:
    st.header("Original vs Reconstructed")
    slice_index = st.slider("Slice", 1, n, n // 2 + 1) - 1
    fig = comparison_figure(original, reconstructed, slice_index)
    st.pyplot(fig, width="stretch")
    st.download_button("⬇️ Download comparison", figure_to_png(fig),
                       file_name=f"comparison_slice_{slice_index+1}.png", mime="image/png")
    st.metric("RMSE", f"{rmse:.6e}")
    st.metric("Maximum absolute voxel error", f"{max_error:.6e}")

    slice_fig = slice_figure(reconstructed, slice_index, "Reconstructed CT")
    st.download_button("⬇️ Download reconstructed slice", figure_to_png(slice_fig),
                       file_name=f"reconstructed_slice_{slice_index+1}.png", mime="image/png")

with tab4:
    st.header("Final 3D CT Volume")
    st.write(f"Reconstructed volume: **{n} × {n} × {n} voxels**.")
    fig3d = volume_3d_figure(reconstructed, threshold)
    st.pyplot(fig3d, width="stretch")
    st.download_button("⬇️ Download final 3D CT image", figure_to_png(fig3d, dpi=220),
                       file_name="final_3d_ct.png", mime="image/png")
    st.info("The 3D view uses voxel faces rather than a point cloud, so the reconstructed structure is easier to see.")

with tab5:
    st.header("Mathematics Used")
    st.subheader("1. Projection model")
    st.latex(r"Ax=b")
    st.write("A contains the ray/voxel relationships, x contains the unknown pixel values of one slice, and b contains the simulated X-ray measurements.")
    st.subheader("2. Least squares")
    st.latex(r"\min_x\|Ax-b\|^2")
    st.subheader("3. Normal equations")
    st.latex(r"A^TAx=A^Tb")
    st.write("The program solves this system using Gaussian elimination. No pseudoinverse is used.")
    st.subheader("4. Rank")
    st.latex(rf"\operatorname{{rank}}(A)={np.linalg.matrix_rank(A)}")
    if show_svd:
        s, rank = svd_analysis(A)
        st.subheader("5. Optional SVD")
        st.latex(r"A=U\Sigma V^T")
        st.write("SVD is included only as an analysis because it is explicitly in your Unit 2 syllabus.")
        st.dataframe(s.reshape(-1, 1), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Educational linear-algebra CT simulation")
