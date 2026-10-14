# Matrix CT Reconstruction — Streamlit

An educational linear-algebra project showing how multiple simulated 2D X-ray projections can be used to reconstruct a 3D voxel volume.

## Main equations

Ax = b

min ||Ax-b||²

AᵀAx = Aᵀb

The normal equations are solved using Gaussian elimination. No pseudoinverse is used.

## Run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Files

- `app.py` — Streamlit UI and downloads
- `phantom.py` — synthetic 3D object
- `projection.py` — simplified X-ray/projection matrix model
- `reconstruction.py` — least squares + Gaussian elimination
- `visualization.py` — X-ray, sinogram, slice and 3D visualizations

## Important limitation

This is a mathematical simulation, not a clinical CT reconstruction system. To keep the mathematics manageable and aligned with a linear-algebra syllabus, reconstruction is performed slice-by-slice using a simplified 2D projection model and the slices are then stacked into a 3D volume.
