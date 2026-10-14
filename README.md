# Eigenslice

**Eigenslice: reconstruct CT volumes from X-ray projections, compare QR, SVD, and regularized solvers, and see where each one fails.**

Eigenslice is a linear-algebra lab for computed tomography. It simulates X-ray projections of a synthetic 3D phantom, reconstructs the volume with several solvers, and shows why each solver succeeds or fails. The course topics it builds on are Gram-Schmidt/QR, eigenvalues and eigenvectors, PCA, and SVD.

> **Status: in progress.** This repository currently holds the original v3 prototype and the Unit 2 question sheets. The v4 lab described in the [game plan](docs/game-plan.md) is not built yet.

## Read first

- [docs/game-plan.md](docs/game-plan.md): audit of the v3 prototype, the v4 target, architecture, data model, and the phased roadmap.
- [legacy-v3/](legacy-v3/): the original Streamlit prototype, kept unchanged as the baseline for the audit.
- [docs/unit2/](docs/unit2/): the four Unit 2 question sheets the solvers map onto.

## Known limitations of v3

v3 is a useful baseline but has measured problems, all listed in the game plan:

- Its projection matrix measures almost every pixel directly, so the reconstruction error is about 1e-31. That is floating-point noise, not accuracy.
- Data and solve use the same matrix with no noise, so the test is rigged.
- It solves with normal equations and Gaussian elimination, which squares the condition number.
- It is slice-by-slice parallel-beam, not true 3D.

## Planned layout (v4)

```
src/eigenslice/    maths package: operators, solvers, metrics, storage
app/               Streamlit pages
tests/             pytest, one file per module
supabase/          SQL migrations for the cloud tables and storage
docs/              game plan and Unit 2 question sheets
legacy-v3/         original prototype, kept as the audit baseline
```

## Run the v3 prototype

```bash
cd legacy-v3
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Licence and course material

The files in `docs/unit2/` are course question sheets. Check your course rules before making this repository public.
