# AURORA external validation on GitHub Actions

This repository starter executes a **real PythonicDISORT 1.7** benchmark on GitHub's hosted Ubuntu runner. It produces a complete downloadable evidence artifact and fails the workflow when the preregistered scientific criteria are not met.

## Upload through the GitHub website

1. Download and extract this starter ZIP locally.
2. Open your empty `aurora-validation` repository.
3. Select **Add file → Upload files**.
4. Drag **the contents of the extracted folder** into the upload area. The hidden `.github` folder must be included.
5. Commit directly to `main`.
6. Open the repository's **Actions** tab.
7. Select **AURORA external physical validation** and choose **Run workflow**.
8. When the run finishes, open it and download the artifact named `aurora-pythonicdisort-validation`.

## What the artifact contains

- exact PythonicDISORT and dependency versions;
- the deterministic benchmark factor matrix and split;
- case-level external and reduced-model radiances;
- derivative, convergence, retrieval and coverage diagnostics;
- two discrepancy alternatives evaluated out of sample;
- the quantitative validity-domain rule and PASS/FAIL decision;
- execution logs, workflow provenance, figure, CSV/JSON results and SHA-256 manifest.

## Scientific limitation

PythonicDISORT is a recognized 1-D plane-parallel multiple-scattering solver. This benchmark maps the urban source to an effective lower-boundary radiance. It validates the column-equivalent atmospheric transfer and establishes a quantitative validity domain, but it does **not** pretend to reproduce arbitrary three-dimensional localized-source transport.
