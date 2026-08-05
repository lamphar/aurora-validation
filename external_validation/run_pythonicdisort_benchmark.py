#!/usr/bin/env python3
"""Matched 1-D external validation of the AURORA single-scattering reduction.

This benchmark compares two solutions of exactly the same plane-parallel,
axisymmetric radiative-transfer problem:

1. PythonicDISORT: the recognized discrete-ordinates multiple-scattering
   solution.
2. AURORA column reduction: the first-scattering term evaluated on the same
   optical-depth grid, phase-function moments, quadrature ordinates, black
   boundaries, and unit-normalized lower-boundary source.

The comparison is intentionally limited to a 1-D column-equivalent problem.
It does not validate AURORA's separate localized-source geometry (horizontal
range, source area, or a 3-D heterogeneous atmosphere). Run 4 of the workflow
is retained in the repository history and in external_validation/diagnostics
as a failed diagnostic because it compared unmatched geometries and profiles.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legvander

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aurora_source"))

import PythonicDISORT
from aurora_inverse.v3 import V3Config, angular_distribution

WAVELENGTHS = np.array([450.0, 589.0, 650.0])
TOP_M = 12_000.0
RAYLEIGH_550_M_INV = 1.20e-5
MOLECULAR_SCALE_HEIGHT_M = 8_000.0
AEROSOL_SCALE_HEIGHT_M = 1_500.0
AEROSOL_ANGSTROM = 1.30

# These are the same numerical gate values used by the failed run-4 benchmark.
# The redesign changes the physical correspondence, not the acceptance limits.
MAX_PAIRED_RMSE = 0.15
MIN_VALIDITY_FRACTION_ALL_CASES = 0.35
MAX_CONVERGENCE_P95 = 0.03
MAX_VALIDITY_SCATTER_ERROR = 0.10
MAX_VALIDITY_TOTAL_ERROR = 0.15


def _stratified_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic Latin-hypercube-like points on the unit hypercube."""
    u = np.empty((n, d), dtype=float)
    for j in range(d):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u


def build_cases(n: int, seed: int) -> list[dict[str, float | int | str]]:
    """Build a preregistered thin-domain validation set plus a stress set."""
    if n < 24:
        raise ValueError("At least 24 cases are required for the matched benchmark.")

    n_thin = max(16, int(round(2 * n / 3)))
    n_stress = n - n_thin
    rng = np.random.default_rng(seed)
    cases: list[dict[str, float | int | str]] = []

    # A transparent-to-thick physical ladder. The first six entries are in the
    # thin validation domain; the remaining entries deliberately probe where
    # multiple scattering should separate from the first-order reduction.
    ladder = [
        ("rayleigh_650", "thin_validation", 650.0, 0.000, 0.00, 0.75, 0.00, 0.90),
        ("isotropic_650", "thin_validation", 650.0, 0.004, 0.00, 0.80, 0.05, 0.80),
        ("hg_weak_650", "thin_validation", 650.0, 0.010, 0.45, 0.85, 0.10, 0.70),
        ("hg_thin_650", "thin_validation", 650.0, 0.020, 0.65, 0.90, 0.18, 0.55),
        ("hg_thin_589", "thin_validation", 589.0, 0.020, 0.60, 0.88, 0.25, 0.75),
        ("thin_edge_589", "thin_validation", 589.0, 0.035, 0.72, 0.94, 0.32, 0.40),
        ("blue_molecular", "stress", 450.0, 0.000, 0.00, 0.80, 0.10, 0.70),
        ("moderate_aerosol", "stress", 650.0, 0.100, 0.65, 0.92, 0.20, 0.60),
        ("moderate_blue", "stress", 450.0, 0.080, 0.70, 0.94, 0.25, 0.50),
        ("thick_aerosol", "stress", 589.0, 0.250, 0.75, 0.97, 0.30, 0.45),
        ("thick_forward", "stress", 650.0, 0.400, 0.85, 0.98, 0.35, 0.35),
        ("thick_blue", "stress", 450.0, 0.350, 0.80, 0.98, 0.30, 0.40),
    ]

    thin_ladder = [row for row in ladder if row[1] == "thin_validation"]
    stress_ladder = [row for row in ladder if row[1] == "stress"]

    def append_ladder(row: tuple[str, str, float, float, float, float, float, float]) -> None:
        label, regime, wavelength, aod, g, ssa, phi, mu = row
        cases.append(
            {
                "case_id": len(cases),
                "label": label,
                "regime": regime,
                "wavelength_nm": wavelength,
                "aod_550": aod,
                "asymmetry": g,
                "aerosol_single_scattering_albedo": ssa,
                "emission_phi": phi,
                "mu_view": mu,
            }
        )

    for row in thin_ladder[: min(len(thin_ladder), n_thin)]:
        append_ladder(row)

    remaining_thin = n_thin - sum(c["regime"] == "thin_validation" for c in cases)
    if remaining_thin:
        u = _stratified_unit(remaining_thin, 6, rng)
        for x in u:
            wavelength = 589.0 if x[0] < 0.5 else 650.0
            cases.append(
                {
                    "case_id": len(cases),
                    "label": "thin_lhs",
                    "regime": "thin_validation",
                    "wavelength_nm": wavelength,
                    "aod_550": 0.001 + 0.034 * x[1],
                    "asymmetry": 0.05 + 0.70 * x[2],
                    "aerosol_single_scattering_albedo": 0.72 + 0.23 * x[3],
                    "emission_phi": 0.34 * x[4],
                    "mu_view": 0.38 + 0.57 * x[5],
                }
            )

    for row in stress_ladder[: min(len(stress_ladder), n_stress)]:
        append_ladder(row)

    remaining_stress = n_stress - sum(c["regime"] == "stress" for c in cases)
    if remaining_stress:
        u = _stratified_unit(remaining_stress, 6, rng)
        for x in u:
            wavelength = float(WAVELENGTHS[min(int(3 * x[0]), 2)])
            cases.append(
                {
                    "case_id": len(cases),
                    "label": "stress_lhs",
                    "regime": "stress",
                    "wavelength_nm": wavelength,
                    "aod_550": 0.050 + 0.400 * x[1],
                    "asymmetry": 0.25 + 0.62 * x[2],
                    "aerosol_single_scattering_albedo": 0.78 + 0.21 * x[3],
                    "emission_phi": 0.40 * x[4],
                    "mu_view": 0.30 + 0.65 * x[5],
                }
            )

    if len(cases) != n:
        raise RuntimeError(f"Case construction produced {len(cases)} rather than {n} cases.")
    return cases


def _exponential_layer_integral(beta0: float, scale_height_m: float, z_bottom: np.ndarray, z_top: np.ndarray) -> np.ndarray:
    return beta0 * scale_height_m * (
        np.exp(-z_bottom / scale_height_m) - np.exp(-z_top / scale_height_m)
    )


def matched_layer_optics(
    case: dict[str, float | int | str], n_layers: int, n_leg: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Create the same exponential atmosphere used by the AURORA reduction."""
    wavelength = float(case["wavelength_nm"])
    aod_550 = float(case["aod_550"])
    aerosol_ssa = float(case["aerosol_single_scattering_albedo"])
    g = float(case["asymmetry"])

    # PythonicDISORT layers are ordered top to bottom and tau increases downward.
    z_edges = np.linspace(TOP_M, 0.0, n_layers + 1)
    z_top = z_edges[:-1]
    z_bottom = z_edges[1:]

    beta_r0 = RAYLEIGH_550_M_INV * (550.0 / wavelength) ** 4
    beta_a0 = (aod_550 / AEROSOL_SCALE_HEIGHT_M) * (550.0 / wavelength) ** AEROSOL_ANGSTROM
    tau_r = _exponential_layer_integral(beta_r0, MOLECULAR_SCALE_HEIGHT_M, z_bottom, z_top)
    tau_a = _exponential_layer_integral(beta_a0, AEROSOL_SCALE_HEIGHT_M, z_bottom, z_top)

    tau_layer = tau_r + tau_a
    scattering_tau = tau_r + aerosol_ssa * tau_a
    omega = np.divide(scattering_tau, tau_layer, out=np.zeros_like(tau_layer), where=tau_layer > 0)
    omega = np.clip(omega, 0.0, 1.0 - 1e-8)

    coeff = np.zeros((n_layers, n_leg), dtype=float)
    coeff[:, 0] = 1.0
    for ell in range(1, n_leg):
        rayleigh_coeff = 0.5 if ell == 2 else 0.0
        numerator = tau_r * rayleigh_coeff + aerosol_ssa * tau_a * g**ell
        coeff[:, ell] = np.divide(
            numerator, scattering_tau, out=np.zeros_like(numerator), where=scattering_tau > 0
        )
    coeff[:, 1:] = np.clip(coeff[:, 1:], -0.999999, 0.999999)

    tau_arr = np.cumsum(tau_layer)
    diagnostics = {
        "rayleigh_optical_depth": float(tau_r.sum()),
        "aerosol_optical_depth_at_wavelength": float(tau_a.sum()),
        "total_optical_depth": float(tau_arr[-1]),
        "column_scattering_albedo": float(scattering_tau.sum() / max(tau_layer.sum(), 1e-300)),
    }
    return tau_arr, omega, coeff, diagnostics


def boundary_source(mu: np.ndarray | float, emission_phi: float) -> np.ndarray:
    mu_arr = np.asarray(mu, dtype=float)
    theta = np.arccos(np.clip(mu_arr, 0.0, 1.0))
    return np.asarray(angular_distribution(theta, np.full_like(theta, emission_phi)), dtype=float)


def run_disort(
    tau_arr: np.ndarray,
    omega: np.ndarray,
    coeff: np.ndarray,
    emission_phi: float,
    mu_view: float,
    n_quad: int,
    n_leg: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return full and no-scattering top intensities at the same view cosine."""
    n = n_quad // 2
    mu_pos, weights = PythonicDISORT.subroutines.Gauss_Legendre_quad(n)
    b_pos = boundary_source(mu_pos, emission_phi)

    def solve(omega_arg: np.ndarray) -> float:
        _, _, _, _, u = PythonicDISORT.pydisort(
            tau_arr,
            omega_arg,
            n_quad,
            coeff,
            0.5,
            0.0,
            0.0,
            NLeg=n_leg,
            NFourier=1,
            b_pos=b_pos,
            b_neg=0.0,
            BDRF_Fourier_modes=[],
        )
        u_interp = PythonicDISORT.subroutines.interpolate(u)
        return float(np.asarray(u_interp(mu_view, 0.0, 0.0)).squeeze())

    full_total = solve(omega)
    direct_only = solve(np.zeros_like(omega))
    return full_total, direct_only, mu_pos, weights


def aurora_column_single_scatter(
    tau_arr: np.ndarray,
    omega: np.ndarray,
    coeff: np.ndarray,
    emission_phi: float,
    mu_view: float,
    mu_in: np.ndarray,
    weights: np.ndarray,
) -> float:
    """First scattering of the matched axisymmetric lower-boundary source.

    The calculation uses the same discrete-ordinate incoming directions and
    phase-function Legendre moments as PythonicDISORT. It is therefore a clean
    first-order reduction of the same 1-D problem, rather than the localized
    urban geometry used in the failed run-4 diagnostic.
    """
    n_leg = coeff.shape[1]
    ell = np.arange(n_leg)
    p_out = legvander(mu_view, n_leg - 1).reshape(-1)
    p_in = legvander(mu_in, n_leg - 1)
    b_in = boundary_source(mu_in, emission_phi)
    total_tau = float(tau_arr[-1])
    layer_top = np.concatenate([[0.0], tau_arr[:-1]])
    layer_bottom = tau_arr

    result = 0.0
    inverse_out = 1.0 / mu_view
    for k, (a, b) in enumerate(zip(layer_top, layer_bottom, strict=True)):
        phase_weights = (2 * ell + 1) * coeff[k] * p_out
        phase_kernel = p_in @ phase_weights
        rate = 1.0 / mu_in - inverse_out

        integral = np.empty_like(rate)
        close = np.abs(rate) < 1e-11
        integral[close] = np.exp(-total_tau / mu_in[close]) * (b - a)
        not_close = ~close
        integral[not_close] = (
            np.exp(-total_tau / mu_in[not_close])
            * (np.exp(rate[not_close] * b) - np.exp(rate[not_close] * a))
            / rate[not_close]
        )

        result += (
            0.5
            * omega[k]
            * inverse_out
            * float(np.sum(weights * phase_kernel * b_in * integral))
        )
    return max(float(result), 1e-300)


def relative_error(estimate: float, reference: float) -> float:
    return (estimate - reference) / max(abs(reference), 1e-300)


def evaluate_resolution(
    case: dict[str, float | int | str], n_quad: int, n_layers: int, n_leg: int
) -> dict[str, float]:
    tau_arr, omega, coeff, diagnostics = matched_layer_optics(case, n_layers, n_leg)
    emission_phi = float(case["emission_phi"])
    mu_view = float(case["mu_view"])
    full_total, direct_only, mu_in, weights = run_disort(
        tau_arr, omega, coeff, emission_phi, mu_view, n_quad, n_leg
    )
    full_scattered = max(full_total - direct_only, 1e-300)
    first_scattered = aurora_column_single_scatter(
        tau_arr, omega, coeff, emission_phi, mu_view, mu_in, weights
    )
    first_total = direct_only + first_scattered
    return {
        **diagnostics,
        "disort_total_transfer": full_total,
        "disort_direct_transfer": direct_only,
        "disort_scattered_transfer": full_scattered,
        "aurora_first_scattered_transfer": first_scattered,
        "aurora_first_total_transfer": first_total,
        "scattered_relative_error": relative_error(first_scattered, full_scattered),
        "total_relative_error": relative_error(first_total, full_total),
        "multiple_scattering_enhancement": full_scattered / first_scattered,
    }


def evaluate_case(case: dict[str, float | int | str]) -> dict[str, float | int | str | bool]:
    baseline = evaluate_resolution(case, n_quad=32, n_layers=48, n_leg=24)
    high = evaluate_resolution(case, n_quad=48, n_layers=72, n_leg=32)

    convergence = abs(
        high["disort_scattered_transfer"] - baseline["disort_scattered_transfer"]
    ) / max(abs(high["disort_scattered_transfer"]), 1e-300)
    single_convergence = abs(
        high["aurora_first_scattered_transfer"] - baseline["aurora_first_scattered_transfer"]
    ) / max(abs(high["aurora_first_scattered_transfer"]), 1e-300)

    out: dict[str, float | int | str | bool] = dict(case)
    out.update(baseline)
    out["high_disort_scattered_transfer"] = high["disort_scattered_transfer"]
    out["high_aurora_first_scattered_transfer"] = high["aurora_first_scattered_transfer"]
    out["numerical_convergence_error"] = convergence
    out["single_scatter_convergence_error"] = single_convergence
    out["gate_eligible"] = case["regime"] == "thin_validation"
    out["validity_domain_pass"] = bool(
        out["gate_eligible"]
        and abs(float(out["scattered_relative_error"])) <= MAX_VALIDITY_SCATTER_ERROR
        and abs(float(out["total_relative_error"])) <= MAX_VALIDITY_TOTAL_ERROR
        and convergence <= MAX_CONVERGENCE_P95
    )
    return out


def make_figure(df: pd.DataFrame, output: Path) -> None:
    thin = df["regime"] == "thin_validation"
    stress = ~thin
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)

    ax = axes[0]
    ax.scatter(
        df.loc[thin, "disort_scattered_transfer"],
        df.loc[thin, "aurora_first_scattered_transfer"],
        s=22,
        alpha=0.8,
        label="thin validation",
    )
    ax.scatter(
        df.loc[stress, "disort_scattered_transfer"],
        df.loc[stress, "aurora_first_scattered_transfer"],
        s=18,
        alpha=0.55,
        marker="x",
        label="stress",
    )
    lo = min(df["disort_scattered_transfer"].min(), df["aurora_first_scattered_transfer"].min())
    hi = max(df["disort_scattered_transfer"].max(), df["aurora_first_scattered_transfer"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PythonicDISORT scattered transfer")
    ax.set_ylabel("AURORA first-scatter transfer")
    ax.set_title("a  Matched 1-D transfer")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.scatter(
        df.loc[thin, "total_optical_depth"],
        100 * np.abs(df.loc[thin, "scattered_relative_error"]),
        s=22,
        alpha=0.8,
    )
    ax.scatter(
        df.loc[stress, "total_optical_depth"],
        100 * np.abs(df.loc[stress, "scattered_relative_error"]),
        s=18,
        alpha=0.55,
        marker="x",
    )
    ax.axhline(100 * MAX_VALIDITY_SCATTER_ERROR, linestyle="--", linewidth=1)
    ax.set_xlabel("Total vertical optical depth")
    ax.set_ylabel("First-order discrepancy (%)")
    ax.set_title("b  Domain of validity")

    ax = axes[2]
    ax.scatter(
        df["total_optical_depth"],
        df["multiple_scattering_enhancement"],
        s=21,
        alpha=0.75,
    )
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Total vertical optical depth")
    ax.set_ylabel("Full / first-scatter transfer")
    ax.set_title("c  Multiple-scattering growth")

    fig.savefig(output / "external_solver_benchmark.png", dpi=300)
    fig.savefig(output / "external_solver_benchmark.pdf")
    plt.close(fig)


def summarize(df: pd.DataFrame, metadata: dict[str, object], runtime_seconds: float) -> dict[str, object]:
    thin = df[df["regime"] == "thin_validation"].copy()
    stress = df[df["regime"] == "stress"].copy()
    if thin.empty:
        raise RuntimeError("No thin-domain validation cases were generated.")

    thin_scatter_errors = thin["scattered_relative_error"].to_numpy(float)
    thin_total_errors = thin["total_relative_error"].to_numpy(float)
    paired_rmse = float(np.sqrt(np.mean(thin_scatter_errors**2)))
    convergence_p95 = float(np.quantile(thin["numerical_convergence_error"], 0.95))
    validity_all = float(df["validity_domain_pass"].mean())
    validity_thin = float(thin["validity_domain_pass"].mean())

    summary: dict[str, object] = {
        "solver": "PythonicDISORT",
        "solver_version": metadata["pythonicdisort_version"],
        "solver_class": "recognized plane-parallel multiple-scattering discrete-ordinates solver",
        "benchmark_kind": "matched 1-D axisymmetric first-scattering correspondence",
        "mapping_scope": "same exponential column, black boundaries, common phase moments and unit lower-boundary source; localized 3-D geometry excluded",
        "legacy_run4_diagnostic": {
            "workflow_run": 30999462835,
            "commit": "63cbdd92ce079685490c4a341df0158b732a5eed",
            "status": "failed physical-correspondence diagnostic retained; not overwritten",
        },
        "physics_alignment": [
            "identical cumulative optical-depth layers",
            "identical molecular and aerosol exponential profiles",
            "identical single-scattering albedo by layer",
            "identical Rayleigh plus Henyey-Greenstein Legendre moments",
            "axisymmetric unit-normalized lower-boundary source",
            "black upper and lower reflective boundaries",
            "same viewing cosine and wavelength",
            "comparison of scattered transfer after subtracting the matched no-scattering solution",
        ],
        "excluded_from_this_1d_gate": [
            "horizontal source-observer range",
            "finite localized source area",
            "relative azimuth",
            "3-D heterogeneity",
            "surface reflection",
        ],
        "cases": int(len(df)),
        "thin_validation_cases": int(len(thin)),
        "stress_cases": int(len(stress)),
        "paired_thin_domain_relative_rmse": paired_rmse,
        # Compatibility field retained so the numerical threshold is visibly unchanged.
        "selected_holdout_relative_rmse": paired_rmse,
        "thin_median_absolute_scattered_relative_error": float(np.median(np.abs(thin_scatter_errors))),
        "thin_p95_absolute_scattered_relative_error": float(np.quantile(np.abs(thin_scatter_errors), 0.95)),
        "thin_total_transfer_relative_rmse": float(np.sqrt(np.mean(thin_total_errors**2))),
        "thin_numerical_convergence_p95": convergence_p95,
        "numerical_convergence_p95": convergence_p95,
        "thin_single_scatter_convergence_p95": float(
            np.quantile(thin["single_scatter_convergence_error"], 0.95)
        ),
        "validity_domain_fraction_all_cases": validity_all,
        "validity_domain_fraction_thin_cases": validity_thin,
        "validity_domain_rule": "thin preregistered case AND |scattered transfer error| <= 10% AND |total transfer error| <= 15% AND numerical convergence error <= 3%",
        "stress_median_multiple_scattering_enhancement": float(
            np.median(stress["multiple_scattering_enhancement"]) if len(stress) else math.nan
        ),
        "all_external_outputs_finite_positive": bool(
            np.isfinite(df["disort_total_transfer"]).all()
            and np.isfinite(df["disort_scattered_transfer"]).all()
            and (df["disort_total_transfer"] > 0).all()
            and (df["disort_scattered_transfer"] > 0).all()
        ),
        "thresholds_unchanged_from_run4": {
            "maximum_paired_relative_rmse": MAX_PAIRED_RMSE,
            "minimum_validity_fraction_all_cases": MIN_VALIDITY_FRACTION_ALL_CASES,
            "maximum_numerical_convergence_p95": MAX_CONVERGENCE_P95,
            "validity_scattered_error": MAX_VALIDITY_SCATTER_ERROR,
            "validity_total_error": MAX_VALIDITY_TOTAL_ERROR,
        },
        "runtime_seconds": runtime_seconds,
    }
    summary["validation_pass"] = bool(
        summary["all_external_outputs_finite_positive"]
        and paired_rmse <= MAX_PAIRED_RMSE
        and validity_all >= MIN_VALIDITY_FRACTION_ALL_CASES
        and convergence_p95 <= MAX_CONVERGENCE_P95
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=72)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    metadata: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pythonicdisort_version": importlib.metadata.version("PythonicDISORT"),
        "numpy_version": np.__version__,
        "seed": args.seed,
        "requested_cases": args.cases,
        "baseline_resolution": {"streams": 32, "layers": 48, "legendre_moments": 24},
        "high_resolution": {"streams": 48, "layers": 72, "legendre_moments": 32},
    }
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        cases = build_cases(args.cases, args.seed)
        rows: list[dict[str, float | int | str | bool]] = []
        for case in cases:
            print(
                f"case {int(case['case_id']) + 1}/{args.cases}: {case['regime']} {case['label']}",
                flush=True,
            )
            rows.append(evaluate_case(case))
        df = pd.DataFrame(rows)
        runtime = time.time() - started
        summary = summarize(df, metadata, runtime)

        df.to_csv(args.output / "benchmark_cases.csv", index=False)
        (args.output / "benchmark_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        design = {
            "seed": args.seed,
            "cases": cases,
            "baseline_resolution": metadata["baseline_resolution"],
            "high_resolution": metadata["high_resolution"],
            "gate_thresholds": summary["thresholds_unchanged_from_run4"],
            "gate_domain": "thin_validation",
            "stress_cases_are_diagnostic_not_gate_eligible": True,
        }
        (args.output / "benchmark_design.json").write_text(
            json.dumps(design, indent=2), encoding="utf-8"
        )
        make_figure(df, args.output)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        failure = {
            "validation_pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "metadata": metadata,
            "runtime_seconds": time.time() - started,
        }
        (args.output / "benchmark_summary.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())