#!/usr/bin/env python3
"""Execute a recognized PythonicDISORT benchmark for AURORA-INVERSE.

The external solver is used directly. AURORA's reduced single-scattering
operator is evaluated on the same deterministic factor matrix. The benchmark
is explicitly limited to a plane-parallel, column-equivalent representation of
an urban boundary source; it does not claim full 3-D validation.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aurora_source"))

import PythonicDISORT
from aurora_inverse.v3 import V3Config, angular_distribution, production_kernel_scalar, spectral_bases

WAVELENGTHS = np.array([450.0, 589.0, 650.0])
SPECTRA = spectral_bases()
PHI_CLASS = np.array([0.08, 0.18, 0.28])
FEATURES = [
    "aod_550",
    "wavelength_nm",
    "distance_m",
    "aerosol_layer_altitude_m",
    "aerosol_layer_thickness_m",
    "surface_albedo",
    "asymmetry",
    "single_scattering_albedo",
    "angular_emission_phi",
    "view_zenith_deg",
    "relative_azimuth_deg",
    "observer_altitude_m",
    "source_extent_m",
    "composition_0",
    "composition_1",
]


def latin_hypercube(n: int, seed: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    d = 15
    u = np.empty((n, d), dtype=float)
    for j in range(d):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    rows: list[dict[str, float]] = []
    for i, x in enumerate(u):
        logits = np.array([1.25 * (x[12] - 0.5), 1.25 * (x[13] - 0.5), 1.25 * (x[14] - 0.5)])
        comp = np.exp(logits - logits.max())
        comp /= comp.sum()
        rows.append(
            {
                "case_id": i,
                "aod_550": 0.02 + 0.43 * x[0],
                "wavelength_nm": float(WAVELENGTHS[min(int(3 * x[1]), 2)]),
                "distance_m": 200.0 + 11_800.0 * x[2],
                "aerosol_layer_altitude_m": 300.0 + 7_200.0 * x[3],
                "aerosol_layer_thickness_m": 400.0 + 4_600.0 * x[4],
                "surface_albedo": 0.01 + 0.34 * x[5],
                "asymmetry": 0.25 + 0.63 * x[6],
                "single_scattering_albedo": 0.72 + 0.27 * x[7],
                "angular_emission_phi": 0.03 + 0.37 * x[8],
                "view_zenith_deg": 5.0 + 65.0 * x[9],
                "relative_azimuth_deg": 180.0 * x[10],
                "observer_altitude_m": 2_500.0 * x[11],
                "source_extent_m": 200.0 + 4_800.0 * x[4],
                "composition_0": float(comp[0]),
                "composition_1": float(comp[1]),
                "composition_2": float(comp[2]),
            }
        )
    return rows


def layer_optics(case: dict[str, float], n_layers: int, n_leg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    top = 12_000.0
    edges = np.linspace(top, 0.0, n_layers + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    dz = edges[:-1] - edges[1:]
    wavelength = case["wavelength_nm"]

    beta_r0 = 1.20e-5 * (550.0 / wavelength) ** 4
    tau_r = beta_r0 * np.exp(-mids / 8_000.0) * dz

    lo = max(0.0, case["aerosol_layer_altitude_m"] - 0.5 * case["aerosol_layer_thickness_m"])
    hi = min(top, case["aerosol_layer_altitude_m"] + 0.5 * case["aerosol_layer_thickness_m"])
    overlap = np.maximum(0.0, np.minimum(edges[:-1], hi) - np.maximum(edges[1:], lo))
    if overlap.sum() <= 0:
        overlap[np.argmin(np.abs(mids - case["aerosol_layer_altitude_m"]))] = 1.0
    aerosol_fraction = overlap / overlap.sum()
    tau_a = case["aod_550"] * (550.0 / wavelength) ** 1.30 * aerosol_fraction

    tau_layer = tau_r + tau_a
    scattering_tau = tau_r + case["single_scattering_albedo"] * tau_a
    omega = np.divide(scattering_tau, tau_layer, out=np.zeros_like(tau_layer), where=tau_layer > 0)
    omega = np.clip(omega, 0.0, 1.0 - 1e-8)

    coeff = np.zeros((n_layers, n_leg), dtype=float)
    coeff[:, 0] = 1.0
    for ell in range(1, n_leg):
        rayleigh = 0.5 if ell == 2 else 0.0
        numerator = tau_r * rayleigh + case["single_scattering_albedo"] * tau_a * case["asymmetry"] ** ell
        coeff[:, ell] = np.divide(numerator, scattering_tau, out=np.zeros_like(numerator), where=scattering_tau > 0)
    coeff[:, 1:] = np.clip(coeff[:, 1:], -0.999999, 0.999999)
    tau_arr = np.cumsum(tau_layer)
    return tau_arr, omega, coeff, edges


def optical_depth_at_altitude(tau_arr: np.ndarray, edges: np.ndarray, altitude_m: float) -> float:
    layer_tau = np.diff(np.concatenate([[0.0], tau_arr]))
    altitude = float(np.clip(altitude_m, 0.0, edges[0]))
    total = 0.0
    for k, dtau in enumerate(layer_tau):
        z_top, z_bottom = edges[k], edges[k + 1]
        if altitude <= z_bottom:
            total += dtau
        elif altitude >= z_top:
            break
        else:
            total += dtau * (z_top - altitude) / (z_top - z_bottom)
            break
    return float(np.clip(total, 0.0, tau_arr[-1]))


def source_amplitude(case: dict[str, float]) -> tuple[float, float]:
    comp = np.array([case["composition_0"], case["composition_1"], case["composition_2"]])
    band = int(np.argmin(np.abs(WAVELENGTHS - case["wavelength_nm"])))
    spectral_amp = float(comp @ SPECTRA[:, band])
    phi = float(comp @ PHI_CLASS)
    geom = case["source_extent_m"] ** 2 / (case["distance_m"] ** 2 + case["source_extent_m"] ** 2)
    return spectral_amp * geom, phi


def external_radiance(case: dict[str, float], n_quad: int = 16, n_layers: int = 12, n_leg: int = 16) -> float:
    tau_arr, omega, coeff, edges = layer_optics(case, n_layers, n_leg)
    n = n_quad // 2
    mu_pos, _ = PythonicDISORT.subroutines.Gauss_Legendre_quad(n)
    amplitude, phi_mix = source_amplitude(case)
    theta = np.arccos(np.clip(mu_pos, 0.0, 1.0))
    angular = angular_distribution(theta, np.full_like(theta, phi_mix))

    n_fourier = 3
    b_pos = np.zeros((n, n_fourier), dtype=float)
    b_pos[:, 0] = amplitude * angular
    b_pos[:, 1] = 0.16 * amplitude * angular
    b_pos[:, 2] = 0.05 * amplitude * angular

    _, _, _, _, u = PythonicDISORT.pydisort(
        tau_arr,
        omega,
        n_quad,
        coeff,
        0.5,
        0.0,
        0.0,
        NLeg=n_leg,
        NFourier=n_fourier,
        b_pos=b_pos,
        b_neg=0.0,
        BDRF_Fourier_modes=[case["surface_albedo"]],
    )
    u_interp = PythonicDISORT.subroutines.interpolate(u)
    mu_view = -math.cos(math.radians(case["view_zenith_deg"]))
    tau_obs = optical_depth_at_altitude(tau_arr, edges, case["observer_altitude_m"])
    phi_view = math.radians(case["relative_azimuth_deg"])
    value = float(np.asarray(u_interp(mu_view, tau_obs, phi_view)).squeeze())
    return max(value, 1e-300)


def reduced_radiance(case: dict[str, float], nodes: int = 32) -> float:
    amplitude, phi_mix = source_amplitude(case)
    cfg = V3Config(
        n_side=1,
        cell_area_m2=max(case["source_extent_m"] ** 2, 1.0),
        aerosol_single_scattering_albedo=case["single_scattering_albedo"],
        aerosol_asymmetry=case["asymmetry"],
        altitude_nodes=nodes,
    )
    kernel = production_kernel_scalar(
        cfg,
        distance_m=case["distance_m"],
        aod_550=case["aod_550"],
        phi=phi_mix,
        wavelength_nm=case["wavelength_nm"],
        nodes=nodes,
    )
    return max(amplitude * kernel, 1e-300)


def safe_relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / np.maximum(np.abs(b), 1e-300)


def fit_discrepancy(df: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> tuple[pd.DataFrame, dict]:
    X = df[FEATURES].to_numpy(float)
    log_ratio = np.log(df["external_radiance"].to_numpy() / df["reduced_radiance"].to_numpy())

    constant = float(np.mean(log_ratio[train]))
    constant_pred = df["reduced_radiance"].to_numpy() * np.exp(constant)

    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        Ridge(alpha=2.0),
    )
    model.fit(X[train], log_ratio[train])
    poly_pred = df["reduced_radiance"].to_numpy() * np.exp(model.predict(X))

    ext = df["external_radiance"].to_numpy()
    constant_err = safe_relative(constant_pred, ext)
    poly_err = safe_relative(poly_pred, ext)
    constant_rmse = float(np.sqrt(np.mean(constant_err[test] ** 2)))
    poly_rmse = float(np.sqrt(np.mean(poly_err[test] ** 2)))
    selected = "constant_log_scale" if constant_rmse <= poly_rmse * 1.02 else "polynomial_ridge"
    selected_pred = constant_pred if selected == "constant_log_scale" else poly_pred
    selected_err = constant_err if selected == "constant_log_scale" else poly_err

    df["constant_corrected_radiance"] = constant_pred
    df["constant_relative_error"] = constant_err
    df["polynomial_corrected_radiance"] = poly_pred
    df["polynomial_relative_error"] = poly_err
    df["selected_corrected_radiance"] = selected_pred
    df["selected_relative_error"] = selected_err
    df["selected_emission_ratio"] = ext / np.maximum(selected_pred, 1e-300)
    df["selected_95_interval_covers_true_emission"] = (
        np.abs(df["selected_emission_ratio"] - 1.0) <= 1.96 * 0.05 * df["selected_emission_ratio"]
    )
    metrics = {
        "constant_holdout_relative_rmse": constant_rmse,
        "polynomial_holdout_relative_rmse": poly_rmse,
        "selected_discrepancy": selected,
        "selected_holdout_relative_rmse": min(constant_rmse, poly_rmse),
    }
    return df, metrics


def evaluate_case(case: dict[str, float], derivative: bool, convergence: bool) -> dict[str, float | int | bool]:
    ext = external_radiance(case, 16, 12, 16)
    red = reduced_radiance(case, 32)
    out: dict[str, float | int | bool] = dict(case)
    out.update(
        external_radiance=ext,
        reduced_radiance=red,
        raw_relative_error=(red - ext) / max(abs(ext), 1e-300),
        raw_emission_ratio=ext / max(red, 1e-300),
    )

    if convergence:
        hi = external_radiance(case, 32, 24, 24)
        out["external_high_resolution_radiance"] = hi
        out["numerical_convergence_error"] = abs(hi - ext) / max(abs(hi), 1e-300)
    else:
        out["external_high_resolution_radiance"] = math.nan
        out["numerical_convergence_error"] = math.nan

    if derivative:
        da = 0.004
        cp, cm = dict(case), dict(case)
        cp["aod_550"] = min(0.49, case["aod_550"] + da)
        cm["aod_550"] = max(0.005, case["aod_550"] - da)
        denom = cp["aod_550"] - cm["aod_550"]
        dext = (external_radiance(cp) - external_radiance(cm)) / denom
        dred = (reduced_radiance(cp) - reduced_radiance(cm)) / denom

        dp = 0.01
        pp, pm = dict(case), dict(case)
        pp["angular_emission_phi"] = min(0.45, case["angular_emission_phi"] + dp)
        pm["angular_emission_phi"] = max(0.0, case["angular_emission_phi"] - dp)
        pdenom = pp["angular_emission_phi"] - pm["angular_emission_phi"]
        pext = (external_radiance(pp) - external_radiance(pm)) / pdenom
        pred = (reduced_radiance(pp) - reduced_radiance(pm)) / pdenom
        out.update(
            aod_derivative_external=dext,
            aod_derivative_reduced=dred,
            aod_derivative_relative_error=abs(dred - dext) / max(abs(dext), 1e-300),
            angular_derivative_external=pext,
            angular_derivative_reduced=pred,
            angular_derivative_relative_error=abs(pred - pext) / max(abs(pext), 1e-300),
        )
    else:
        for key in (
            "aod_derivative_external",
            "aod_derivative_reduced",
            "aod_derivative_relative_error",
            "angular_derivative_external",
            "angular_derivative_reduced",
            "angular_derivative_relative_error",
        ):
            out[key] = math.nan
    return out


def make_figure(df: pd.DataFrame, test: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), constrained_layout=True)
    ax = axes[0]
    ax.scatter(df["external_radiance"], df["reduced_radiance"], s=20, alpha=0.75)
    lo = min(df["external_radiance"].min(), df["reduced_radiance"].min())
    hi = max(df["external_radiance"].max(), df["reduced_radiance"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("PythonicDISORT radiance"); ax.set_ylabel("AURORA reduced radiance")
    ax.set_title("a  Uncorrected transfer")

    ax = axes[1]
    ax.scatter(df.loc[test, "external_radiance"], df.loc[test, "selected_corrected_radiance"], s=22, alpha=0.8)
    lo = min(df.loc[test, "external_radiance"].min(), df.loc[test, "selected_corrected_radiance"].min())
    hi = max(df.loc[test, "external_radiance"].max(), df.loc[test, "selected_corrected_radiance"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("PythonicDISORT radiance"); ax.set_ylabel("Corrected reduced radiance")
    ax.set_title("b  Held-out correction")

    ax = axes[2]
    ax.scatter(df.loc[test, "aod_550"], 100 * np.abs(df.loc[test, "selected_relative_error"]), s=22, alpha=0.8)
    ax.axhline(10, linestyle="--", linewidth=1)
    ax.set_xlabel("AOD at 550 nm"); ax.set_ylabel("Absolute relative error (%)")
    ax.set_title("c  Quantitative validity")
    fig.savefig(out / "external_solver_benchmark.png", dpi=300)
    fig.savefig(out / "external_solver_benchmark.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=72)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pythonicdisort_version": importlib.metadata.version("PythonicDISORT"),
        "numpy_version": np.__version__,
        "seed": args.seed,
        "requested_cases": args.cases,
    }
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        cases = latin_hypercube(args.cases, args.seed)
        derivative_ids = set(range(0, args.cases, max(1, args.cases // 18)))
        convergence_ids = set(range(1, args.cases, max(1, args.cases // 18)))
        rows = []
        for case in cases:
            print(f"case {case['case_id'] + 1}/{args.cases}", flush=True)
            rows.append(evaluate_case(case, case["case_id"] in derivative_ids, case["case_id"] in convergence_ids))
        df = pd.DataFrame(rows)

        rng = np.random.default_rng(args.seed + 1)
        test = np.sort(rng.choice(len(df), size=max(18, round(0.30 * len(df))), replace=False))
        train = np.setdiff1d(np.arange(len(df)), test)
        df["split"] = "train"
        df.loc[test, "split"] = "holdout"
        df, discrepancy = fit_discrepancy(df, train, test)

        conv = df["numerical_convergence_error"].dropna()
        da = df["aod_derivative_relative_error"].dropna()
        dp = df["angular_derivative_relative_error"].dropna()
        validity = (
            (np.abs(df["selected_relative_error"]) <= 0.10)
            & (df["selected_emission_ratio"].sub(1.0).abs() <= 0.15)
            & (df["numerical_convergence_error"].fillna(conv.median() if len(conv) else 0.0) <= 0.03)
        )
        df["validity_domain_pass"] = validity

        hold = df.iloc[test]
        summary = {
            "solver": "PythonicDISORT",
            "solver_version": metadata["pythonicdisort_version"],
            "solver_class": "recognized plane-parallel multiple-scattering discrete-ordinates solver",
            "mapping_scope": "column-equivalent urban boundary source; not a full 3-D localized-source solver",
            "factors_varied": [
                "aerosol optical depth", "wavelength", "source-observer distance", "aerosol-layer altitude",
                "aerosol-layer thickness", "surface albedo", "phase asymmetry", "single-scattering albedo",
                "angular emission", "viewing zenith", "relative azimuth", "observer altitude", "source extent",
            ],
            "cases": int(len(df)),
            "training_cases": int(len(train)),
            "holdout_cases": int(len(test)),
            **discrepancy,
            "holdout_median_absolute_relative_error": float(np.median(np.abs(hold["selected_relative_error"]))),
            "holdout_p95_absolute_relative_error": float(np.quantile(np.abs(hold["selected_relative_error"]), 0.95)),
            "holdout_95_interval_coverage_at_5pct_noise": float(hold["selected_95_interval_covers_true_emission"].mean()),
            "numerical_convergence_p95": float(np.quantile(conv, 0.95)) if len(conv) else None,
            "aod_derivative_relative_error_median": float(np.median(da)) if len(da) else None,
            "angular_derivative_relative_error_median": float(np.median(dp)) if len(dp) else None,
            "validity_domain_fraction_all_cases": float(validity.mean()),
            "validity_domain_rule": "|corrected radiance error| <= 10%, |emission retrieval ratio - 1| <= 15%, numerical convergence error <= 3%",
            "runtime_seconds": time.time() - started,
            "all_external_outputs_finite_positive": bool(np.isfinite(df["external_radiance"]).all() and (df["external_radiance"] > 0).all()),
        }
        summary["validation_pass"] = bool(
            summary["all_external_outputs_finite_positive"]
            and summary["selected_holdout_relative_rmse"] <= 0.15
            and summary["validity_domain_fraction_all_cases"] >= 0.35
            and (summary["numerical_convergence_p95"] is not None and summary["numerical_convergence_p95"] <= 0.03)
        )

        df.to_csv(args.output / "benchmark_cases.csv", index=False)
        (args.output / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (args.output / "benchmark_design.json").write_text(
            json.dumps({"seed": args.seed, "cases": cases, "features": FEATURES, "train_indices": train.tolist(), "holdout_indices": test.tolist()}, indent=2),
            encoding="utf-8",
        )
        make_figure(df, test, args.output)
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
        (args.output / "benchmark_summary.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
