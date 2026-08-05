"""Corrected AURORA-INVERSE model and validation utilities.

This module implements the third-generation synthetic methods study. The
realized synthetic emission field is never used to construct an inferential
prior. Molecular and aerosol vertical profiles enter both the local scattering
coefficient and both optical-depth paths. Night effects use T-1 unconstrained
coordinates with a deterministic final coordinate that enforces a zero sum.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import gamma, pi, sqrt
from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.special import expit, logit
from scipy.stats import norm

Array = NDArray[np.float64]


@dataclass(frozen=True)
class V3Config:
    n_side: int = 3
    n_nights: int = 8
    cell_spacing_m: float = 1_000.0
    cell_area_m2: float = 1_000_000.0
    wavelengths_nm: tuple[float, ...] = (450.0, 589.0, 650.0)
    rayleigh_550_m_inv: float = 1.20e-5
    molecular_scale_height_m: float = 8_000.0
    aerosol_scale_height_m: float = 1_500.0
    aerosol_angstrom: float = 1.30
    aerosol_single_scattering_albedo: float = 0.90
    aerosol_asymmetry: float = 0.65
    altitude_top_m: float = 12_000.0
    altitude_nodes: int = 32
    random_seed: int = 20260802

    @property
    def n_cells(self) -> int:
        return self.n_side**2

    @property
    def n_classes(self) -> int:
        return 3


@dataclass
class V3Dataset:
    config: V3Config
    cell_xy_m: Array
    photometer_xy_m: Array
    spectrometer_xy_m: Array
    spectra: Array
    dnb_response: Array
    photopic_response: Array
    psf: Array
    covariates: Array
    y: dict[str, Array]
    sigma: dict[str, Array]
    train_mask: dict[str, NDArray[np.bool_]]
    spatial_holdout_mask: dict[str, NDArray[np.bool_]]
    temporal_holdout_mask: dict[str, NDArray[np.bool_]]
    sensor_holdout_mask: dict[str, NDArray[np.bool_]]
    truth: dict[str, Array]
    priors: dict[str, Array]
    scenario: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class V3Fit:
    method: str
    modalities: tuple[str, ...]
    z_map: Array
    covariance: Array
    samples: Array
    metrics: dict[str, float]
    predictions: dict[str, Array]
    runtime_s: float
    success: bool
    message: str


def cell_grid(config: V3Config) -> Array:
    axis = (np.arange(config.n_side) - (config.n_side - 1) / 2.0) * config.cell_spacing_m
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(float)


def spectral_bases() -> Array:
    spectra = np.array(
        [[0.015, 0.950, 0.035], [0.180, 0.560, 0.260], [0.455, 0.365, 0.180]],
        dtype=float,
    )
    return spectra / spectra.sum(axis=1, keepdims=True)


def _horizon_norm(power: int = 8) -> float:
    integral = sqrt(pi) * gamma((power + 2) / 2) / (2 * gamma((power + 3) / 2))
    return 1.0 / (2 * pi * integral)


def angular_distribution(theta_rad: Array, phi: Array) -> Array:
    theta = np.asarray(theta_rad, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)
    lambert = np.clip(np.cos(theta), 0.0, None) / pi
    horizon = _horizon_norm(8) * np.sin(theta) ** 8
    return (1.0 - phi_arr) * lambert + phi_arr * horizon


def phase_rayleigh(cos_psi: Array) -> Array:
    return 3.0 * (1.0 + np.asarray(cos_psi) ** 2) / (16.0 * pi)


def phase_henyey_greenstein(cos_psi: Array, g: float) -> Array:
    denom = np.maximum(1.0 + g * g - 2.0 * g * np.asarray(cos_psi), 1e-12) ** 1.5
    return (1.0 - g * g) / (4.0 * pi * denom)


@lru_cache(maxsize=32)
def altitude_quadrature(n_nodes: int, top_m: float) -> tuple[Array, Array]:
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    h = 0.5 * (x + 1.0) * top_m
    weights = 0.5 * top_m * w
    return h.astype(float), weights.astype(float)


def _profile_terms(config: V3Config, aod_550: float, heights: Array) -> tuple[Array, Array, Array, Array]:
    lam = np.asarray(config.wavelengths_nm)
    beta_r0 = config.rayleigh_550_m_inv * (550.0 / lam) ** 4
    beta_a0_ext = (aod_550 / config.aerosol_scale_height_m) * (550.0 / lam) ** config.aerosol_angstrom
    beta_r_h = np.exp(-heights[:, None] / config.molecular_scale_height_m) * beta_r0[None, :]
    beta_a_ext_h = np.exp(-heights[:, None] / config.aerosol_scale_height_m) * beta_a0_ext[None, :]
    beta_a_sca_h = config.aerosol_single_scattering_albedo * beta_a_ext_h
    int_r = beta_r0[None, :] * config.molecular_scale_height_m * (
        1.0 - np.exp(-heights[:, None] / config.molecular_scale_height_m)
    )
    int_a = beta_a0_ext[None, :] * config.aerosol_scale_height_m * (
        1.0 - np.exp(-heights[:, None] / config.aerosol_scale_height_m)
    )
    return beta_r_h, beta_a_ext_h, beta_a_sca_h, int_r + int_a


def ground_kernel(
    config: V3Config,
    cell_xy_m: Array,
    sensor_xy_m: Array,
    aod_550: float,
    phi: Array,
    *,
    altitude_nodes: int | None = None,
) -> Array:
    """Corrected zenith single-scattering kernel.

    Exponential molecular and aerosol profiles are used in the local scattering
    coefficient. Optical depth is integrated separately along the oblique
    source-to-scatter path and the vertical scatter-to-sensor path.
    """
    n_nodes = altitude_nodes or config.altitude_nodes
    heights, weights = altitude_quadrature(n_nodes, config.altitude_top_m)
    delta = cell_xy_m[None, :, :] - sensor_xy_m[:, None, :]
    d = np.maximum(np.sqrt(np.sum(delta * delta, axis=-1)), 50.0)
    h = heights[:, None, None]
    r1 = np.sqrt(d[None, :, :] ** 2 + h**2)
    theta = np.arctan2(d[None, :, :], h)
    cos_psi = -h / r1

    beta_r_h, _, beta_a_sca_h, vertical_tau = _profile_terms(config, float(aod_550), heights)
    p_r = phase_rayleigh(cos_psi)[..., None]
    p_a = phase_henyey_greenstein(cos_psi, config.aerosol_asymmetry)[..., None]
    volume_sca = beta_r_h[:, None, None, :] * p_r + beta_a_sca_h[:, None, None, :] * p_a

    path_factor = r1 / np.maximum(h, 1e-9)
    total_tau = vertical_tau[:, None, None, :] * (1.0 + path_factor[..., None])
    transmission = np.exp(-total_tau)

    ang = np.stack(
        [angular_distribution(theta, np.full_like(theta, p)) for p in np.asarray(phi)],
        axis=3,
    )
    integrand = (
        config.cell_area_m2
        * ang[..., None]
        * volume_sca[:, :, :, None, :]
        * transmission[:, :, :, None, :]
        / r1[..., None, None] ** 2
    )
    return np.tensordot(weights, integrand, axes=(0, 0))


def _scalar_reference_integrand(
    h: float,
    config: V3Config,
    distance_m: float,
    aod_550: float,
    phi: float,
    wavelength_nm: float,
) -> float:
    d = max(distance_m, 50.0)
    r1 = sqrt(d * d + h * h)
    theta = np.arctan2(d, h)
    cos_psi = -h / r1
    beta_r0 = config.rayleigh_550_m_inv * (550.0 / wavelength_nm) ** 4
    beta_a0 = (aod_550 / config.aerosol_scale_height_m) * (550.0 / wavelength_nm) ** config.aerosol_angstrom
    beta_r = beta_r0 * np.exp(-h / config.molecular_scale_height_m)
    beta_a_sca = config.aerosol_single_scattering_albedo * beta_a0 * np.exp(-h / config.aerosol_scale_height_m)
    int_r = beta_r0 * config.molecular_scale_height_m * (1.0 - np.exp(-h / config.molecular_scale_height_m))
    int_a = beta_a0 * config.aerosol_scale_height_m * (1.0 - np.exp(-h / config.aerosol_scale_height_m))
    tau = (int_r + int_a) * (1.0 + r1 / max(h, 1e-12))
    volume = beta_r * phase_rayleigh(np.array(cos_psi)) + beta_a_sca * phase_henyey_greenstein(np.array(cos_psi), config.aerosol_asymmetry)
    ang = angular_distribution(np.array(theta), np.array(phi))
    return float(config.cell_area_m2 * ang * volume * np.exp(-tau) / (r1 * r1))


def reference_kernel_scalar(
    config: V3Config,
    distance_m: float,
    aod_550: float,
    phi: float,
    wavelength_nm: float,
) -> float:
    val, _ = quad(
        _scalar_reference_integrand,
        1e-3,
        config.altitude_top_m,
        args=(config, distance_m, aod_550, phi, wavelength_nm),
        epsabs=1e-14,
        epsrel=2e-9,
        limit=300,
    )
    return float(val)


def production_kernel_scalar(
    config: V3Config,
    distance_m: float,
    aod_550: float,
    phi: float,
    wavelength_nm: float,
    nodes: int = 48,
) -> float:
    c = np.array([[distance_m, 0.0]])
    s = np.array([[0.0, 0.0]])
    k = ground_kernel(config, c, s, aod_550, np.array([phi, phi, phi]), altitude_nodes=nodes)
    b = int(np.argmin(np.abs(np.asarray(config.wavelengths_nm) - wavelength_nm)))
    return float(k[0, 0, 0, b])


def satellite_psf(cell_xy_m: Array, sigma_m: float = 550.0) -> Array:
    delta = cell_xy_m[:, None, :] - cell_xy_m[None, :, :]
    d2 = np.sum(delta * delta, axis=-1)
    matrix = np.exp(-0.5 * d2 / sigma_m**2)
    return matrix / matrix.sum(axis=1, keepdims=True)


def sensor_locations(config: V3Config) -> tuple[Array, Array]:
    r = 2.8 * config.cell_spacing_m
    photo = np.array(
        [[-r, 0.0], [r, 0.0], [0.0, -r], [0.0, r], [-0.72 * r, 0.72 * r], [0.72 * r, -0.72 * r]],
        dtype=float,
    )
    return photo, photo[[0, 1, 2, 3]]


def external_covariates(config: V3Config, xy: Array, seed: int) -> Array:
    """Construct covariates without access to the realized emission field."""
    rng = np.random.default_rng(seed)
    x = xy[:, 0] / max(config.cell_spacing_m, 1.0)
    y = xy[:, 1] / max(config.cell_spacing_m, 1.0)
    population = np.exp(-((x - 0.2) ** 2 + (y + 0.15) ** 2) / 3.2)
    road = np.exp(-(y + 0.55 * x - 0.15) ** 2 / 0.70)
    commercial = np.exp(-((x - 0.7) ** 2 + (y + 0.5) ** 2) / 1.8)
    independent = rng.normal(0.0, 0.18, len(x))
    cov = np.column_stack([population, road, commercial, x, y, independent])
    cov[:, :3] = (cov[:, :3] - cov[:, :3].mean(axis=0)) / np.maximum(cov[:, :3].std(axis=0), 1e-8)
    if len(x) > 1:
        cov[:, 3:5] = (cov[:, 3:5] - cov[:, 3:5].mean(axis=0)) / np.maximum(cov[:, 3:5].std(axis=0), 1e-8)
    # Sparse urban covariates can have vanishing empirical variance when the
    # mesh extends far beyond a compact feature. Unbounded z-scores then create
    # scale-dependent prior and truth explosions (the former 20x20 generator
    # reached >11 standard deviations). Winsorization is part of the declared
    # covariate preprocessing and prevents extrapolation beyond the support in
    # which the prior coefficients were calibrated.
    cov[:, :5] = np.clip(cov[:, :5], -3.0, 3.0)
    return cov


def prior_mean_from_covariates(covariates: Array, scenario: str = "moderate") -> Array:
    pop, road, commercial, x, y, independent = covariates.T
    log_total = np.log(2.25e-4) + 0.33 * pop + 0.22 * road + 0.14 * commercial + 0.10 * independent
    logits = np.column_stack(
        [0.20 - 0.18 * x + 0.08 * road, 0.55 + 0.15 * y + 0.05 * commercial, -0.10 + 0.20 * x - 0.08 * y]
    )
    if scenario == "prior_misspecification":
        log_total = log_total + 0.45 - 0.45 * pop
        logits = logits[:, [1, 2, 0]]
    frac = np.exp(logits - logits.max(axis=1, keepdims=True))
    frac /= frac.sum(axis=1, keepdims=True)
    return log_total[:, None] + np.log(frac)


def _truth_from_covariates(config: V3Config, covariates: Array, rng: np.random.Generator, scenario: str) -> Array:
    pop, road, commercial, x, y, independent = covariates.T
    latent = rng.normal(0.0, 0.25, config.n_cells)
    if config.n_side > 1:
        grid = latent.reshape(config.n_side, config.n_side)
        latent = (0.55 * grid + 0.1125 * (np.roll(grid, 1, 0) + np.roll(grid, -1, 0) + np.roll(grid, 1, 1) + np.roll(grid, -1, 1))).ravel()
    log_total = np.log(2.10e-4) + 0.48 * pop + 0.30 * road + 0.20 * commercial + 0.22 * latent
    logits = np.column_stack(
        [0.30 - 0.24 * x + 0.11 * road, 0.62 + 0.20 * y + 0.09 * commercial, -0.14 + 0.25 * x - 0.12 * y]
    )
    logits += rng.normal(0.0, 0.16, logits.shape)
    if scenario == "out_of_distribution":
        log_total += 0.35 * np.sign(x + 0.1) - 0.30 * pop
        logits = logits[:, [2, 0, 1]]
    frac = np.exp(logits - logits.max(axis=1, keepdims=True))
    frac /= frac.sum(axis=1, keepdims=True)
    return np.exp(log_total[:, None]) * frac


def _night_process(rng: np.random.Generator, n: int, sd: float = 0.065, rho: float = 0.65) -> Array:
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal(0.0, sd * sqrt(1.0 - rho**2))
    return x - x.mean()


def forward_model(
    config: V3Config,
    cell_xy_m: Array,
    photometer_xy_m: Array,
    spectrometer_xy_m: Array,
    spectra: Array,
    dnb_response: Array,
    photopic_response: Array,
    psf: Array,
    base_w: Array,
    night_log_scale: Array,
    aod: Array,
    phi: Array,
    calibration: Array | None = None,
) -> dict[str, Array]:
    n_t = len(night_log_scale)
    calibration = np.ones(3) if calibration is None else np.asarray(calibration)
    emissions = base_w[None, :, :] * np.exp(night_log_scale[:, None, None])
    lam = np.asarray(config.wavelengths_nm)
    spectral_dnb = spectra * dnb_response[None, :]
    vza = np.deg2rad(5.0)
    ang_sat = angular_distribution(np.full(config.n_classes, vza), phi)
    sat = np.empty((n_t, config.n_cells))
    photo = np.empty((n_t, len(photometer_xy_m)))
    spec = np.empty((n_t, len(spectrometer_xy_m), len(lam)))
    for t in range(n_t):
        beta_r0 = config.rayleigh_550_m_inv * (550.0 / lam) ** 4
        beta_a0 = (aod[t] / config.aerosol_scale_height_m) * (550.0 / lam) ** config.aerosol_angstrom
        vertical_tau = beta_r0 * config.molecular_scale_height_m + beta_a0 * config.aerosol_scale_height_m
        toa_factor = np.sum(spectral_dnb * np.exp(-vertical_tau)[None, :], axis=1) * ang_sat
        sat[t] = (psf @ np.sum(emissions[t] * toa_factor[None, :], axis=1)) * 1.0e5
        kernel_p = ground_kernel(config, cell_xy_m, photometer_xy_m, float(aod[t]), phi)
        spectral_rad_p = np.einsum("sikb,ik,kb->sb", kernel_p, emissions[t], spectra, optimize=True)
        photo[t] = spectral_rad_p @ photopic_response * 1.0e5
        kernel_s = ground_kernel(config, cell_xy_m, spectrometer_xy_m, float(aod[t]), phi)
        spec[t] = np.einsum("sikb,ik,kb->sb", kernel_s, emissions[t], spectra, optimize=True) * 1.0e5
    natural_photo = 2.0e-4
    natural_spec = np.array([0.8e-4, 1.0e-4, 0.7e-4])
    return {
        "satellite": calibration[0] * sat,
        "photometry": calibration[1] * (photo + natural_photo),
        "spectral": calibration[2] * (spec + natural_spec[None, None, :]),
    }


def generate_scene(
    config: V3Config | None = None,
    *,
    scenario: str = "moderate",
    seed: int | None = None,
    prior_mode: str = "external",
) -> V3Dataset:
    config = config or V3Config()
    seed = config.random_seed if seed is None else seed
    rng = np.random.default_rng(seed)
    xy = cell_grid(config)
    photo_xy, spec_xy = sensor_locations(config)
    spectra = spectral_bases()
    dnb = np.array([0.025, 0.70, 0.92])
    photopic = np.array([0.08, 0.76, 0.42])
    psf = satellite_psf(xy)
    cov = external_covariates(config, xy, seed + 771_103)
    w = _truth_from_covariates(config, cov, rng, scenario)
    u = _night_process(rng, config.n_nights)
    time = np.linspace(0.0, 1.0, config.n_nights)
    met_proxy = np.clip(0.105 + 0.030 * np.sin(2.4 * pi * time + 0.4) + rng.normal(0, 0.008, config.n_nights), 0.035, 0.30)
    aod = np.clip(met_proxy * np.exp(rng.normal(0.0, 0.16, config.n_nights)), 0.025, 0.34)
    phi = np.array([0.075, 0.175, 0.285]) + rng.normal(0.0, 0.012, 3)
    phi = np.clip(phi, 0.025, 0.42)
    calibration = np.exp(rng.normal(0.0, 0.028, 3))
    if scenario == "source_atmosphere_degeneracy":
        aod = np.clip(0.20 + 0.045 * np.sin(2 * pi * time), 0.10, 0.32)
        met_proxy = np.clip(aod * 0.72, 0.03, 0.35)
    noiseless = forward_model(config, xy, photo_xy, spec_xy, spectra, dnb, photopic, psf, w, u, aod, phi, calibration)
    rel_noise = {"satellite": 0.050, "photometry": 0.038, "spectral": 0.032}
    floors = {"satellite": 0.002, "photometry": 2.0e-5, "spectral": 1.5e-5}
    y: dict[str, Array] = {}
    sigma: dict[str, Array] = {}
    train: dict[str, NDArray[np.bool_]] = {}
    spatial: dict[str, NDArray[np.bool_]] = {}
    temporal: dict[str, NDArray[np.bool_]] = {}
    sensor: dict[str, NDArray[np.bool_]] = {}
    n_hold_nights = max(1, config.n_nights // 5)
    for modality, mu in noiseless.items():
        sig = floors[modality] + rel_noise[modality] * np.maximum(mu, np.median(mu) * 0.12)
        obs = mu + rng.normal(0.0, sig)
        if scenario == "structural_discrepancy":
            if modality == "satellite":
                obs *= 1.0 + 0.055 * np.sin(2 * pi * time)[:, None]
            if modality == "spectral":
                obs[..., 0] += 0.10 * np.maximum(mu[..., 0], floors[modality])
        valid = rng.random(mu.shape) > 0.04
        spatial_mask = np.zeros(mu.shape, dtype=bool)
        temporal_mask = np.zeros(mu.shape, dtype=bool)
        sensor_mask = np.zeros(mu.shape, dtype=bool)
        if modality == "satellite":
            # A complete one-cell spatial block is withheld across all nights.
            spatial_mask[:, -1] = True
        else:
            # Ground observations from the final nights are withheld while
            # satellite observations remain available on those nights.
            temporal_mask[-n_hold_nights:] = True
            if modality == "photometry":
                sensor_mask[:, -1] = True
            elif modality == "spectral":
                sensor_mask[:, -1, :] = True
        spatial_mask &= valid
        temporal_mask &= valid & ~spatial_mask
        sensor_mask &= valid & ~spatial_mask & ~temporal_mask
        train_mask = valid & ~spatial_mask & ~temporal_mask & ~sensor_mask
        y[modality] = obs
        sigma[modality] = sig
        train[modality] = train_mask
        spatial[modality] = spatial_mask
        temporal[modality] = temporal_mask
        sensor[modality] = sensor_mask
    if prior_mode == "weak":
        prior_logw = np.full_like(w, np.log(2.0e-4 / config.n_classes))
    else:
        prior_scenario = "prior_misspecification" if scenario == "prior_misspecification" else "moderate"
        prior_logw = prior_mean_from_covariates(cov, prior_scenario)
    return V3Dataset(
        config=config,
        cell_xy_m=xy,
        photometer_xy_m=photo_xy,
        spectrometer_xy_m=spec_xy,
        spectra=spectra,
        dnb_response=dnb,
        photopic_response=photopic,
        psf=psf,
        covariates=cov,
        y=y,
        sigma=sigma,
        train_mask=train,
        spatial_holdout_mask=spatial,
        temporal_holdout_mask=temporal,
        sensor_holdout_mask=sensor,
        truth={"base_w": w, "night_log_scale": u, "aod": aod, "phi": phi, "calibration": calibration},
        priors={"logw": prior_logw, "aod": met_proxy, "phi": np.array([0.10, 0.19, 0.26])},
        scenario=scenario,
        metadata={
            "prior_provenance": "external coordinate and land-use surrogate generated independently of realized truth",
            "holdout_design": "blocked nights plus complete held-out ground sensors",
            "observations": "synthetic",
        },
    )


class ParameterLayout:
    def __init__(self, data: V3Dataset, infer_calibration: bool = True):
        self.data = data
        self.infer_calibration = infer_calibration
        n_w = data.config.n_cells * data.config.n_classes
        n_t = data.config.n_nights
        cursor = 0
        self.logw = slice(cursor, cursor + n_w); cursor += n_w
        self.u_free = slice(cursor, cursor + n_t - 1); cursor += n_t - 1
        self.logaod = slice(cursor, cursor + n_t); cursor += n_t
        self.phi_z = slice(cursor, cursor + data.config.n_classes); cursor += data.config.n_classes
        self.logcal = slice(cursor, cursor + 3) if infer_calibration else None
        cursor += 3 if infer_calibration else 0
        self.size = cursor

    def initial(self, rng: np.random.Generator | None = None, jitter: float = 0.0) -> Array:
        rng = rng or np.random.default_rng(0)
        d = self.data
        parts = [d.priors["logw"].ravel(), np.zeros(d.config.n_nights - 1), np.log(d.priors["aod"])]
        scaled = np.clip((d.priors["phi"] - 0.01) / 0.44, 1e-5, 1 - 1e-5)
        parts.append(logit(scaled))
        if self.infer_calibration:
            parts.append(np.zeros(3))
        z = np.concatenate(parts)
        if jitter:
            z += rng.normal(0.0, jitter, z.shape)
        return z

    def decode(self, z: Array) -> dict[str, Array]:
        d = self.data
        logw = z[self.logw].reshape(d.config.n_cells, d.config.n_classes)
        u_free = z[self.u_free]
        u = np.concatenate([u_free, np.array([-u_free.sum()])])
        phi = 0.01 + 0.44 * expit(z[self.phi_z])
        cal = np.exp(z[self.logcal]) if self.logcal is not None else np.ones(3)
        return {
            "logw": logw,
            "base_w": np.exp(logw),
            "night_log_scale": u,
            "aod": np.exp(z[self.logaod]),
            "phi": phi,
            "calibration": cal,
        }


def neighbor_pairs(config: V3Config) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r in range(config.n_side):
        for c in range(config.n_side):
            i = r * config.n_side + c
            if r + 1 < config.n_side:
                pairs.append((i, (r + 1) * config.n_side + c))
            if c + 1 < config.n_side:
                pairs.append((i, r * config.n_side + c + 1))
    return pairs


def prior_residuals(data: V3Dataset, layout: ParameterLayout, z: Array, method: str) -> list[Array]:
    p = layout.decode(z)
    chunks: list[Array] = []
    if method == "weak":
        chunks.append((p["logw"] - data.priors["logw"]).ravel() / 1.45)
    else:
        chunks.append((p["logw"] - data.priors["logw"]).ravel() / 0.92)
        if data.config.n_cells > 1:
            spatial = np.array([p["logw"][i] - p["logw"][j] for i, j in neighbor_pairs(data.config)]).ravel()
            chunks.append(spatial / 0.68)
    u = p["night_log_scale"]
    chunks.append(u / 0.18)
    if len(u) > 1:
        chunks.append(np.diff(u) / 0.12)
    loga = np.log(p["aod"])
    chunks.append((loga - np.log(data.priors["aod"])) / 0.28)
    if len(loga) > 1:
        chunks.append(np.diff(loga) / 0.24)
    scaled = np.clip((data.priors["phi"] - 0.01) / 0.44, 1e-5, 1 - 1e-5)
    chunks.append((z[layout.phi_z] - logit(scaled)) / 0.92)
    if layout.logcal is not None:
        chunks.append(z[layout.logcal] / 0.050)
    return chunks


def predict_from_z(data: V3Dataset, layout: ParameterLayout, z: Array) -> dict[str, Array]:
    p = layout.decode(z)
    return forward_model(
        data.config,
        data.cell_xy_m,
        data.photometer_xy_m,
        data.spectrometer_xy_m,
        data.spectra,
        data.dnb_response,
        data.photopic_response,
        data.psf,
        p["base_w"],
        p["night_log_scale"],
        p["aod"],
        p["phi"],
        p["calibration"],
    )


def fit_gauss_newton_gaussian(
    data: V3Dataset,
    *,
    modalities: Iterable[str] = ("satellite", "photometry", "spectral"),
    method: str = "external",
    seed: int = 41,
    max_nfev: int = 70,
    sample_count: int = 500,
    initialization_jitter: float = 0.0,
) -> V3Fit:
    modalities = tuple(modalities)
    layout = ParameterLayout(data, infer_calibration=True)
    rng = np.random.default_rng(seed)
    z0 = layout.initial(rng, initialization_jitter)

    def residual(z: Array) -> Array:
        pred = predict_from_z(data, layout, z)
        chunks: list[Array] = []
        for modality in modalities:
            mask = data.train_mask[modality]
            chunks.append(((pred[modality] - data.y[modality]) / data.sigma[modality])[mask])
        chunks.extend(prior_residuals(data, layout, z, method))
        return np.concatenate(chunks)

    lower = np.full(layout.size, -np.inf)
    upper = np.full(layout.size, np.inf)
    lower[layout.logw] = np.log(1.0e-7)
    upper[layout.logw] = np.log(3.0e-3)
    lower[layout.u_free] = -0.55
    upper[layout.u_free] = 0.55
    lower[layout.logaod] = np.log(0.012)
    upper[layout.logaod] = np.log(0.50)
    lower[layout.phi_z] = -7.0
    upper[layout.phi_z] = 7.0
    if layout.logcal is not None:
        lower[layout.logcal] = -0.25
        upper[layout.logcal] = 0.25
    z0 = np.clip(z0, lower + 1e-8, upper - 1e-8)

    start = perf_counter()
    opt = least_squares(
        residual,
        z0,
        method="trf",
        jac="2-point",
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=max_nfev,
        ftol=4e-7,
        xtol=4e-7,
        gtol=4e-7,
    )
    runtime = perf_counter() - start
    residual_at_optimum = opt.fun
    gradient = opt.jac.T @ residual_at_optimum
    projected_gradient = gradient.copy()
    active_lower = np.isclose(opt.x, lower, rtol=0.0, atol=1e-7)
    active_upper = np.isclose(opt.x, upper, rtol=0.0, atol=1e-7)
    projected_gradient[active_lower & (gradient > 0.0)] = 0.0
    projected_gradient[active_upper & (gradient < 0.0)] = 0.0

    precision = opt.jac.T @ opt.jac
    precision = (precision + precision.T) / 2.0
    raw_evals, evecs = np.linalg.eigh(precision)
    floor = max(np.max(raw_evals) * 1e-10, 1e-10)
    regularized_evals = np.maximum(raw_evals, floor)
    covariance = (evecs * (1.0 / regularized_evals)) @ evecs.T
    samples = rng.multivariate_normal(opt.x, covariance, size=sample_count, method="eigh")
    pred = predict_from_z(data, layout, opt.x)
    metrics = evaluate_fit(data, layout, opt.x, covariance, samples, pred, modalities, method)
    metrics.update({
        "objective_cost": float(opt.cost),
        "n_function_evaluations": float(opt.nfev),
        "gradient_l2_norm": float(np.linalg.norm(gradient)),
        "gradient_infinity_norm": float(np.linalg.norm(gradient, ord=np.inf)),
        "projected_gradient_infinity_norm": float(np.linalg.norm(projected_gradient, ord=np.inf)),
        "active_bound_count": float(np.sum(active_lower | active_upper)),
        "gauss_newton_minimum_eigenvalue_raw": float(np.min(raw_evals)),
        "gauss_newton_maximum_eigenvalue_raw": float(np.max(raw_evals)),
        "gauss_newton_condition_number_regularized": float(np.max(regularized_evals) / np.min(regularized_evals)),
        "gauss_newton_eigenvalue_floor": float(floor),
        "optimizer_status": float(opt.status),
    })
    return V3Fit(method, modalities, opt.x, covariance, samples, metrics, pred, runtime, bool(opt.success), str(opt.message))


def _prior_logw_marginal_sd(data: V3Dataset, method: str) -> Array:
    n = data.config.n_cells * data.config.n_classes
    if method == "weak":
        return np.full(n, 1.45)
    q = np.eye(n) / 0.92**2
    for i, j in neighbor_pairs(data.config):
        for k in range(data.config.n_classes):
            a = i * data.config.n_classes + k
            b = j * data.config.n_classes + k
            q[a, a] += 1.0 / 0.68**2
            q[b, b] += 1.0 / 0.68**2
            q[a, b] -= 1.0 / 0.68**2
            q[b, a] -= 1.0 / 0.68**2
    cov = np.linalg.inv(q)
    return np.sqrt(np.diag(cov))


def evaluate_fit(
    data: V3Dataset,
    layout: ParameterLayout,
    z_map: Array,
    covariance: Array,
    samples: Array,
    predictions: dict[str, Array],
    modalities: tuple[str, ...],
    method: str,
) -> dict[str, float]:
    p = layout.decode(z_map)
    truth = data.truth
    w_true = truth["base_w"].ravel()
    w_hat = p["base_w"].ravel()
    decoded_w = np.stack([layout.decode(s)["base_w"].ravel() for s in samples])
    lo, hi = np.quantile(decoded_w, [0.05, 0.95], axis=0)
    total_true = truth["base_w"].sum(axis=1)
    total_hat = p["base_w"].sum(axis=1)
    total_samples = np.stack([layout.decode(s)["base_w"].sum(axis=1) for s in samples])
    tlo, thi = np.quantile(total_samples, [0.05, 0.95], axis=0)
    spatial_sq: list[Array] = []
    temporal_sq: list[Array] = []
    sensor_sq: list[Array] = []
    train_sq: list[Array] = []
    for modality in modalities:
        xm = data.spatial_holdout_mask[modality]
        tm = data.temporal_holdout_mask[modality]
        sm = data.sensor_holdout_mask[modality]
        tr = data.train_mask[modality]
        std = (predictions[modality] - data.y[modality]) / data.sigma[modality]
        if np.any(xm): spatial_sq.append(std[xm] ** 2)
        if np.any(tm): temporal_sq.append(std[tm] ** 2)
        if np.any(sm): sensor_sq.append(std[sm] ** 2)
        if np.any(tr): train_sq.append(std[tr] ** 2)
    post_sd = np.sqrt(np.maximum(np.diag(covariance)[layout.logw], 0.0))
    prior_sd = _prior_logw_marginal_sd(data, method)
    total_emission_samples = np.array([layout.decode(s)["base_w"].sum() for s in samples])
    mean_aod_samples = np.array([layout.decode(s)["aod"].mean() for s in samples])
    corr = float(np.corrcoef(total_emission_samples, mean_aod_samples)[0, 1])
    rel = np.abs(w_hat - w_true) / np.maximum(w_true, 1e-12)
    return {
        "emission_relative_mae": float(np.mean(rel)),
        "total_cell_relative_mae": float(np.mean(np.abs(total_hat - total_true) / np.maximum(total_true, 1e-12))),
        "total_spatial_correlation": float(np.corrcoef(total_hat, total_true)[0, 1]) if len(total_true) > 1 else 1.0,
        "aod_rmse": float(np.sqrt(np.mean((p["aod"] - truth["aod"]) ** 2))),
        "angular_mae": float(np.mean(np.abs(p["phi"] - truth["phi"]))),
        "coverage_90_coefficients": float(np.mean((w_true >= lo) & (w_true <= hi))),
        "coverage_90_cell_totals": float(np.mean((total_true >= tlo) & (total_true <= thi))),
        "relative_interval_width_90": float(np.mean((hi - lo) / np.maximum(w_true, 1e-12))),
        "posterior_contraction": float(1.0 - np.mean(post_sd / prior_sd)),
        "spatial_holdout_rmse": float(np.sqrt(np.mean(np.concatenate(spatial_sq)))) if spatial_sq else float("nan"),
        "temporal_holdout_rmse": float(np.sqrt(np.mean(np.concatenate(temporal_sq)))) if temporal_sq else float("nan"),
        "sensor_holdout_rmse": float(np.sqrt(np.mean(np.concatenate(sensor_sq)))) if sensor_sq else float("nan"),
        "training_standardized_rmse": float(np.sqrt(np.mean(np.concatenate(train_sq)))) if train_sq else float("nan"),
        "source_aod_correlation": corr,
    }


def standardized_prediction_jacobian(data: V3Dataset, fit: V3Fit, relative_step: float = 2e-5) -> Array:
    layout = ParameterLayout(data, True)
    z = fit.z_map
    y0_chunks = []
    pred0 = predict_from_z(data, layout, z)
    for modality in fit.modalities:
        m = data.train_mask[modality]
        y0_chunks.append((pred0[modality] / data.sigma[modality])[m])
    y0 = np.concatenate(y0_chunks)
    jac = np.empty((len(y0), len(z)))
    for j in range(len(z)):
        step = relative_step * max(1.0, abs(float(z[j])))
        zp = z.copy(); zm = z.copy(); zp[j] += step; zm[j] -= step
        pp = predict_from_z(data, layout, zp); pm = predict_from_z(data, layout, zm)
        cp = []; cm = []
        for modality in fit.modalities:
            m = data.train_mask[modality]
            cp.append((pp[modality] / data.sigma[modality])[m])
            cm.append((pm[modality] / data.sigma[modality])[m])
        jac[:, j] = (np.concatenate(cp) - np.concatenate(cm)) / (2 * step)
    return jac


def identifiability(data: V3Dataset, fit: V3Fit) -> dict[str, Array | float | int]:
    jac = standardized_prediction_jacobian(data, fit)
    singular = np.linalg.svd(jac, compute_uv=False)
    threshold = singular[0] * 1e-3
    return {
        "singular_values": singular,
        "effective_rank": int(np.sum(singular > threshold)),
        "parameter_count": int(jac.shape[1]),
        "condition_number": float(singular[0] / max(singular[-1], 1e-15)),
    }


def active_measurement_scores(data: V3Dataset, fit: V3Fit, grid_points: int = 15) -> dict[str, Array]:
    layout = ParameterLayout(data, True)
    p = layout.decode(fit.z_map)
    axis = np.linspace(-4_500.0, 4_500.0, grid_points)
    xx, yy = np.meshgrid(axis, axis)
    candidates = np.column_stack([xx.ravel(), yy.ravel()])
    idx = np.arange(layout.logw.start, layout.logw.stop)
    cov = fit.covariance[np.ix_(idx, idx)]
    scores = np.empty(len(candidates))
    predicted = np.empty(len(candidates))
    mean_aod = float(np.mean(p["aod"]))
    for q, point in enumerate(candidates):
        kernel = ground_kernel(data.config, data.cell_xy_m, point[None, :], mean_aod, p["phi"])
        conv = np.einsum("sikb,kb,b->ik", kernel, data.spectra, data.photopic_response, optimize=True)
        value = float(np.sum(p["base_w"] * conv) * 1e5 + 2e-4)
        h = (p["base_w"] * conv * 1e5).ravel()
        noise = 2e-5 + 0.038 * max(value, 1e-5)
        var = float(h @ cov @ h)
        scores[q] = 0.5 * np.log1p(max(var, 0.0) / noise**2)
        predicted[q] = value
    return {"xy_m": candidates, "score_nat": scores, "predicted_radiance": predicted}


def physical_benchmark(config: V3Config | None = None) -> list[dict[str, float]]:
    config = config or V3Config(n_side=1, n_nights=2, altitude_nodes=48)
    rows: list[dict[str, float]] = []
    for d in (250.0, 750.0, 1_500.0, 3_000.0, 6_000.0):
        for aod in (0.03, 0.10, 0.20, 0.30):
            for phi in (0.06, 0.20, 0.36):
                for wavelength in config.wavelengths_nm:
                    ref = reference_kernel_scalar(config, d, aod, phi, wavelength)
                    prod = production_kernel_scalar(config, d, aod, phi, wavelength, nodes=48)
                    rel = abs(prod - ref) / max(abs(ref), 1e-20)
                    rows.append({
                        "distance_m": d,
                        "aod_550": aod,
                        "phi": phi,
                        "wavelength_nm": wavelength,
                        "reference": ref,
                        "production": prod,
                        "relative_error": rel,
                    })
    return rows


def _precision_from_residual_matrix(a: Array) -> Array:
    return a.T @ a


def _sample_gaussian_from_factors(mean: Array, factors: Sequence[tuple[Array, float]], rng: np.random.Generator) -> Array:
    n = len(mean)
    rows = []
    for mat, sd in factors:
        rows.append(mat / sd)
    a = np.vstack(rows)
    q = _precision_from_residual_matrix(a)
    l = np.linalg.cholesky(q)
    return mean + np.linalg.solve(l.T, rng.normal(size=n))


def generate_prior_predictive_scene(config: V3Config, seed: int) -> V3Dataset:
    """Draw a scene exactly from the priors used by the reduced SBC model."""
    rng = np.random.default_rng(seed)
    xy = cell_grid(config)
    photo_xy, spec_xy = sensor_locations(config)
    spectra = spectral_bases(); dnb = np.array([0.025, 0.70, 0.92]); photopic = np.array([0.08, 0.76, 0.42]); psf = satellite_psf(xy)
    cov = external_covariates(config, xy, seed + 991_003)
    mu = prior_mean_from_covariates(cov)
    n_w = config.n_cells * config.n_classes
    i_mat = np.eye(n_w)
    edge_rows = []
    for i, j in neighbor_pairs(config):
        for k in range(config.n_classes):
            row = np.zeros(n_w); row[i * config.n_classes + k] = 1.0; row[j * config.n_classes + k] = -1.0
            edge_rows.append(row)
    factors_w: list[tuple[Array, float]] = [(i_mat, 0.92)]
    if edge_rows: factors_w.append((np.vstack(edge_rows), 0.68))
    logw = _sample_gaussian_from_factors(mu.ravel(), factors_w, rng).reshape(config.n_cells, config.n_classes)
    n_free = config.n_nights - 1
    c = np.vstack([np.eye(n_free), -np.ones((1, n_free))])
    dmat = np.diff(np.eye(config.n_nights), axis=0) @ c
    u_free = _sample_gaussian_from_factors(np.zeros(n_free), [(c, 0.18), (dmat, 0.12)], rng)
    u = c @ u_free
    time = np.linspace(0, 1, config.n_nights)
    proxy = np.clip(0.11 + 0.025 * np.sin(2.2 * pi * time + rng.uniform(-0.5, 0.5)), 0.04, 0.28)
    i_t = np.eye(config.n_nights); d_t = np.diff(i_t, axis=0)
    loga = _sample_gaussian_from_factors(np.log(proxy), [(i_t, 0.28), (d_t, 0.24)], rng)
    phi_prior = np.array([0.10, 0.19, 0.26]); scaled = (phi_prior - 0.01) / 0.44
    phi_z = logit(scaled) + rng.normal(0, 0.92, 3)
    phi = 0.01 + 0.44 * expit(phi_z)
    logcal = rng.normal(0, 0.05, 3); cal = np.exp(logcal)
    noiseless = forward_model(config, xy, photo_xy, spec_xy, spectra, dnb, photopic, psf, np.exp(logw), u, np.exp(loga), phi, cal)
    rel_noise = {"satellite": 0.050, "photometry": 0.038, "spectral": 0.032}
    floors = {"satellite": 0.002, "photometry": 2.0e-5, "spectral": 1.5e-5}
    y = {}; sigma = {}; train = {}; spatial = {}; temporal = {}; sensor = {}
    for modality, mu_obs in noiseless.items():
        sig = floors[modality] + rel_noise[modality] * np.maximum(mu_obs, np.median(mu_obs) * 0.12)
        y[modality] = mu_obs + rng.normal(0, sig)
        sigma[modality] = sig
        train[modality] = np.ones(mu_obs.shape, dtype=bool)
        spatial[modality] = np.zeros(mu_obs.shape, dtype=bool)
        temporal[modality] = np.zeros(mu_obs.shape, dtype=bool)
        sensor[modality] = np.zeros(mu_obs.shape, dtype=bool)
    return V3Dataset(
        config, xy, photo_xy, spec_xy, spectra, dnb, photopic, psf, cov, y, sigma, train, spatial, temporal, sensor,
        {"base_w": np.exp(logw), "night_log_scale": u, "aod": np.exp(loga), "phi": phi, "calibration": cal},
        {"logw": mu, "aod": proxy, "phi": phi_prior}, "matched_prior_predictive",
        {"prior_provenance": "exact prior predictive draw for SBC"},
    )


def sbc_quantities(data: V3Dataset, layout: ParameterLayout, z: Array) -> Array:
    p = layout.decode(z)
    w = p["base_w"]
    class_totals = w.sum(axis=0)
    return np.array([
        np.log(w.sum()),
        np.log(class_totals[0] / class_totals[1]),
        np.log(class_totals[2] / class_totals[1]),
        np.log(p["aod"]).mean(),
        np.std(p["night_log_scale"]),
        p["phi"][1],
        np.log(p["calibration"][0]),
    ])


def run_sbc_replicate(seed: int, draws: int = 160) -> dict[str, Array | bool | float]:
    config = V3Config(n_side=1, n_nights=4, random_seed=seed)
    data = generate_prior_predictive_scene(config, seed)
    fit = fit_gauss_newton_gaussian(data, modalities=("satellite", "photometry", "spectral"), method="external", seed=seed + 17, max_nfev=34, sample_count=draws)
    layout = ParameterLayout(data, True)
    true_z = layout.initial()
    true_z[layout.logw] = np.log(data.truth["base_w"]).ravel()
    true_z[layout.u_free] = data.truth["night_log_scale"][:-1]
    true_z[layout.logaod] = np.log(data.truth["aod"])
    scaled = (data.truth["phi"] - 0.01) / 0.44
    true_z[layout.phi_z] = logit(np.clip(scaled, 1e-8, 1 - 1e-8))
    true_z[layout.logcal] = np.log(data.truth["calibration"])
    truth_q = sbc_quantities(data, layout, true_z)
    sample_q = np.stack([sbc_quantities(data, layout, z) for z in fit.samples])
    ranks = np.sum(sample_q < truth_q[None, :], axis=0)
    coverage = []
    for level in (0.50, 0.80, 0.90, 0.95):
        alpha = (1 - level) / 2
        lo, hi = np.quantile(sample_q, [alpha, 1 - alpha], axis=0)
        coverage.append((truth_q >= lo) & (truth_q <= hi))
    return {
        "ranks": ranks.astype(int),
        "coverage": np.asarray(coverage, dtype=int),
        "success": fit.success,
        "runtime_s": fit.runtime_s,
    }
