#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import linalg as spla  # type: ignore

    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    spla = None
    HAVE_SCIPY = False


K_H_RE = re.compile(r"^K(\d+)_H\.npy$")
K_S_RE = re.compile(r"^K(\d+)_S\.npy$")


class CanonicalOrthogonalizationError(ValueError):
    def __init__(self, message: str, *, meta: Optional[Mapping[str, float]] = None) -> None:
        super().__init__(message)
        self.meta = dict(meta or {})


@dataclass(frozen=True)
class RunConfig:
    n_draws: int
    n_adv: int
    pd_tol: float
    seed: int
    abs_error: bool
    h0_factor: float
    s0_factor: float
    h0_override: Optional[float]
    s0_override: Optional[float]
    adv_h_align: str  # "random" or "pattern"
    diag_scale: bool
    eig_cut_rel: Optional[float]
    eig_cut_abs: Optional[float]
    min_rank: int
    report_raw_indef: bool
    metric: str  # "drift" or "abs_err"
    use_kappa_cap: bool
    kappa_factor: float
    eta_floor: float
    sdiag_floor_rel: float
    clip_plot_quantile: float

    @property
    def truncation_enabled(self) -> bool:
        return self.use_kappa_cap or (self.eig_cut_rel is not None) or (self.eig_cut_abs is not None)


def hermitize(a: np.ndarray) -> np.ndarray:
    return (a + a.conj().T) / 2.0


def _parse_k_map(folder: Path, regex: re.Pattern[str]) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for p in folder.glob("K*.npy"):
        m = regex.match(p.name)
        if m is None:
            continue
        out[int(m.group(1))] = p
    return out


def load_method_mats(data_root: Path | str, method: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    root = Path(data_root)
    method_dir = root / method
    if not method_dir.exists():
        raise FileNotFoundError(f"Method directory not found: {method_dir}")

    h_map = _parse_k_map(method_dir, K_H_RE)
    s_map = _parse_k_map(method_dir, K_S_RE)
    common_ks = sorted(set(h_map) & set(s_map))
    if not common_ks:
        raise ValueError(f"No complete (H,S) K-pairs found in {method_dir}")

    mats: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for k in common_ks:
        h = np.load(h_map[k])
        s = np.load(s_map[k])
        if h.ndim != 2 or s.ndim != 2:
            raise ValueError(f"K={k}: H and S must be 2D arrays; got {h.ndim}D and {s.ndim}D")
        if h.shape[0] != h.shape[1] or s.shape[0] != s.shape[1]:
            raise ValueError(f"K={k}: H and S must be square; got {h.shape} and {s.shape}")
        if h.shape != s.shape:
            raise ValueError(f"K={k}: H and S shapes must match; got {h.shape} and {s.shape}")

        mats[k] = (hermitize(np.asarray(h)), hermitize(np.asarray(s)))

    missing_h = sorted(set(s_map) - set(h_map))
    missing_s = sorted(set(h_map) - set(s_map))
    if missing_h:
        print(f"[warn] {method}: missing H files for Ks={missing_h}; skipping them.")
    if missing_s:
        print(f"[warn] {method}: missing S files for Ks={missing_s}; skipping them.")

    return mats


def _stable_seed(base_seed: int, *parts: Any) -> int:
    text = "|".join([str(base_seed)] + [str(p) for p in parts])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _phase_sign(z: np.ndarray) -> np.ndarray:
    out = np.ones_like(z, dtype=np.complex128)
    mag = np.abs(z)
    mask = mag > 0.0
    out[mask] = z[mask] / mag[mask]
    return out


def _hermitian_rademacher(n: int, rng: np.random.Generator, complex_out: bool) -> np.ndarray:
    dtype = np.complex128 if complex_out else np.float64
    m = np.zeros((n, n), dtype=dtype)
    pm = np.array([-1.0, 1.0], dtype=np.float64)
    d = rng.choice(pm, size=n)
    m[np.diag_indices(n)] = d
    iu = np.triu_indices(n, 1)
    vals = rng.choice(pm, size=iu[0].size)
    m[iu] = vals
    m[(iu[1], iu[0])] = vals
    return m


def _floor_from_diag(a: np.ndarray, factor: float, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    diag = np.diag(a)
    if diag.size == 0:
        return 0.0
    return float(factor * float(np.max(np.abs(diag))))


def _compute_scales(
    h: np.ndarray,
    s: np.ndarray,
    h0_factor: float,
    s0_factor: float,
    h0_override: Optional[float],
    s0_override: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    h0 = _floor_from_diag(h, h0_factor, h0_override)
    s0 = _floor_from_diag(s, s0_factor, s0_override)
    scale_h = np.maximum(np.abs(h), h0)
    scale_s = np.maximum(np.abs(s), s0)
    return scale_h, scale_s, h0, s0


def perturb_typical(
    h: np.ndarray,
    s: np.ndarray,
    eta: float,
    scale_h: np.ndarray,
    scale_s: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    complex_out = np.iscomplexobj(h) or np.iscomplexobj(s)
    n = h.shape[0]
    r = _hermitian_rademacher(n, rng, complex_out=complex_out)
    q = _hermitian_rademacher(n, rng, complex_out=complex_out)
    delta_h = eta * scale_h * r
    delta_s = eta * scale_s * q
    return hermitize(h + delta_h), hermitize(s + delta_s)


def perturb_adversarial_lite(
    h: np.ndarray,
    s: np.ndarray,
    eta: float,
    scale_h: np.ndarray,
    scale_s: np.ndarray,
    rng: np.random.Generator,
    *,
    adv_h_align: str,
    v_adv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pattern = _phase_sign(np.outer(v_adv, v_adv.conj()))
    delta_s = hermitize(-eta * scale_s * pattern)

    if adv_h_align == "pattern":
        delta_h = hermitize(eta * scale_h * pattern)
    else:
        complex_out = np.iscomplexobj(h) or np.iscomplexobj(s)
        n = h.shape[0]
        r = _hermitian_rademacher(n, rng, complex_out=complex_out)
        delta_h = hermitize(eta * scale_h * r)

    return hermitize(h + delta_h), hermitize(s + delta_s)


def solve_gevp_min(h: np.ndarray, s: np.ndarray) -> float:
    h = hermitize(h)
    s = hermitize(s)

    if HAVE_SCIPY:
        eigs = spla.eigh(  # type: ignore[union-attr]
            h,
            s,
            eigvals_only=True,
            subset_by_index=[0, 0],
            check_finite=False,
        )
        return float(np.real(np.asarray(eigs)[0]))

    l = np.linalg.cholesky(s)
    eye = np.eye(s.shape[0], dtype=s.dtype)
    inv_l = np.linalg.solve(l, eye)
    reduced = hermitize(inv_l @ h @ inv_l.conj().T)
    return float(np.real(np.linalg.eigvalsh(reduced)[0]))


def solve_reduced(h_red: np.ndarray) -> float:
    return float(np.real(np.linalg.eigvalsh(hermitize(h_red))[0]))


def compute_cond_metrics(s: np.ndarray, pd_tol: float) -> Tuple[float, float]:
    vals = np.linalg.eigvalsh(hermitize(s))
    sigma_min = float(np.real(vals[0]))
    sigma_max = float(np.real(vals[-1]))
    if (not np.isfinite(sigma_min)) or (sigma_min <= pd_tol):
        return sigma_min, float("inf")
    return sigma_min, float(sigma_max / sigma_min)


def diag_scale(h: np.ndarray, s: np.ndarray, pd_tol: float, sdiag_floor_rel: float) -> Tuple[np.ndarray, np.ndarray]:
    s = hermitize(s)
    h = hermitize(h)

    sdiag = np.real(np.diag(s))
    finite_diag = sdiag[np.isfinite(sdiag)]
    if finite_diag.size == 0:
        med_diag = float(pd_tol)
    else:
        med_diag = float(np.median(finite_diag))

    sdiag_floor = max(float(pd_tol), float(sdiag_floor_rel) * med_diag)
    safe_diag = np.maximum(sdiag, sdiag_floor)

    d = np.diag((1.0 / np.sqrt(safe_diag)).astype(np.complex128))
    h2 = hermitize(d @ h @ d)
    s2 = hermitize(d @ s @ d)
    return h2, s2


def _kappa_cap_for_eta(eta: float, cfg: RunConfig) -> float:
    if not cfg.use_kappa_cap:
        return float("inf")
    denom = max(float(eta), float(cfg.eta_floor))
    return float(cfg.kappa_factor / denom)


def truncate_with_cap(h: np.ndarray, s: np.ndarray, eta: float, cfg: RunConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    h = hermitize(h)
    s = hermitize(s)

    evals, evecs = np.linalg.eigh(s)
    lam = np.real(evals)
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    u = evecs[:, order]

    k_dim = int(lam.size)
    lam_max = float(lam[0]) if k_dim > 0 else float("nan")
    tau_abs = max(float(cfg.pd_tol), 0.0 if cfg.eig_cut_abs is None else float(cfg.eig_cut_abs))
    tau_rel = 0.0 if cfg.eig_cut_rel is None else float(cfg.eig_cut_rel) * lam_max
    tau_floor = max(tau_abs, tau_rel)
    kappa_cap = _kappa_cap_for_eta(eta, cfg)

    keep: List[int] = []
    for i, lam_i in enumerate(lam):
        if (not np.isfinite(lam_i)) or (lam_i <= tau_floor):
            break
        if cfg.use_kappa_cap:
            if (not np.isfinite(kappa_cap)) or (kappa_cap <= 0.0):
                break
            cond_i = lam_max / lam_i if lam_i > 0.0 else float("inf")
            if cond_i > kappa_cap:
                break
        keep.append(i)

    rank_eff = int(len(keep))
    rank_frac = float(rank_eff / k_dim) if k_dim > 0 else 0.0

    meta: Dict[str, float] = {
        "eff_lambda_min": float("nan"),
        "eff_lambda_max": float("nan"),
        "eff_cond2": float("inf"),
        "rank_eff": float(rank_eff),
        "rank_frac": rank_frac,
        "kappa_cap_used": float(kappa_cap),
    }

    if rank_eff > 0:
        lam_r = lam[keep]
        eff_lmin = float(np.min(lam_r))
        eff_lmax = float(np.max(lam_r))
        meta["eff_lambda_min"] = eff_lmin
        meta["eff_lambda_max"] = eff_lmax
        meta["eff_cond2"] = float(eff_lmax / eff_lmin) if eff_lmin > 0.0 else float("inf")

    if rank_eff < int(cfg.min_rank):
        raise CanonicalOrthogonalizationError(
            f"Retained rank {rank_eff} < min_rank={cfg.min_rank}",
            meta=meta,
        )

    lam_r = lam[keep]
    u_r = u[:, keep]
    x = u_r / np.sqrt(lam_r)[None, :]
    h_red = hermitize(x.conj().T @ h @ x)
    return h_red, meta


def _default_eff_meta_no_trunc(k: int, scaled_sigma: float, scaled_cond: float, eta: float, cfg: RunConfig) -> Dict[str, float]:
    eff_lmin = scaled_sigma if np.isfinite(scaled_sigma) else float("nan")
    if np.isfinite(scaled_cond) and np.isfinite(eff_lmin):
        eff_lmax = float(eff_lmin * scaled_cond)
    else:
        eff_lmax = float("nan")

    return {
        "eff_lambda_min": eff_lmin,
        "eff_lambda_max": eff_lmax,
        "eff_cond2": float(scaled_cond),
        "rank_eff": float(k),
        "rank_frac": 1.0,
        "kappa_cap_used": _kappa_cap_for_eta(eta, cfg),
    }


def _stats(values: Sequence[float], *, positive_only: bool = False) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "iqr": float("nan"),
            "std": float("nan"),
        }

    arr = arr[np.isfinite(arr)]
    if positive_only:
        arr = arr[arr > 0.0]
    if arr.size == 0:
        return {
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "iqr": float("nan"),
            "std": float("nan"),
        }

    q25, q75 = np.percentile(arr, [25.0, 75.0])
    return {
        "median": float(np.median(arr)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "std": float(np.std(arr, ddof=0)),
    }


def _inv_cond(cond2: float) -> float:
    if (not np.isfinite(cond2)) or cond2 <= 0.0:
        return 0.0
    return float(1.0 / cond2)


def _solve_single(h: np.ndarray, s: np.ndarray, eta: float, cfg: RunConfig) -> Tuple[bool, Optional[float], Dict[str, float], bool, bool]:
    h = hermitize(h)
    s = hermitize(s)

    raw_sigma, raw_cond = compute_cond_metrics(s, cfg.pd_tol)
    raw_indef = (not np.isfinite(raw_sigma)) or (raw_sigma <= cfg.pd_tol)

    if cfg.diag_scale:
        h_work, s_work = diag_scale(h, s, cfg.pd_tol, cfg.sdiag_floor_rel)
    else:
        h_work, s_work = h, s

    scaled_sigma, scaled_cond = compute_cond_metrics(s_work, cfg.pd_tol)

    meta: Dict[str, float] = {
        "raw_sigma_min": raw_sigma,
        "raw_cond2": raw_cond,
        "scaled_sigma_min": scaled_sigma,
        "scaled_cond2": scaled_cond,
        "eff_lambda_min": float("nan"),
        "eff_lambda_max": float("nan"),
        "eff_cond2": float("inf"),
        "rank_eff": float("nan"),
        "rank_frac": float("nan"),
        "kappa_cap_used": _kappa_cap_for_eta(eta, cfg),
    }

    solver_failed = False

    if cfg.truncation_enabled:
        try:
            h_red, trunc_meta = truncate_with_cap(h_work, s_work, eta, cfg)
            meta.update(trunc_meta)
            e0 = solve_reduced(h_red)
            return True, float(e0), meta, raw_indef, solver_failed
        except CanonicalOrthogonalizationError as exc:
            meta.update(exc.meta)
            return False, None, meta, raw_indef, solver_failed
        except Exception:
            solver_failed = True
            return False, None, meta, raw_indef, solver_failed

    # Old-style path: no truncation
    meta.update(_default_eff_meta_no_trunc(h.shape[0], scaled_sigma, scaled_cond, eta, cfg))
    if (not np.isfinite(scaled_sigma)) or (scaled_sigma <= cfg.pd_tol):
        return False, None, meta, raw_indef, solver_failed

    try:
        e0 = solve_gevp_min(h_work, s_work)
        return True, float(e0), meta, raw_indef, solver_failed
    except Exception:
        solver_failed = True
        return False, None, meta, raw_indef, solver_failed


def compute_baseline_E0(
    method: str,
    k: int,
    perturbation_type: str,
    h: np.ndarray,
    s: np.ndarray,
    cfg: RunConfig,
) -> Tuple[Optional[float], bool, Dict[str, float]]:
    _ = (method, k, perturbation_type)
    success, e0, meta, _raw_indef, _solver_failed = _solve_single(h, s, eta=0.0, cfg=cfg)
    return e0, success, meta


def run(
    method: str,
    k: int,
    etas: Sequence[float],
    h: np.ndarray,
    s: np.ndarray,
    cfg: RunConfig,
    e_ref_override: Optional[float] = None,
) -> List[Dict[str, Any]]:
    h = hermitize(h)
    s = hermitize(s)

    scale_h, scale_s, h0, s0 = _compute_scales(
        h,
        s,
        h0_factor=cfg.h0_factor,
        s0_factor=cfg.s0_factor,
        h0_override=cfg.h0_override,
        s0_override=cfg.s0_override,
    )

    # Adversarial-lite direction from smallest mode of scaled S if enabled, else raw S.
    s_ref_adv = s
    if cfg.diag_scale:
        _h_tmp, s_ref_adv = diag_scale(h, s, cfg.pd_tol, cfg.sdiag_floor_rel)
    try:
        evals_adv, vecs_adv = np.linalg.eigh(hermitize(s_ref_adv))
        v_adv = vecs_adv[:, int(np.argmin(np.real(evals_adv)))]
    except np.linalg.LinAlgError:
        v_adv = np.ones(s.shape[0], dtype=np.complex128)
        v_adv = v_adv / np.linalg.norm(v_adv)

    if e_ref_override is not None:
        e_ref = float(e_ref_override)
    else:
        try:
            # Baseline reference energy for abs_err metric.
            e_ref = float(solve_gevp_min(h, s))
        except Exception:
            e_ref = float("nan")

    out: List[Dict[str, Any]] = []
    perturbation_plan = [
        ("typical", int(cfg.n_draws)),
        ("adversarial_lite", int(cfg.n_adv)),
    ]

    for ptype, n_total in perturbation_plan:
        if n_total <= 0:
            continue

        baseline_e0, baseline_ok, baseline_meta = compute_baseline_E0(
            method,
            k,
            ptype,
            h,
            s,
            cfg,
        )
        _ = baseline_meta

        for eta in etas:
            eta = float(eta)
            rng = np.random.default_rng(_stable_seed(cfg.seed, method, k, f"{eta:.16e}", ptype))

            n_fail_eff = 0
            n_raw_indef = 0
            n_solver_fail = 0

            e0_success: List[float] = []
            err_success: List[float] = []
            abs_err_success: List[float] = []
            drift_success: List[float] = []

            raw_sigma_all: List[float] = []
            raw_cond_all: List[float] = []
            scaled_sigma_all: List[float] = []
            scaled_cond_all: List[float] = []
            eff_lmin_all: List[float] = []
            eff_lmax_all: List[float] = []
            eff_cond_all: List[float] = []
            rank_eff_all: List[float] = []
            rank_frac_all: List[float] = []
            kappa_used_all: List[float] = []

            for _draw in range(n_total):
                if ptype == "typical":
                    h_pert, s_pert = perturb_typical(h, s, eta, scale_h, scale_s, rng)
                else:
                    h_pert, s_pert = perturb_adversarial_lite(
                        h,
                        s,
                        eta,
                        scale_h,
                        scale_s,
                        rng,
                        adv_h_align=cfg.adv_h_align,
                        v_adv=v_adv,
                    )

                ok, e0, meta, raw_indef, solver_failed = _solve_single(h_pert, s_pert, eta, cfg)

                raw_sigma_all.append(float(meta.get("raw_sigma_min", float("nan"))))
                raw_cond_all.append(float(meta.get("raw_cond2", float("nan"))))
                scaled_sigma_all.append(float(meta.get("scaled_sigma_min", float("nan"))))
                scaled_cond_all.append(float(meta.get("scaled_cond2", float("nan"))))
                eff_lmin_all.append(float(meta.get("eff_lambda_min", float("nan"))))
                eff_lmax_all.append(float(meta.get("eff_lambda_max", float("nan"))))
                eff_cond_all.append(float(meta.get("eff_cond2", float("nan"))))
                rank_eff_all.append(float(meta.get("rank_eff", float("nan"))))
                rank_frac_all.append(float(meta.get("rank_frac", float("nan"))))
                kappa_used_all.append(float(meta.get("kappa_cap_used", float("nan"))))

                if raw_indef:
                    n_raw_indef += 1
                if solver_failed:
                    n_solver_fail += 1

                if (not ok) or (e0 is None):
                    n_fail_eff += 1
                    continue

                # If baseline fails, drift is undefined and this draw is counted as failure by request.
                if not baseline_ok or baseline_e0 is None or not np.isfinite(baseline_e0):
                    n_fail_eff += 1
                    continue

                e0_success.append(float(e0))

                if np.isfinite(e_ref):
                    err = float(e0 - e_ref)
                    err_success.append(err)
                    abs_err_success.append(abs(err))

                drift_success.append(abs(float(e0) - float(baseline_e0)))

            n_success = len(e0_success)
            fail_rate_eff = float(n_fail_eff / n_total)
            raw_indef_rate = float(n_raw_indef / n_total)

            e0_stats = _stats(e0_success)
            err_source = abs_err_success if cfg.abs_error else err_success
            err_stats = _stats(err_source)
            abs_err_stats = _stats(abs_err_success)
            drift_stats = _stats(drift_success)

            raw_sigma_stats = _stats(raw_sigma_all)
            raw_cond_stats = _stats(raw_cond_all)
            scaled_sigma_stats = _stats(scaled_sigma_all)
            scaled_cond_stats = _stats(scaled_cond_all)
            eff_lmin_stats = _stats(eff_lmin_all)
            eff_lmax_stats = _stats(eff_lmax_all)
            eff_cond_stats = _stats(eff_cond_all)
            rank_eff_stats = _stats(rank_eff_all)
            rank_frac_stats = _stats(rank_frac_all)
            kappa_stats = _stats(kappa_used_all)

            record: Dict[str, Any] = {
                "method": method,
                "K": int(k),
                "eta": eta,
                "perturbation_type": ptype,
                "metric": str(cfg.metric),
                "n_total": int(n_total),
                "n_success": int(n_success),
                "n_fail_eff": int(n_fail_eff),
                "fail_rate_eff": fail_rate_eff,
                "n_raw_indef": int(n_raw_indef),
                "raw_indef_rate": raw_indef_rate,
                "n_solver_fail": int(n_solver_fail),
                "h0": h0,
                "s0": s0,
                "E_ref": e_ref,
                "baseline_e0": float(baseline_e0) if baseline_e0 is not None else float("nan"),
                "baseline_ok": float(1.0 if baseline_ok else 0.0),
                "median_E0": e0_stats["median"],
                "q25_E0": e0_stats["q25"],
                "q75_E0": e0_stats["q75"],
                "iqr_E0": e0_stats["iqr"],
                "std_E0": e0_stats["std"],
                "median_err": err_stats["median"],
                "q25_err": err_stats["q25"],
                "q75_err": err_stats["q75"],
                "iqr_err": err_stats["iqr"],
                "std_err": err_stats["std"],
                "median_abs_err": abs_err_stats["median"],
                "q25_abs_err": abs_err_stats["q25"],
                "q75_abs_err": abs_err_stats["q75"],
                "iqr_abs_err": abs_err_stats["iqr"],
                "std_abs_err": abs_err_stats["std"],
                "median_drift": drift_stats["median"],
                "q25_drift": drift_stats["q25"],
                "q75_drift": drift_stats["q75"],
                "iqr_drift": drift_stats["iqr"],
                "std_drift": drift_stats["std"],
                "raw_sigma_min": raw_sigma_stats["median"],
                "raw_cond2": raw_cond_stats["median"],
                "scaled_sigma_min": scaled_sigma_stats["median"],
                "scaled_cond2": scaled_cond_stats["median"],
                "eff_lambda_min": eff_lmin_stats["median"],
                "eff_lambda_max": eff_lmax_stats["median"],
                "eff_cond2": eff_cond_stats["median"],
                "kappa_cap_used": kappa_stats["median"],
                "rank_eff": rank_eff_stats["median"],
                "rank_frac": rank_frac_stats["median"],
                "rank_eff_median": rank_eff_stats["median"],
                "rank_eff_iqr": rank_eff_stats["iqr"],
                "q25_rank_frac": rank_frac_stats["q25"],
                "q75_rank_frac": rank_frac_stats["q75"],
                "iqr_rank_frac": rank_frac_stats["iqr"],
            }
            record["inv_raw_cond2"] = _inv_cond(float(record["raw_cond2"]))
            record["inv_scaled_cond2"] = _inv_cond(float(record["scaled_cond2"]))
            record["inv_eff_cond2"] = _inv_cond(float(record["eff_cond2"]))
            record["inv_kappa_cap"] = _inv_cond(float(record["kappa_cap_used"]))

            out.append(record)

    return out


def write_csv(records: Sequence[Mapping[str, Any]], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("No records to write.")

    fieldnames = [
        "method",
        "K",
        "eta",
        "perturbation_type",
        "metric",
        "n_total",
        "n_success",
        "n_fail_eff",
        "fail_rate_eff",
        "n_raw_indef",
        "raw_indef_rate",
        "n_solver_fail",
        "baseline_e0",
        "baseline_ok",
        "median_E0",
        "iqr_E0",
        "std_E0",
        "q25_E0",
        "q75_E0",
        "median_abs_err",
        "iqr_abs_err",
        "std_abs_err",
        "q25_abs_err",
        "q75_abs_err",
        "median_drift",
        "iqr_drift",
        "std_drift",
        "q25_drift",
        "q75_drift",
        "raw_sigma_min",
        "raw_cond2",
        "scaled_sigma_min",
        "scaled_cond2",
        "eff_lambda_min",
        "eff_lambda_max",
        "eff_cond2",
        "kappa_cap_used",
        "rank_eff",
        "rank_frac",
        "rank_eff_median",
        "rank_eff_iqr",
        "q25_rank_frac",
        "q75_rank_frac",
        "iqr_rank_frac",
        "inv_raw_cond2",
        "inv_scaled_cond2",
        "inv_eff_cond2",
        "inv_kappa_cap",
        "h0",
        "s0",
        "E_ref",
    ]

    def _key(r: Mapping[str, Any]) -> Tuple[str, int, float, str]:
        return (
            str(r["method"]),
            int(r["K"]),
            float(r["eta"]),
            str(r["perturbation_type"]),
        )

    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in sorted(records, key=_key):
            writer.writerow({k: rec.get(k, "") for k in fieldnames})


def _setup_plot_style(dpi: int) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.25,
            "lines.linewidth": 1.8,
        }
    )


def _method_color_map(methods: Sequence[str]) -> Dict[str, str]:
    fixed = {"qkud": "tab:blue", "qrte": "tab:orange"}
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    out: Dict[str, str] = {}
    idx = 0
    for m in methods:
        if m in fixed:
            out[m] = fixed[m]
        else:
            out[m] = palette[idx % len(palette)]
            idx += 1
    return out


def _safe_legend(ax: Any, **kwargs: Any) -> None:
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def _clip_max(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return float("inf")
    if q <= 0.0 or q >= 1.0:
        return float("inf")
    return float(np.quantile(arr, q))


def make_plots(
    records: Sequence[Mapping[str, Any]],
    out_prefix: Path | str,
    k_focus: int,
    cfg: RunConfig,
    *,
    dpi: int,
    plot_tradeoff: bool,
) -> Dict[str, Path]:
    if not records:
        raise ValueError("No records to plot.")

    _setup_plot_style(dpi)
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    panel_base = prefix.parent / f"{prefix.name}_sensitivity_panels"
    spread_base = prefix.parent / f"{prefix.name}_spread_vs_cond"
    tradeoff_base = prefix.parent / f"{prefix.name}_rank_stability_tradeoff"

    methods = sorted({str(r["method"]) for r in records})
    ptypes = [
        p for p in ["typical", "adversarial_lite"] if any(str(r["perturbation_type"]) == p for r in records)
    ]

    colors = _method_color_map(methods)
    line_styles = {"typical": "-", "adversarial_lite": "--"}
    markers = {"typical": "o", "adversarial_lite": "s"}
    ptype_label = {"typical": "typical", "adversarial_lite": "adversarial-lite"}

    focus_records = [r for r in records if int(r["K"]) == int(k_focus)]

    metric_name = str(cfg.metric).lower()
    if metric_name == "abs_err":
        med_field = "median_abs_err"
        q25_field = "q25_abs_err"
        q75_field = "q75_abs_err"
        y1_label = r"Median $|E_0 - E_{\mathrm{ref}}|$ (Hartree)"
    else:
        med_field = "median_drift"
        q25_field = "q25_drift"
        q75_field = "q75_drift"
        y1_label = r"Median $\Delta E(\eta)=|E_0(\eta)-E_0(0)|$ (Hartree)"

    clip1 = _clip_max([float(r.get(med_field, float("nan"))) for r in focus_records], cfg.clip_plot_quantile)
    clip3 = _clip_max([float(r.get("iqr_drift", float("nan"))) for r in records], cfg.clip_plot_quantile)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax1, ax2, ax3 = axes

    for method in methods:
        for ptype in ptypes:
            rows = [
                r for r in focus_records if str(r["method"]) == method and str(r["perturbation_type"]) == ptype
            ]
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: float(r["eta"]))
            x = np.asarray([float(r["eta"]) for r in rows], dtype=float)

            med = np.asarray([float(r.get(med_field, float("nan"))) for r in rows], dtype=float)
            q25 = np.asarray([float(r.get(q25_field, float("nan"))) for r in rows], dtype=float)
            q75 = np.asarray([float(r.get(q75_field, float("nan"))) for r in rows], dtype=float)

            if np.isfinite(clip1):
                med = np.minimum(med, clip1)
                q25 = np.minimum(q25, clip1)
                q75 = np.minimum(q75, clip1)

            yerr_low = np.maximum(med - q25, 0.0)
            yerr_high = np.maximum(q75 - med, 0.0)
            valid = np.isfinite(med) & (med > 0.0)

            label = f"{method.upper()} {ptype_label[ptype]}"
            if np.any(valid):
                ax1.errorbar(
                    x[valid],
                    med[valid],
                    yerr=np.vstack([yerr_low[valid], yerr_high[valid]]),
                    fmt=markers[ptype],
                    linestyle=line_styles[ptype],
                    color=colors[method],
                    capsize=2.5,
                    label=label,
                )

            eff = np.asarray([float(r.get("fail_rate_eff", float("nan"))) for r in rows], dtype=float)
            raw = np.asarray([float(r.get("raw_indef_rate", float("nan"))) for r in rows], dtype=float)

            ax2.plot(
                x,
                eff,
                linestyle=line_styles[ptype],
                marker=markers[ptype],
                color=colors[method],
                label=f"{method.upper()} {ptype_label[ptype]} eff-fail",
            )
            ax2.plot(
                x,
                raw,
                linestyle=":",
                marker=None,
                color=colors[method],
                alpha=0.9,
                label=f"{method.upper()} {ptype_label[ptype]} raw-indef",
            )

    x_field = "inv_eff_cond2" if cfg.truncation_enabled else "inv_raw_cond2"
    x3_label = r"$1/\mathrm{cond}_2(S_{\mathrm{eff}})$" if cfg.truncation_enabled else r"$1/\mathrm{cond}_2(S)$"

    for method in methods:
        for ptype in ptypes:
            rows = [
                r for r in records if str(r["method"]) == method and str(r["perturbation_type"]) == ptype
            ]
            if not rows:
                continue

            x = np.asarray([float(r.get(x_field, float("nan"))) for r in rows], dtype=float)
            y = np.asarray([float(r.get("iqr_drift", float("nan"))) for r in rows], dtype=float)

            if np.isfinite(clip3):
                y = np.minimum(y, clip3)

            valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
            if not np.any(valid):
                continue

            ax3.scatter(
                x[valid],
                y[valid],
                marker=markers[ptype],
                color=colors[method],
                alpha=0.75,
                s=34,
                label=f"{method.upper()} {ptype_label[ptype]}",
            )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Perturbation amplitude $\eta$")
    ax1.set_ylabel(y1_label)
    ax1.set_title(f"Stability Metric vs η (K={k_focus})")
    _safe_legend(ax1, frameon=False, loc="best")

    ax2.set_xscale("log")
    ax2.set_xlabel(r"Perturbation amplitude $\eta$")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title(f"Raw-Indef (dotted) & Eff-Fail (solid), K={k_focus}")
    _safe_legend(ax2, frameon=False, loc="best")

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel(x3_label)
    ax3.set_ylabel(r"$\mathrm{IQR}(\Delta E)$ (Hartree)")
    ax3.set_title("Spread vs Conditioning (pooled K, η)")
    _safe_legend(ax3, frameon=False, loc="best")

    fig.tight_layout()
    panel_png = panel_base.with_suffix(".png")
    panel_pdf = panel_base.with_suffix(".pdf")
    fig.savefig(panel_png)
    fig.savefig(panel_pdf)
    plt.close(fig)

    # Secondary spread-focused figure
    fig2, axs = plt.subplots(figsize=(6.4, 4.8))
    for method in methods:
        for ptype in ptypes:
            rows = [
                r for r in records if str(r["method"]) == method and str(r["perturbation_type"]) == ptype
            ]
            if not rows:
                continue
            x = np.asarray([float(r.get(x_field, float("nan"))) for r in rows], dtype=float)
            y = np.asarray([float(r.get("iqr_drift", float("nan"))) for r in rows], dtype=float)
            if np.isfinite(clip3):
                y = np.minimum(y, clip3)
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
            if not np.any(valid):
                continue
            axs.scatter(
                x[valid],
                y[valid],
                marker=markers[ptype],
                color=colors[method],
                alpha=0.8,
                s=42,
                label=f"{method.upper()} {ptype_label[ptype]}",
            )
    axs.set_xscale("log")
    axs.set_yscale("log")
    axs.set_xlabel(x3_label)
    axs.set_ylabel(r"$\mathrm{IQR}(\Delta E)$ (Hartree)")
    _safe_legend(axs, frameon=False, loc="best")
    axs.grid(True, which="both", linestyle="--", alpha=0.25)
    fig2.tight_layout()
    spread_png = spread_base.with_suffix(".png")
    spread_pdf = spread_base.with_suffix(".pdf")
    fig2.savefig(spread_png)
    fig2.savefig(spread_pdf)
    plt.close(fig2)

    out_paths: Dict[str, Path] = {
        "panel_png": panel_png,
        "panel_pdf": panel_pdf,
        "spread_png": spread_png,
        "spread_pdf": spread_pdf,
    }

    if plot_tradeoff:
        fig3, axr = plt.subplots(figsize=(7.0, 4.8))
        for method in methods:
            for ptype in ptypes:
                rows = [
                    r
                    for r in focus_records
                    if str(r["method"]) == method and str(r["perturbation_type"]) == ptype
                ]
                if not rows:
                    continue
                rows = sorted(rows, key=lambda r: float(r["eta"]))

                x = np.asarray([float(r["eta"]) for r in rows], dtype=float)
                y = np.asarray([float(r.get("rank_frac", float("nan"))) for r in rows], dtype=float)
                y25 = np.asarray([float(r.get("q25_rank_frac", float("nan"))) for r in rows], dtype=float)
                y75 = np.asarray([float(r.get("q75_rank_frac", float("nan"))) for r in rows], dtype=float)

                valid = np.isfinite(x) & np.isfinite(y)
                if not np.any(valid):
                    continue

                label = f"{method.upper()} {ptype_label[ptype]}"
                axr.plot(
                    x[valid],
                    y[valid],
                    marker=markers[ptype],
                    linestyle=line_styles[ptype],
                    color=colors[method],
                    label=label,
                )

                band_valid = valid & np.isfinite(y25) & np.isfinite(y75)
                if np.any(band_valid):
                    axr.fill_between(
                        x[band_valid],
                        y25[band_valid],
                        y75[band_valid],
                        color=colors[method],
                        alpha=0.15,
                        linewidth=0.0,
                    )

        axr.set_xscale("log")
        axr.set_xlabel(r"Perturbation amplitude $\eta$")
        axr.set_ylabel("Retained rank fraction")
        axr.set_ylim(-0.02, 1.02)
        axr.set_title(f"Rank–Stability Tradeoff (K={k_focus})")
        _safe_legend(axr, frameon=False, loc="best")
        axr.grid(True, which="both", linestyle="--", alpha=0.25)
        fig3.tight_layout()
        tradeoff_png = tradeoff_base.with_suffix(".png")
        tradeoff_pdf = tradeoff_base.with_suffix(".pdf")
        fig3.savefig(tradeoff_png)
        fig3.savefig(tradeoff_pdf)
        plt.close(fig3)
        out_paths["tradeoff_png"] = tradeoff_png
        out_paths["tradeoff_pdf"] = tradeoff_pdf

    return out_paths


def _load_optional_e_ref(data_root: Path) -> Optional[float]:
    e_ref_path = data_root / "E_ref.npy"
    if not e_ref_path.exists():
        return None
    arr = np.asarray(np.load(e_ref_path))
    if arr.size == 0:
        return None
    return float(np.ravel(arr)[0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministic sensitivity stress test for H6 GEVP (QRTE vs QKUD)."
    )
    p.add_argument("--data_root", type=Path, default=Path("data/h6"))
    p.add_argument("--methods", nargs="+", default=["qrte", "qkud"])
    p.add_argument("--Ks", nargs="+", type=int, default=None)
    p.add_argument(
        "--etas",
        nargs="+",
        type=float,
        default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    )
    p.add_argument("--n_draws", type=int, default=200)
    p.add_argument("--n_adv", type=int, default=20)
    p.add_argument("--pd_tol", type=float, default=1e-12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--E_ref", type=float, default=None)
    p.add_argument("--abs_error", action="store_true")
    p.add_argument("--h0_factor", type=float, default=1e-10)
    p.add_argument("--s0_factor", type=float, default=1e-10)
    p.add_argument("--h0", type=float, default=None, help="Optional fixed floor for H scaling.")
    p.add_argument("--s0", type=float, default=None, help="Optional fixed floor for S scaling.")
    p.add_argument("--adv_h_align", choices=["random", "pattern"], default="random")

    # New stability controls
    p.add_argument("--metric", choices=["drift", "abs_err"], default="drift")
    p.add_argument("--use_kappa_cap", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--kappa_factor", type=float, default=0.1)
    p.add_argument("--eta_floor", type=float, default=1e-12)

    p.add_argument("--diag_scale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sdiag_floor_rel", type=float, default=1e-12)

    p.add_argument("--eig_cut_rel", type=float, default=None)
    p.add_argument("--eig_cut_abs", type=float, default=None)
    p.add_argument("--min_rank", type=int, default=1)

    p.add_argument("--report_raw_indef", action="store_true")
    p.add_argument("--plot_tradeoff", action=argparse.BooleanOptionalAction, default=True)

    # Backward-compat alias
    p.add_argument("--plot_rank", action="store_true", help=argparse.SUPPRESS)

    p.add_argument("--K_focus", type=int, default=None)
    p.add_argument("--clip_plot_quantile", type=float, default=0.99)
    p.add_argument("--figures_dir", type=Path, default=Path("figures"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    methods: List[str] = [str(m) for m in args.methods]
    etas: List[float] = [float(e) for e in args.etas]

    cfg = RunConfig(
        n_draws=int(args.n_draws),
        n_adv=int(args.n_adv),
        pd_tol=float(args.pd_tol),
        seed=int(args.seed),
        abs_error=bool(args.abs_error),
        h0_factor=float(args.h0_factor),
        s0_factor=float(args.s0_factor),
        h0_override=None if args.h0 is None else float(args.h0),
        s0_override=None if args.s0 is None else float(args.s0),
        adv_h_align=str(args.adv_h_align),
        diag_scale=bool(args.diag_scale),
        eig_cut_rel=None if args.eig_cut_rel is None else float(args.eig_cut_rel),
        eig_cut_abs=None if args.eig_cut_abs is None else float(args.eig_cut_abs),
        min_rank=int(args.min_rank),
        report_raw_indef=bool(args.report_raw_indef),
        metric=str(args.metric),
        use_kappa_cap=bool(args.use_kappa_cap),
        kappa_factor=float(args.kappa_factor),
        eta_floor=float(args.eta_floor),
        sdiag_floor_rel=float(args.sdiag_floor_rel),
        clip_plot_quantile=float(args.clip_plot_quantile),
    )

    method_mats: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for method in methods:
        try:
            method_mats[method] = load_method_mats(data_root, method)
        except Exception as exc:
            print(f"[warn] Skipping method '{method}': {exc}")

    if not method_mats:
        raise RuntimeError(f"No usable method data found under {data_root}")

    available_ks = sorted(set().union(*(set(mats.keys()) for mats in method_mats.values())))
    if not available_ks:
        raise RuntimeError("No Krylov dimensions K found in loaded data.")

    if args.Ks is None:
        target_ks = available_ks
    else:
        target_ks = sorted(set(int(k) for k in args.Ks))

    if not target_ks:
        raise RuntimeError("No target K values selected.")

    if args.E_ref is not None:
        e_ref_global: Optional[float] = float(args.E_ref)
    else:
        e_ref_global = _load_optional_e_ref(data_root)

    all_records: List[Dict[str, Any]] = []
    for method, mats in method_mats.items():
        ks_run = [k for k in target_ks if k in mats]
        if not ks_run:
            print(f"[warn] Method '{method}' has no requested K values in {target_ks}; skipping.")
            continue
        for k in ks_run:
            h, s = mats[k]
            all_records.extend(
                run(
                    method=method,
                    k=k,
                    etas=etas,
                    h=h,
                    s=s,
                    cfg=cfg,
                    e_ref_override=e_ref_global,
                )
            )

    if not all_records:
        raise RuntimeError("No records were generated. Check inputs and Ks.")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / "h6_sensitivity_summary.csv"
    write_csv(all_records, csv_path)

    available_plot_ks = sorted({int(r["K"]) for r in all_records})
    if args.K_focus is None:
        k_focus = max(available_plot_ks)
    else:
        k_focus = int(args.K_focus)
        if k_focus not in available_plot_ks:
            raise RuntimeError(
                f"K_focus={k_focus} not present in generated records. Available Ks={available_plot_ks}"
            )

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    out_paths = make_plots(
        all_records,
        out_prefix=args.figures_dir / "h6",
        k_focus=k_focus,
        cfg=cfg,
        dpi=int(args.dpi),
        plot_tradeoff=bool(args.plot_tradeoff) or bool(args.plot_rank),
    )

    print("Deterministic sensitivity run complete.")
    print(f"metric: {cfg.metric}")
    print(f"diag_scale: {cfg.diag_scale}")
    print(f"use_kappa_cap: {cfg.use_kappa_cap}")
    print(f"kappa_factor: {cfg.kappa_factor}")
    print(f"eta_floor: {cfg.eta_floor}")
    print(f"eig_cut_rel: {cfg.eig_cut_rel}")
    print(f"eig_cut_abs: {cfg.eig_cut_abs}")
    print(f"K_focus used: {k_focus}")
    print(f"CSV saved: {csv_path}")
    print(f"Panels saved: {out_paths['panel_png']} and {out_paths['panel_pdf']}")
    print(f"Spread plot saved: {out_paths['spread_png']} and {out_paths['spread_pdf']}")
    if "tradeoff_png" in out_paths and "tradeoff_pdf" in out_paths:
        print(f"Tradeoff plot saved: {out_paths['tradeoff_png']} and {out_paths['tradeoff_pdf']}")


if __name__ == "__main__":
    main()
