#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


K_H_RE = re.compile(r"^K(\d+)_H\.npy$")
K_S_RE = re.compile(r"^K(\d+)_S\.npy$")


class TruncationFailure(RuntimeError):
    def __init__(self, message: str, meta: Optional[Mapping[str, float]] = None) -> None:
        super().__init__(message)
        self.meta = dict(meta or {})


@dataclass(frozen=True)
class RunConfig:
    etas: Tuple[float, ...]
    n_draws: int
    n_adv: int
    seed: int
    pd_tol: float
    eig_cut_rel: float
    eig_cut_abs: float
    min_rank: int
    kappa_factor: float
    eta_floor: float
    kappa_cap_unpert: float
    kappa_cap_fixed: Optional[float]
    sdiag_floor_rel: float
    h0_factor: float
    s0_factor: float
    h0_override: Optional[float]
    s0_override: Optional[float]
    clip_plot_quantile: float
    rank_plot_mode: str


@dataclass
class MethodData:
    name: str
    per_k: Dict[int, Tuple[np.ndarray, np.ndarray]]
    full_h: Optional[np.ndarray]
    full_s: Optional[np.ndarray]

    @property
    def full_dim(self) -> int:
        if self.full_h is None:
            return 0
        return int(self.full_h.shape[0])

    def available_ks(self, kmax: Optional[int] = None) -> List[int]:
        ks = set(self.per_k.keys())
        if self.full_h is not None:
            ks.update(range(1, self.full_dim + 1))
        if kmax is not None:
            ks = {k for k in ks if k <= int(kmax)}
        return sorted(ks)

    def has_k(self, k: int) -> bool:
        if int(k) in self.per_k:
            return True
        return self.full_h is not None and int(k) <= self.full_dim

    def get_k(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        kk = int(k)
        if kk in self.per_k:
            h, s = self.per_k[kk]
            return hermitize(h.copy()), hermitize(s.copy())
        if self.full_h is not None and self.full_s is not None and kk <= self.full_dim:
            h = self.full_h[:kk, :kk].copy()
            s = self.full_s[:kk, :kk].copy()
            return hermitize(h), hermitize(s)
        raise KeyError(f"{self.name}: K={kk} unavailable")


def hermitize(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + np.asarray(a).conj().T) / 2.0


def _stable_seed(base_seed: int, *parts: Any) -> int:
    token = "|".join([str(base_seed)] + [str(p) for p in parts])
    digest = hashlib.sha256(token.encode("utf-8")).digest()
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
    diag = rng.choice(pm, size=n)
    m[np.diag_indices(n)] = diag
    iu = np.triu_indices(n, 1)
    vals = rng.choice(pm, size=iu[0].size)
    m[iu] = vals
    m[(iu[1], iu[0])] = vals
    return m


def _stats(values: Sequence[float], positive_only: bool = False) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"median": np.nan, "q25": np.nan, "q75": np.nan, "iqr": np.nan, "std": np.nan}
    arr = arr[np.isfinite(arr)]
    if positive_only:
        arr = arr[arr > 0.0]
    if arr.size == 0:
        return {"median": np.nan, "q25": np.nan, "q75": np.nan, "iqr": np.nan, "std": np.nan}
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    return {
        "median": float(np.median(arr)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "std": float(np.std(arr, ddof=0)),
    }


def _safe_inv(x: float) -> float:
    if (not np.isfinite(x)) or x <= 0.0:
        return 0.0
    return float(1.0 / x)


def _parse_k_paths(method_dir: Path, regex: re.Pattern[str]) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for p in method_dir.glob("K*.npy"):
        m = regex.match(p.name)
        if m is None:
            continue
        out[int(m.group(1))] = p
    return out


def _load_matrix_pair(h_path: Path, s_path: Path, k_hint: int) -> Tuple[np.ndarray, np.ndarray]:
    h = np.asarray(np.load(h_path))
    s = np.asarray(np.load(s_path))
    if h.ndim != 2 or s.ndim != 2:
        raise ValueError(f"K={k_hint}: H/S must be 2D")
    if h.shape[0] != h.shape[1] or s.shape[0] != s.shape[1]:
        raise ValueError(f"K={k_hint}: H/S must be square")
    if h.shape != s.shape:
        raise ValueError(f"K={k_hint}: H/S shape mismatch {h.shape} vs {s.shape}")
    if h.shape[0] < k_hint:
        raise ValueError(f"K={k_hint}: matrix shape {h.shape} is too small")
    if h.shape[0] != k_hint:
        h = h[:k_hint, :k_hint]
        s = s[:k_hint, :k_hint]
    return hermitize(h), hermitize(s)


def load_method_data(data_root: Path, method: str) -> MethodData:
    method_dir = data_root / method
    if not method_dir.exists():
        raise FileNotFoundError(f"Missing method directory: {method_dir}")

    h_map = _parse_k_paths(method_dir, K_H_RE)
    s_map = _parse_k_paths(method_dir, K_S_RE)
    common = sorted(set(h_map) & set(s_map))
    per_k: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for k in common:
        try:
            per_k[k] = _load_matrix_pair(h_map[k], s_map[k], k)
        except Exception as exc:
            print(f"[warn] {method}: skipping K={k} due to load/shape issue: {exc}")

    miss_h = sorted(set(s_map) - set(h_map))
    miss_s = sorted(set(h_map) - set(s_map))
    if miss_h:
        print(f"[warn] {method}: missing K*_H for K={miss_h}")
    if miss_s:
        print(f"[warn] {method}: missing K*_S for K={miss_s}")

    full_h_path = method_dir / "H_full.npy"
    full_s_path = method_dir / "S_full.npy"
    full_h: Optional[np.ndarray] = None
    full_s: Optional[np.ndarray] = None
    if full_h_path.exists() and full_s_path.exists():
        hh = np.asarray(np.load(full_h_path))
        ss = np.asarray(np.load(full_s_path))
        if hh.ndim != 2 or ss.ndim != 2 or hh.shape != ss.shape:
            raise ValueError(f"{method}: invalid full matrix shapes {hh.shape} and {ss.shape}")
        if hh.shape[0] != hh.shape[1]:
            raise ValueError(f"{method}: full matrices must be square")
        full_h = hermitize(hh)
        full_s = hermitize(ss)
    elif full_h_path.exists() != full_s_path.exists():
        print(f"[warn] {method}: only one of H_full/S_full exists; ignoring full matrices.")

    if not per_k and full_h is None:
        raise ValueError(f"{method}: no usable per-K files or full matrices")

    return MethodData(name=method, per_k=per_k, full_h=full_h, full_s=full_s)


def load_optional_e_ref(data_root: Path) -> Optional[float]:
    p = data_root / "E_ref.npy"
    if not p.exists():
        return None
    arr = np.asarray(np.load(p))
    if arr.size == 0:
        return None
    return float(np.ravel(arr)[0])


def _floor_from_diag(a: np.ndarray, factor: float, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    d = np.diag(a)
    if d.size == 0:
        return 0.0
    return float(factor * float(np.max(np.abs(d))))


def compute_perturb_scales(
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
    n = h.shape[0]
    complex_out = np.iscomplexobj(h) or np.iscomplexobj(s)
    r = _hermitian_rademacher(n, rng, complex_out=complex_out)
    q = _hermitian_rademacher(n, rng, complex_out=complex_out)
    dh = eta * scale_h * r
    ds = eta * scale_s * q
    return hermitize(h + dh), hermitize(s + ds)


def perturb_adversarial_lite(
    h: np.ndarray,
    s: np.ndarray,
    eta: float,
    scale_h: np.ndarray,
    scale_s: np.ndarray,
    v_adv: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n = h.shape[0]
    complex_out = np.iscomplexobj(h) or np.iscomplexobj(s)
    r = _hermitian_rademacher(n, rng, complex_out=complex_out)
    dh = eta * scale_h * r
    pattern = _phase_sign(np.outer(v_adv, v_adv.conj()))
    ds = -eta * scale_s * pattern
    return hermitize(h + dh), hermitize(s + ds)


def compute_cond_metrics(s: np.ndarray, pd_tol: float) -> Tuple[float, float]:
    vals = np.linalg.eigvalsh(hermitize(s))
    lam_min = float(np.real(vals[0]))
    lam_max = float(np.real(vals[-1]))
    if (not np.isfinite(lam_min)) or (lam_min <= pd_tol) or (not np.isfinite(lam_max)):
        return lam_min, float("inf")
    return lam_min, float(lam_max / lam_min)


def _diag_scaling_matrix(s: np.ndarray, pd_tol: float, sdiag_floor_rel: float) -> np.ndarray:
    s = hermitize(s)
    diag = np.real(np.diag(s))
    finite = diag[np.isfinite(diag)]
    if finite.size == 0:
        med = float(pd_tol)
    else:
        med = float(np.median(finite))
    sdiag_floor = float(sdiag_floor_rel) * med
    if (not np.isfinite(sdiag_floor)) or (sdiag_floor <= 0.0):
        sdiag_floor = float(pd_tol)
    sdiag_floor = max(float(pd_tol), sdiag_floor)
    safe = np.maximum(diag, sdiag_floor)
    inv_sqrt = 1.0 / np.sqrt(safe)
    return np.diag(inv_sqrt.astype(np.complex128))


def diag_scale_mats(h: np.ndarray, s: np.ndarray, pd_tol: float, sdiag_floor_rel: float) -> Tuple[np.ndarray, np.ndarray]:
    d = _diag_scaling_matrix(s, pd_tol, sdiag_floor_rel)
    hs = hermitize(d @ h @ d)
    ss = hermitize(d @ s @ d)
    return hs, ss


def compute_adv_vector_from_scaled_s(s: np.ndarray, pd_tol: float, sdiag_floor_rel: float) -> np.ndarray:
    d = _diag_scaling_matrix(s, pd_tol, sdiag_floor_rel)
    ss = hermitize(d @ s @ d)
    vals, vecs = np.linalg.eigh(ss)
    idx = int(np.argmin(np.real(vals)))
    v = vecs[:, idx]
    norm = np.linalg.norm(v)
    if norm <= 0.0:
        return np.ones(s.shape[0], dtype=np.complex128) / np.sqrt(float(s.shape[0]))
    return v / norm


def kappa_cap(eta: float, kappa_factor: float, eta_floor: float) -> float:
    return float(kappa_factor / max(float(eta), float(eta_floor)))


def resolve_kappa_cap_sensitivity(eta: float, cfg: RunConfig) -> float:
    if cfg.kappa_cap_fixed is not None:
        return float(cfg.kappa_cap_fixed)
    return kappa_cap(eta, cfg.kappa_factor, cfg.eta_floor)


def truncate_with_cap(
    hs: np.ndarray,
    ss: np.ndarray,
    kappa_cap_value: float,
    cfg: RunConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    evals, evecs = np.linalg.eigh(hermitize(ss))
    lam = np.real(evals)
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    u = evecs[:, order]

    kdim = int(lam.size)
    if kdim == 0:
        raise TruncationFailure("Empty overlap spectrum", meta={"rank_eff": 0.0, "rank_frac": 0.0})

    lam_max = float(lam[0])
    cap = float(kappa_cap_value)
    tau_abs = max(float(cfg.pd_tol), float(cfg.eig_cut_abs))
    tau_rel = float(cfg.eig_cut_rel) * lam_max
    tau = max(tau_abs, tau_rel)

    keep: List[int] = []
    for i, lv in enumerate(lam):
        if (not np.isfinite(lv)) or (lv <= tau):
            break
        cond_i = lam_max / lv if lv > 0.0 else float("inf")
        if np.isfinite(cap) and (cond_i > cap):
            break
        keep.append(i)

    rank_eff = int(len(keep))
    rank_frac = float(rank_eff / kdim)
    meta = {
        "kappa_cap_used": float(cap),
        "rank_eff": float(rank_eff),
        "rank_frac": float(rank_frac),
        "eff_lambda_min": float("nan"),
        "eff_lambda_max": float("nan"),
        "eff_cond2": float("inf"),
    }

    if rank_eff > 0:
        lam_r = lam[keep]
        eff_lmin = float(np.min(lam_r))
        eff_lmax = float(np.max(lam_r))
        eff_cond = float(eff_lmax / eff_lmin) if eff_lmin > 0.0 else float("inf")
        meta["eff_lambda_min"] = eff_lmin
        meta["eff_lambda_max"] = eff_lmax
        meta["eff_cond2"] = eff_cond

    if rank_eff < int(cfg.min_rank):
        raise TruncationFailure(
            f"Retained rank {rank_eff} < min_rank {cfg.min_rank}",
            meta=meta,
        )

    lam_r = lam[keep]
    u_r = u[:, keep]
    x = u_r / np.sqrt(lam_r)[None, :]
    h_red = hermitize(x.conj().T @ hs @ x)
    return h_red, meta


def solve_reduced(h_red: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(hermitize(h_red))
    return float(np.real(vals[0]))


def solve_thresholded(h: np.ndarray, s: np.ndarray, kappa_cap_value: float, cfg: RunConfig) -> Dict[str, Any]:
    h = hermitize(h)
    s = hermitize(s)

    raw_sigma_min, raw_cond2 = compute_cond_metrics(s, cfg.pd_tol)
    raw_indef = bool((not np.isfinite(raw_sigma_min)) or (raw_sigma_min <= cfg.pd_tol))

    hs, ss = diag_scale_mats(h, s, cfg.pd_tol, cfg.sdiag_floor_rel)
    scaled_sigma_min, scaled_cond2 = compute_cond_metrics(ss, cfg.pd_tol)

    meta: Dict[str, Any] = {
        "raw_sigma_min": float(raw_sigma_min),
        "raw_cond2": float(raw_cond2),
        "scaled_sigma_min": float(scaled_sigma_min),
        "scaled_cond2": float(scaled_cond2),
        "kappa_cap_used": float(kappa_cap_value),
        "rank_eff": float(0.0),
        "rank_frac": float(0.0),
        "eff_lambda_min": float("nan"),
        "eff_lambda_max": float("nan"),
        "eff_cond2": float("inf"),
        "raw_indef": raw_indef,
        "eff_fail": True,
        "success": False,
        "E0": float("nan"),
    }

    try:
        h_red, trunc_meta = truncate_with_cap(hs, ss, kappa_cap_value, cfg)
        meta.update(trunc_meta)
        e0 = solve_reduced(h_red)
        meta["E0"] = float(e0)
        meta["success"] = True
        meta["eff_fail"] = False
        return meta
    except TruncationFailure as exc:
        meta.update(exc.meta)
        meta["eff_fail"] = True
        return meta
    except Exception:
        meta["eff_fail"] = True
        return meta


def build_unperturbed_baselines(
    method_data: Mapping[str, MethodData],
    methods: Sequence[str],
    kmax: int,
    cfg: RunConfig,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for method in methods:
        if method not in method_data:
            continue
        md = method_data[method]
        recs: Dict[int, Dict[str, Any]] = {}
        for k in md.available_ks(kmax):
            h, s = md.get_k(k)
            recs[k] = solve_thresholded(h, s, kappa_cap_value=cfg.kappa_cap_unpert, cfg=cfg)
        out[method] = recs
    return out


def run_window_sensitivity(
    method_data: Mapping[str, MethodData],
    methods: Sequence[str],
    k_window: Sequence[int],
    cfg: RunConfig,
    e_ref: Optional[float],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for method in methods:
        md = method_data.get(method)
        if md is None:
            continue
        for k in k_window:
            if not md.has_k(int(k)):
                continue
            h, s = md.get_k(int(k))

            scale_h, scale_s, _h0, _s0 = compute_perturb_scales(
                h,
                s,
                h0_factor=cfg.h0_factor,
                s0_factor=cfg.s0_factor,
                h0_override=cfg.h0_override,
                s0_override=cfg.s0_override,
            )
            v_adv = compute_adv_vector_from_scaled_s(s, cfg.pd_tol, cfg.sdiag_floor_rel)

            baseline_by_eta: Dict[float, Dict[str, Any]] = {}
            for eta in cfg.etas:
                eta = float(eta)
                cap_used = resolve_kappa_cap_sensitivity(eta, cfg)
                baseline_by_eta[eta] = solve_thresholded(h, s, kappa_cap_value=cap_used, cfg=cfg)

            plan = [("typical", int(cfg.n_draws)), ("adversarial-lite", int(cfg.n_adv))]
            for perturb_type, n_total in plan:
                if n_total <= 0:
                    continue
                for eta in cfg.etas:
                    eta = float(eta)
                    cap_used = resolve_kappa_cap_sensitivity(eta, cfg)
                    rng = np.random.default_rng(_stable_seed(cfg.seed, method, k, perturb_type, f"{eta:.16e}"))
                    n_raw_indef = 0
                    n_eff_fail = 0
                    drifts: List[float] = []
                    rank_eff_vals: List[float] = []
                    rank_frac_vals: List[float] = []
                    eff_cond_vals: List[float] = []
                    inv_eff_cond_vals: List[float] = []

                    eta_base = baseline_by_eta[eta]
                    e0_unpert_eta = float(eta_base.get("E0", np.nan))
                    baseline_ok = bool(eta_base.get("success", False)) and np.isfinite(e0_unpert_eta)
                    baseline_bias = (
                        float(abs(e0_unpert_eta - e_ref))
                        if (e_ref is not None and np.isfinite(e0_unpert_eta))
                        else float("nan")
                    )

                    for _ in range(n_total):
                        if perturb_type == "typical":
                            h_pert, s_pert = perturb_typical(h, s, eta, scale_h, scale_s, rng)
                        else:
                            h_pert, s_pert = perturb_adversarial_lite(h, s, eta, scale_h, scale_s, v_adv, rng)

                        sol = solve_thresholded(h_pert, s_pert, kappa_cap_value=cap_used, cfg=cfg)
                        raw_indef = bool(sol.get("raw_indef", False))
                        eff_fail = bool(sol.get("eff_fail", True))

                        if raw_indef:
                            n_raw_indef += 1

                        rank_eff = float(sol.get("rank_eff", np.nan))
                        rank_frac = float(sol.get("rank_frac", np.nan))
                        eff_cond = float(sol.get("eff_cond2", np.nan))
                        rank_eff_vals.append(rank_eff)
                        rank_frac_vals.append(rank_frac)
                        eff_cond_vals.append(eff_cond)
                        inv_eff_cond_vals.append(_safe_inv(eff_cond))

                        this_fail = eff_fail or (not baseline_ok)
                        if this_fail:
                            n_eff_fail += 1
                            continue

                        e0 = float(sol.get("E0", np.nan))
                        if not np.isfinite(e0):
                            n_eff_fail += 1
                            continue
                        drifts.append(abs(e0 - e0_unpert_eta))

                    rank_eff_stats = _stats(rank_eff_vals)
                    rank_frac_stats = _stats(rank_frac_vals)
                    eff_cond_stats = _stats(eff_cond_vals)
                    inv_eff_cond_stats = _stats(inv_eff_cond_vals, positive_only=True)
                    drift_stats = _stats(drifts, positive_only=True)

                    row: Dict[str, Any] = {
                        "method": method,
                        "K": int(k),
                        "eta": float(eta),
                        "perturb_type": perturb_type,
                        "n_total": int(n_total),
                        "n_success": int(max(0, n_total - n_eff_fail)),
                        "raw_indef_rate": float(n_raw_indef / n_total),
                        "eff_fail_rate": float(n_eff_fail / n_total),
                        "kappa_cap_used": float(cap_used),
                        "rank_eff_median": rank_eff_stats["median"],
                        "rank_eff_iqr": rank_eff_stats["iqr"],
                        "rank_eff_q25": rank_eff_stats["q25"],
                        "rank_eff_q75": rank_eff_stats["q75"],
                        "rank_frac_median": rank_frac_stats["median"],
                        "rank_frac_iqr": rank_frac_stats["iqr"],
                        "rank_frac_q25": rank_frac_stats["q25"],
                        "rank_frac_q75": rank_frac_stats["q75"],
                        "eff_cond2_median": eff_cond_stats["median"],
                        "eff_cond2_iqr": eff_cond_stats["iqr"],
                        "inv_eff_cond2_median": inv_eff_cond_stats["median"],
                        "inv_eff_cond2_iqr": inv_eff_cond_stats["iqr"],
                        "inv_eff_cond2_q25": inv_eff_cond_stats["q25"],
                        "inv_eff_cond2_q75": inv_eff_cond_stats["q75"],
                        "median_drift": drift_stats["median"],
                        "q25_drift": drift_stats["q25"],
                        "q75_drift": drift_stats["q75"],
                        "iqr_drift": drift_stats["iqr"],
                        "std_drift": drift_stats["std"],
                        "E0_unpert_eta": e0_unpert_eta,
                        "baseline_bias": baseline_bias,
                    }
                    records.append(row)

    return records


def write_summary_csv(records: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "K",
        "eta",
        "perturb_type",
        "n_total",
        "n_success",
        "raw_indef_rate",
        "eff_fail_rate",
        "kappa_cap_used",
        "rank_eff_median",
        "rank_eff_iqr",
        "rank_eff_q25",
        "rank_eff_q75",
        "rank_frac_median",
        "rank_frac_iqr",
        "rank_frac_q25",
        "rank_frac_q75",
        "eff_cond2_median",
        "eff_cond2_iqr",
        "inv_eff_cond2_median",
        "inv_eff_cond2_iqr",
        "inv_eff_cond2_q25",
        "inv_eff_cond2_q75",
        "median_drift",
        "q25_drift",
        "q75_drift",
        "iqr_drift",
        "std_drift",
        "E0_unpert_eta",
        "baseline_bias",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in sorted(
            records,
            key=lambda r: (str(r["method"]), int(r["K"]), float(r["eta"]), str(r["perturb_type"])),
        ):
            writer.writerow({k: rec.get(k, "") for k in fields})


def _setup_plot_style(dpi: int) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 2.0,
        }
    )


def _clip_upper(values: Iterable[float], quantile: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return float("inf")
    q = float(quantile)
    if q <= 0.0 or q >= 1.0:
        return float("inf")
    return float(np.quantile(arr, q))


def plot_unperturbed_convergence(
    baselines: Mapping[str, Mapping[int, Mapping[str, Any]]],
    methods: Sequence[str],
    k_window: Sequence[int],
    kmax: int,
    e_ref_input: Optional[float],
    out_png: Path,
    out_pdf: Path,
    dpi: int,
) -> None:
    _setup_plot_style(dpi)
    colors = {"qkud": "tab:blue", "qrte": "tab:orange"}

    finite_baselines: List[float] = []
    for method in methods:
        for k, rec in baselines.get(method, {}).items():
            if k > kmax:
                continue
            e0 = float(rec.get("E0", np.nan))
            ok = bool(rec.get("success", False))
            if ok and np.isfinite(e0):
                finite_baselines.append(e0)

    if e_ref_input is not None:
        e_ref_plot = float(e_ref_input)
    elif finite_baselines:
        e_ref_plot = float(np.min(finite_baselines))
    else:
        e_ref_plot = float(0.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in methods:
        recs = baselines.get(method, {})
        ks = sorted([k for k in recs.keys() if k <= kmax])
        if not ks:
            continue
        x: List[int] = []
        y: List[float] = []
        for k in ks:
            e0 = float(recs[k].get("E0", np.nan))
            ok = bool(recs[k].get("success", False))
            if not ok or (not np.isfinite(e0)):
                continue
            yy = abs(e0 - e_ref_plot)
            if yy <= 0.0:
                yy = 1e-16
            x.append(int(k))
            y.append(float(yy))
        if not x:
            continue
        ax.plot(x, y, marker="o", markersize=3, color=colors.get(method, None), label=method.upper())

    for kw in k_window:
        if kw <= kmax:
            ax.axvline(float(kw), color="0.5", linestyle=":", linewidth=1.3)

    ax.set_yscale("log")
    ax.set_xlabel("Krylov Dimension K")
    ax.set_ylabel(r"$|E_0(K)-E_{\mathrm{ref}}|$ (Hartree)")
    ax.set_title(r"H6 (STO-3G, R=5.0 $\AA$): Unperturbed Krylov Convergence")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_window_sensitivity(
    records: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    k_window: Sequence[int],
    out_png: Path,
    out_pdf: Path,
    clip_plot_quantile: float,
    rank_plot_mode: str,
    dpi: int,
) -> None:
    _setup_plot_style(dpi)
    colors = {"qkud": "tab:blue", "qrte": "tab:orange"}
    linestyles = {"typical": "-", "adversarial-lite": "--"}
    markers = {"typical": "o", "adversarial-lite": "s"}

    rows_k = [k for k in k_window if any(int(r["K"]) == int(k) for r in records)]
    if not rows_k:
        raise RuntimeError("No records available for requested K_window.")

    drift_clip = _clip_upper(
        [float(r.get("median_drift", np.nan)) for r in records]
        + [float(r.get("q75_drift", np.nan)) for r in records]
        + [float(r.get("iqr_drift", np.nan)) for r in records],
        clip_plot_quantile,
    )
    clipped_any = False

    n_rows = len(rows_k)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16.0, 4.4 * n_rows), squeeze=False)

    legend_handles: List[Any] = []
    legend_labels: List[str] = []

    for i, k in enumerate(rows_k):
        ax_drift = axes[i, 0]
        ax_rank = axes[i, 1]
        ax_scatter = axes[i, 2]
        ax_rank_frac = ax_rank.twinx() if rank_plot_mode == "both" else None

        k_records = [r for r in records if int(r["K"]) == int(k)]

        for method in methods:
            for ptype in ["typical", "adversarial-lite"]:
                rows = [
                    r for r in k_records if str(r["method"]) == method and str(r["perturb_type"]) == ptype
                ]
                if not rows:
                    continue
                rows = sorted(rows, key=lambda r: float(r["eta"]))

                x = np.asarray([float(r["eta"]) for r in rows], dtype=float)

                med_drift = np.asarray([float(r.get("median_drift", np.nan)) for r in rows], dtype=float)
                q25_drift = np.asarray([float(r.get("q25_drift", np.nan)) for r in rows], dtype=float)
                q75_drift = np.asarray([float(r.get("q75_drift", np.nan)) for r in rows], dtype=float)
                if np.isfinite(drift_clip):
                    clipped_any = clipped_any or bool(np.any(med_drift > drift_clip) or np.any(q75_drift > drift_clip))
                    med_drift = np.minimum(med_drift, drift_clip)
                    q25_drift = np.minimum(q25_drift, drift_clip)
                    q75_drift = np.minimum(q75_drift, drift_clip)

                yerr_lo = np.maximum(med_drift - q25_drift, 0.0)
                yerr_hi = np.maximum(q75_drift - med_drift, 0.0)
                valid = np.isfinite(med_drift) & (med_drift > 0.0) & np.isfinite(x)
                if np.any(valid):
                    eb = ax_drift.errorbar(
                        x[valid],
                        med_drift[valid],
                        yerr=np.vstack([yerr_lo[valid], yerr_hi[valid]]),
                        color=colors.get(method, None),
                        linestyle=linestyles[ptype],
                        marker=markers[ptype],
                        markersize=6,
                        capsize=2.5,
                        label=f"{method.upper()} {ptype}",
                    )
                    if i == 0:
                        legend_handles.append(eb[0])
                        legend_labels.append(f"{method.upper()} {ptype}")

                rank_abs_med = np.asarray([float(r.get("rank_eff_median", np.nan)) for r in rows], dtype=float)
                rank_abs_q25 = np.asarray([float(r.get("rank_eff_q25", np.nan)) for r in rows], dtype=float)
                rank_abs_q75 = np.asarray([float(r.get("rank_eff_q75", np.nan)) for r in rows], dtype=float)
                rank_frac_med = np.asarray([float(r.get("rank_frac_median", np.nan)) for r in rows], dtype=float)
                rank_frac_q25 = np.asarray([float(r.get("rank_frac_q25", np.nan)) for r in rows], dtype=float)
                rank_frac_q75 = np.asarray([float(r.get("rank_frac_q75", np.nan)) for r in rows], dtype=float)

                if rank_plot_mode in {"abs", "both"}:
                    valid_abs = np.isfinite(rank_abs_med) & np.isfinite(x)
                    if np.any(valid_abs):
                        ax_rank.plot(
                            x[valid_abs],
                            rank_abs_med[valid_abs],
                            color=colors.get(method, None),
                            linestyle=linestyles[ptype],
                            marker=markers[ptype],
                            markersize=6,
                        )
                        band_ok = valid_abs & np.isfinite(rank_abs_q25) & np.isfinite(rank_abs_q75)
                        if np.any(band_ok):
                            ax_rank.fill_between(
                                x[band_ok],
                                rank_abs_q25[band_ok],
                                rank_abs_q75[band_ok],
                                color=colors.get(method, None),
                                alpha=0.14,
                                linewidth=0.0,
                            )

                if rank_plot_mode in {"frac", "both"}:
                    target_ax = ax_rank if rank_plot_mode == "frac" else ax_rank_frac
                    if target_ax is not None:
                        valid_frac = np.isfinite(rank_frac_med) & np.isfinite(x)
                        if np.any(valid_frac):
                            target_ax.plot(
                                x[valid_frac],
                                rank_frac_med[valid_frac],
                                color=colors.get(method, None),
                                linestyle=":" if rank_plot_mode == "both" else linestyles[ptype],
                                marker=markers[ptype],
                                markersize=5,
                                alpha=0.8 if rank_plot_mode == "both" else 1.0,
                            )
                            band_ok = valid_frac & np.isfinite(rank_frac_q25) & np.isfinite(rank_frac_q75)
                            if np.any(band_ok):
                                target_ax.fill_between(
                                    x[band_ok],
                                    rank_frac_q25[band_ok],
                                    rank_frac_q75[band_ok],
                                    color=colors.get(method, None),
                                    alpha=0.10,
                                    linewidth=0.0,
                                )

                x_sc = np.asarray([float(r.get("inv_eff_cond2_median", np.nan)) for r in rows], dtype=float)
                y_sc = np.asarray([float(r.get("iqr_drift", np.nan)) for r in rows], dtype=float)
                if np.isfinite(drift_clip):
                    clipped_any = clipped_any or bool(np.any(y_sc > drift_clip))
                    y_sc = np.minimum(y_sc, drift_clip)
                valid_sc = np.isfinite(x_sc) & np.isfinite(y_sc) & (x_sc > 0.0) & (y_sc > 0.0)
                if np.any(valid_sc):
                    j_rng = np.random.default_rng(_stable_seed(112358, method, ptype, k, i, "scatter"))
                    x_plot = x_sc[valid_sc] * np.exp(j_rng.normal(0.0, 0.03, size=np.count_nonzero(valid_sc)))
                    ax_scatter.scatter(
                        x_plot,
                        y_sc[valid_sc],
                        color=colors.get(method, None),
                        marker=markers[ptype],
                        s=44,
                        alpha=0.82,
                    )

        ax_drift.set_xscale("log")
        ax_drift.set_yscale("log")
        ax_drift.set_title(f"K={k}  Drift vs $\\eta$")
        ax_drift.set_ylabel(r"Median $\Delta E$ (Hartree)")

        ax_rank.set_xscale("log")
        ax_rank.set_title(f"K={k}  Rank Retention")
        if rank_plot_mode == "abs":
            ax_rank.set_ylabel(r"Retained rank $r_{\mathrm{eff}}$")
            ax_rank.set_ylim(0.0, float(k) + 1.0)
        elif rank_plot_mode == "frac":
            ax_rank.set_ylabel("Rank Fraction")
            ax_rank.set_ylim(0.0, 1.05)
        else:
            ax_rank.set_ylabel(r"Retained rank $r_{\mathrm{eff}}$")
            ax_rank.set_ylim(0.0, float(k) + 1.0)
            if ax_rank_frac is not None:
                ax_rank_frac.set_ylabel("Rank Fraction")
                ax_rank_frac.set_ylim(0.0, 1.05)

        ax_scatter.set_xscale("log")
        ax_scatter.set_yscale("log")
        ax_scatter.set_title(f"K={k}  IQR($\\Delta E$) vs $1/\\mathrm{{cond}}_2$")
        ax_scatter.set_ylabel(r"$\mathrm{IQR}(\Delta E)$ (Hartree)")

        if i == n_rows - 1:
            ax_drift.set_xlabel(r"Perturbation amplitude $\eta$")
            ax_rank.set_xlabel(r"Perturbation amplitude $\eta$")
            ax_scatter.set_xlabel(r"$1/\mathrm{cond}_2(S_{\mathrm{eff}})$")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(4, len(legend_handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
        )

    if clipped_any and np.isfinite(drift_clip):
        fig.text(
            0.5,
            0.004,
            f"Drift statistics clipped above {drift_clip:.2e} (quantile={clip_plot_quantile:.2f}) for visibility.",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def print_sanity(records: Sequence[Mapping[str, Any]], methods: Sequence[str], k_window: Sequence[int]) -> None:
    sanity_etas = [1e-6, 1e-2]
    for method in methods:
        for k in k_window:
            for eta_target in sanity_etas:
                eta_rows = [
                    r
                    for r in records
                    if str(r.get("method")) == method
                    and int(r.get("K")) == int(k)
                    and np.isclose(float(r.get("eta")), float(eta_target), rtol=0.0, atol=1e-20)
                ]
                if not eta_rows:
                    continue
                for ptype in ["typical", "adversarial-lite"]:
                    row = next((rr for rr in eta_rows if str(rr.get("perturb_type")) == ptype), None)
                    if row is None:
                        continue
                    print(
                        "[sanity] "
                        f"method={method} K={k} eta={eta_target:.0e} ptype={ptype} "
                        f"rank_eff_median={float(row.get('rank_eff_median', np.nan)):.6g} "
                        f"inv_eff_cond2_median={float(row.get('inv_eff_cond2_median', np.nan)):.6g} "
                        f"median_drift={float(row.get('median_drift', np.nan)):.6g}"
                    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H6 transition-window deterministic sensitivity analysis.")
    p.add_argument("--data_root", type=Path, default=Path("data/h6"))
    p.add_argument("--methods", nargs="+", default=["qrte", "qkud"])
    p.add_argument("--E_ref", type=float, default=None)
    p.add_argument("--etas", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    p.add_argument("--n_draws", type=int, default=200)
    p.add_argument("--n_adv", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--K_window", nargs="+", type=int, default=[35, 40, 50])
    p.add_argument("--Kmax", type=int, default=None)
    p.add_argument("--kappa_factor", type=float, default=0.1)
    p.add_argument("--eta_floor", type=float, default=1e-12)
    p.add_argument("--kappa_cap_unpert", type=float, default=1e12)
    p.add_argument("--kappa_cap_fixed", type=float, default=None)
    p.add_argument("--pd_tol", type=float, default=1e-12)
    p.add_argument("--eig_cut_rel", type=float, default=0.0)
    p.add_argument("--eig_cut_abs", type=float, default=0.0)
    p.add_argument("--min_rank", type=int, default=2)
    p.add_argument("--sdiag_floor_rel", type=float, default=1e-12)
    p.add_argument("--clip_plot_quantile", type=float, default=0.99)
    p.add_argument("--rank_plot_mode", choices=["abs", "frac", "both"], default="abs")
    p.add_argument("--h0_factor", type=float, default=1e-10)
    p.add_argument("--s0_factor", type=float, default=1e-10)
    p.add_argument("--h0", type=float, default=None)
    p.add_argument("--s0", type=float, default=None)
    p.add_argument("--figures_dir", type=Path, default=Path("figures"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    methods = [str(m).lower() for m in args.methods]

    cfg = RunConfig(
        etas=tuple(float(e) for e in args.etas),
        n_draws=int(args.n_draws),
        n_adv=int(args.n_adv),
        seed=int(args.seed),
        pd_tol=float(args.pd_tol),
        eig_cut_rel=float(args.eig_cut_rel),
        eig_cut_abs=float(args.eig_cut_abs),
        min_rank=int(args.min_rank),
        kappa_factor=float(args.kappa_factor),
        eta_floor=float(args.eta_floor),
        kappa_cap_unpert=float(args.kappa_cap_unpert),
        kappa_cap_fixed=None if args.kappa_cap_fixed is None else float(args.kappa_cap_fixed),
        sdiag_floor_rel=float(args.sdiag_floor_rel),
        h0_factor=float(args.h0_factor),
        s0_factor=float(args.s0_factor),
        h0_override=None if args.h0 is None else float(args.h0),
        s0_override=None if args.s0 is None else float(args.s0),
        clip_plot_quantile=float(args.clip_plot_quantile),
        rank_plot_mode=str(args.rank_plot_mode),
    )

    method_data: Dict[str, MethodData] = {}
    for method in methods:
        try:
            method_data[method] = load_method_data(data_root, method)
        except Exception as exc:
            print(f"[warn] skipping method '{method}': {exc}")

    if not method_data:
        raise RuntimeError(f"No usable method data found under {data_root}")

    available_union = sorted(set().union(*(set(md.available_ks()) for md in method_data.values())))
    if not available_union:
        raise RuntimeError("No K values found in loaded data")

    detected_kmax = int(max(available_union))
    chosen_kmax = detected_kmax if args.Kmax is None else int(args.Kmax)

    baselines = build_unperturbed_baselines(method_data, methods, chosen_kmax, cfg)

    if args.E_ref is not None:
        e_ref: Optional[float] = float(args.E_ref)
    else:
        e_ref = load_optional_e_ref(data_root)

    requested_k_window = [int(k) for k in args.K_window]
    chosen_k_window = [k for k in requested_k_window if k <= chosen_kmax and any(md.has_k(k) for md in method_data.values())]
    if not chosen_k_window:
        raise RuntimeError(
            f"No requested K_window values are available. requested={requested_k_window}, Kmax={chosen_kmax}"
        )

    records = run_window_sensitivity(
        method_data=method_data,
        methods=methods,
        k_window=chosen_k_window,
        cfg=cfg,
        e_ref=e_ref,
    )

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "h6_window_sensitivity_fixed_drift_summary.csv"
    write_summary_csv(records, csv_path)

    conv_png = figures_dir / "h6_unperturbed_convergence.png"
    conv_pdf = figures_dir / "h6_unperturbed_convergence.pdf"
    plot_unperturbed_convergence(
        baselines=baselines,
        methods=methods,
        k_window=chosen_k_window,
        kmax=chosen_kmax,
        e_ref_input=e_ref,
        out_png=conv_png,
        out_pdf=conv_pdf,
        dpi=int(args.dpi),
    )

    k_tag = "_".join(str(k) for k in requested_k_window)
    window_png = figures_dir / f"h6_window_sensitivity_fixed_drift_K{k_tag}.png"
    window_pdf = figures_dir / f"h6_window_sensitivity_fixed_drift_K{k_tag}.pdf"
    plot_window_sensitivity(
        records=records,
        methods=methods,
        k_window=chosen_k_window,
        out_png=window_png,
        out_pdf=window_pdf,
        clip_plot_quantile=cfg.clip_plot_quantile,
        rank_plot_mode=cfg.rank_plot_mode,
        dpi=int(args.dpi),
    )

    print(f"Detected Kmax: {detected_kmax}")
    print(f"Chosen Kmax: {chosen_kmax}")
    print(f"Chosen K_window: {chosen_k_window}")
    print(
        "Parameters: "
        f"kappa_factor={cfg.kappa_factor}, pd_tol={cfg.pd_tol}, "
        f"eig_cut_rel={cfg.eig_cut_rel}, eig_cut_abs={cfg.eig_cut_abs}, "
        f"sdiag_floor_rel={cfg.sdiag_floor_rel}, "
        f"kappa_cap_unpert={cfg.kappa_cap_unpert}, "
        f"kappa_cap_fixed={cfg.kappa_cap_fixed}"
    )
    print_sanity(records, methods, chosen_k_window)
    print(f"Saved convergence figure: {conv_png} and {conv_pdf}")
    print(f"Saved window figure: {window_png} and {window_pdf}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
