"""Build kaggle_kernel_v6_regularized_robust_ensemble.ipynb from V5 template.

V6 = V5 residual-anchor framework + regularized ensemble + shuffled CV +
     CatBoost diversity + clean post-processing (no SavGol, no w_pf).

Run: python scripts/_build_v6_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.strip("\n").splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.strip("\n").splitlines(keepends=True),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "kaggle_kernel_v6_regularized_robust_ensemble.ipynb"

    # Read the V5 notebook as base
    v5_path = repo / "kaggle_kernel_v5_lgb_xgb_residual.ipynb"
    v5 = json.loads(v5_path.read_text(encoding="utf-8"))

    # We'll build cells manually using V5 cells as reference,
    # patching only the cells that differ in V6.

    # Cell index map from V5 -> our new content.
    # Most cells stay identical. We replace: config, imports, training, stacking, post-processing, submission, notes.

    cells = []

    # ── Cell 0: Header (REPLACED) ──
    cells.append(md("""\
# v6 Regularized Robust Ensemble (LightGBM + XGBoost + CatBoost)

This notebook extends the v5 residual-anchor framework with anti-overfitting
measures to improve hidden-leaderboard robustness while targeting sub-9
Public LB.

Key improvements over v5:

1. **Shuffled GroupKFold** — unique well IDs are permuted before fold
   assignment to break geographic/temporal clustering inside folds.
2. **Stronger LGBM regularization** — `num_leaves=127`, `reg_lambda=5.0`,
   `reg_alpha=0.1`, unified `learning_rate=0.035`.
3. **CatBoost diversity** — one CatBoost model added for ensemble diversity
   (oblivious trees vs histogram trees).
4. **Refined features** — keep X, Y, Z for macro geology; add dx, dy, dz, dxy
   for local trajectory; add ktvt_range, ktvt_std, gr_vs_slp; remove ease_frac.
5. **Clean post-processing** — only alpha * tau grid search (no w_pf blending,
   no Savitzky-Golay smoothing).

What v6 deliberately omits:

- Exact (X, Y, Z) coordinate replacement — label leakage.
- Savitzky-Golay smoothing — introduced initial-row bias (v10 confirmed).
- w_pf post-processing blend — extra OOF tuning dimension -> overfitting risk.
- TabICL / cal-zone augmentation / hill-climb stacking — complexity without
  proven stable OOF gain.

The notebook expects Kaggle competition input under
`/kaggle/input/rogii-wellbore-geology-prediction/` and writes
`/kaggle/working/submission.csv`. It also runs locally if the repo data layout
is present in the current working directory."""))

    # ── Cell 1: Config section header ──
    cells.append(md("## Config and Kaggle environment"))

    # ── Cell 2: Config (REPLACED with V6 params) ──
    cells.append(code("""\
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Numba and CatBoost are required. Install on Kaggle if missing.
for pkg in ["numba", "catboost"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                       check=False)

# Numba cache directory. On Kaggle we want it on /kaggle/working; locally use
# a temp dir under the repo so we never pollute the home directory.
_kaggle_working = Path("/kaggle/working")
if _kaggle_working.is_dir():
    _numba_cache = _kaggle_working / ".numba"
else:
    _numba_cache = Path.cwd() / ".numba_cache"
_numba_cache.mkdir(parents=True, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = str(_numba_cache)

SEED = 42
N_SPLITS = 5

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]

PLANE_K = 10        # Formation plane KNN neighbors
DENSE_SPW = 60      # Dense imputer samples per well
DENSE_K = 20        # Dense imputer neighbors

import numpy as np

ANCH_OFFS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], np.float32)
BEAM_OFFS = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], np.float32)
SC_OFFS   = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
PF_OFFS   = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)

# (beam_size, move_cost, emit_scale, smooth_radius, tag)
BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),
    (10,  8.0,  64.0, 2, "loose"),
    ( 8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0,  90.0, 5, "sm5"),
    (20,  4.0,  36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]

# Particle-filter constants (matching the 9.830 reference exactly).
PF_N = 600
ANCC_N = 600
PF_MOM = 0.993
PF_VN = 0.005
PF_PN = 0.01
PF_IV = 0.02
PF_IS = 0.5
PF_RESAMP = 0.5
PF_RP = 0.2
PF_RV = 0.003
PF_GW = 5
PF_GWT = 0.3
ANCC_A = 0.998
ANCC_RN = 0.002
ANCC_PN = 0.005
ANCC_IR = 0.01
ANCC_IS = 0.3
ANCC_RP = 0.1
ANCC_RR = 0.001
PF_GS_MIN = 10.0
PF_GS_MAX = 60.0
PF_GS_DEF = 30.0

# ── V6: Stronger-regularized LightGBM ──
# num_leaves 255->127 (less capacity per tree)
# reg_lambda 3.0->5.0, reg_alpha 0.05->0.1 (stronger L1/L2)
# min_child_samples 15->20
# unified learning_rate, n_estimators 8000->6000
LGB_BASE = dict(
    boosting_type="gbdt",
    num_leaves=127,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.1,
    objective="regression",
    verbose=-1,
    n_jobs=-1,
    max_bin=255,
    force_row_wise=True,
    device="gpu",
)
LGB_CONFIGS = [
    dict(learning_rate=0.035, n_estimators=6000, random_state=42),
    dict(learning_rate=0.035, n_estimators=6000, random_state=7),
    dict(learning_rate=0.035, n_estimators=6000, random_state=123),
]

# XGBoost params. Same GPU->CPU fallback pattern as v5.
XGB_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.025,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=2.0,
    min_child_weight=10,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    device="cuda",
    early_stopping_rounds=200,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

# ── V6: CatBoost (NEW) ──
# depth=8, l2_leaf_reg=3.0 for modest capacity + regularization.
# Single seed (42) — CatBoost is for ensemble diversity, not multi-seed tuning.
CB_PARAMS = dict(
    iterations=4000,
    learning_rate=0.035,
    depth=8,
    l2_leaf_reg=3.0,
    min_data_in_leaf=20,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    task_type="GPU",
    devices="0",
    od_type="Iter",
    od_wait=150,
    verbose=0,
)

# Data path detection: Kaggle competition path first, then recursive fallback,
# then the current working directory (local repo run).
def _find_data_dir() -> Path:
    explicit = [
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path.cwd() / "data",
        Path.cwd(),
    ]
    for p in explicit:
        if (p / "train").is_dir() and (p / "test").is_dir() and (p / "sample_submission.csv").is_file():
            return p
    root = Path("/kaggle/input")
    if root.exists():
        for sample in root.glob("**/sample_submission.csv"):
            cand = sample.parent
            if (cand / "train").is_dir() and (cand / "test").is_dir():
                return cand
    raise FileNotFoundError("ROGII competition data not found.")

DATA = _find_data_dir()
TRAIN_DIR = DATA / "train"
TEST_DIR = DATA / "test"
SAMPLE = DATA / "sample_submission.csv"

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "submission.csv"

print(f"DATA   = {DATA}")
print(f"OUT    = {OUT}")
print(f"#train wells: {len(list(TRAIN_DIR.glob('*__horizontal_well.csv')))}")
print(f"#test  wells: {len(list(TEST_DIR.glob('*__horizontal_well.csv')))}")
"""))

    # ── Cell 3: Imports section header ──
    cells.append(md("## Imports"))

    # ── Cell 4: Imports (REPLACED: +catboost) ──
    cells.append(code("""\
import gc
import multiprocessing
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from joblib import Parallel, delayed
from numba import njit
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

np.random.seed(SEED)
NCPU = min(4, multiprocessing.cpu_count())
print(f"NCPU={NCPU}")
"""))

    # ── Cell 5: JIT section header ──
    cells.append(md("## Numba JIT kernels (beam search + particle filters)"))

    # ── Cell 6: JIT kernels (IDENTICAL to V5) ──
    cells.append(code("""\
@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N):
        cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0.0, 1.0 / N)
    np2 = np.empty(N)
    na = np.empty(N)
    ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u:
            ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na


@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr)
    nt = len(tw_gr)
    MAX = BS * 6
    bidx = np.zeros(BS, np.int64)
    bidx[0] = si
    bcost = np.full(BS, 1e30)
    bcost[0] = 0.0
    bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64)
    hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64)
    cC = np.full(MAX, 1e30)
    cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]
        nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]
            cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt:
                    continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni:
                        fnd = ci
                        break
                if fnd >= 0:
                    if tot < cC[fnd]:
                        cC[fnd] = tot
                        cP[fnd] = bi
                else:
                    if nc < MAX:
                        cI[nc] = ni
                        cC[nc] = tot
                        cP[nc] = bi
                        nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]:
                    mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]
        hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]
        bcost[:kept] = cC[:kept]
        bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]:
            best = b
    path = np.zeros(n, np.int64)
    b = best
    for s in range(n - 1, -1, -1):
        path[s] = hI[s, b]
        b = hP[s, b]
    return path


@njit(cache=True)
def _pf_ancc_jit(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N,
                  ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N)
    rate = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0
    for i in range(len(md_v)):
        dm = max(md_v[i] - pm, 1.0)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tv = pos[j] - z_v[i]
            tv = max(tv, vmin - 50.0)
            tv = min(tv, vmin + len(gg) * step + 50.0)
            pos[j] = tv + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600.0 else 0.0, 1e-300)
                w[j] *= lk
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N):
                w[j] = 1.0 / N
        tv = 0.0
        for j in range(N):
            tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv
        va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va ** 0.5
        pm = md_v[i]
    return pts, std_


@njit(cache=True)
def _pf_z_jit(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv,
               beta, icpt, zsig, N, MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    pos = np.empty(N)
    vel = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ip + 0.5 * np.random.randn()
        vel[j] = iv + 0.02 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0
    pz = z_v[0] - 1.0
    for i in range(len(md_v)):
        dm = max(md_v[i] - pm, 1.0)
        dzd = (z_v[i] - pz) / dm
        ve = beta * dzd + icpt
        for j in range(N):
            vel[j] = MOM * vel[j] + VN * np.random.randn()
            pos[j] += vel[j] * dm + PN * np.random.randn()
            pos[j] = max(pos[j], vmin - 50.0)
            pos[j] = min(pos[j], vmin + len(gg_p) * step + 50.0)
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step)
                dp = (gr_v[i] - ep) / gs
                lp = max(np.exp(-0.5 * dp * dp) if dp * dp < 600.0 else 0.0, 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(gg_s, pos[j], vmin, step)
                    ds = (gr_sm_v[i] - es) / (gs * 1.5)
                    ls = max(np.exp(-0.5 * ds * ds) if ds * ds < 600.0 else 0.0, 1e-300)
                    lk = (1.0 - GR_WT) * lp + GR_WT * ls
                else:
                    lk = lp
                lk = max(lk, 1e-300)
                w[j] *= lk
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ws2 = 0.0
        for j in range(N):
            dv = (vel[j] - ve) / max(zsig * 2.0, 0.005)
            lz = max(np.exp(-0.5 * dv * dv) if dv * dv < 600.0 else 0.0, 1e-300)
            w[j] *= lz
            ws2 += w[j]
        if ws2 > 0.0:
            for j in range(N):
                w[j] /= ws2
        else:
            for j in range(N):
                w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N):
                w[j] = 1.0 / N
        wm = 0.0
        for j in range(N):
            wm += w[j] * pos[j]
        pts[i] = wm
        va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - wm) ** 2
        std_[i] = va ** 0.5
        pm = md_v[i]
        pz = z_v[i]
    return pts, std_


def _make_grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min())
    tmax = float(tw_tvt.max())
    g = np.arange(tmin, tmax + step, step)
    return np.interp(g, tw_tvt, tw_gr).astype(np.float64), tmin, step


print("Compiling Numba JIT kernels...")
_dummy_gr = np.ones(10, np.float64)
_dummy_tw = np.ones(20, np.float64)
_dummy_gg, _dm, _ds = _make_grid(np.arange(20, dtype=np.float64), _dummy_tw)
_beam_jit(_dummy_gr, _dummy_tw, 5, 5, 10.0, 100.0)
_pf_ancc_jit(np.ones(5), np.zeros(5), np.ones(5), _dummy_gg, _dm, _ds, 30.0,
             0.0, 0.0, 10, ANCC_A, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, 0.5)
print("Numba JIT ready.")
"""))

    # ── Cell 7: Physics wrappers section header ──
    cells.append(md("## Physics wrappers"))

    # ── Cell 8: Physics wrappers (IDENTICAL to V5) ──
    cells.append(code("""\
def beam_search(hgr, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    sgr = pd.Series(hgr.astype(np.float64)).interpolate(limit_direction="both").fillna(
        float(np.nanmean(tw_gr))
    )
    if r > 0:
        sgr = sgr.rolling(r * 2 + 1, center=True, min_periods=1).mean()
    sgr = sgr.to_numpy(np.float64)
    si = int(np.searchsorted(tw_tvt, start_tvt))
    si = max(0, min(si, len(tw_tvt) - 1))
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, bs, mc, es)
    return tw_tvt[path.clip(0, len(tw_tvt) - 1)].astype(np.float32)


def run_pf_ancc(hw, tw_tvt, tw_gr):
    gg, vmin, step = _make_grid(tw_tvt, tw_gr)
    k = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    k2 = k[k["GR"].notna()]
    gs = PF_GS_DEF
    if len(k2) >= 20:
        gs = float(np.clip(
            np.std(k2["GR"].values - np.interp(k2["TVT_input"].values, tw_tvt, tw_gr)),
            PF_GS_MIN, PF_GS_MAX,
        ))
    ls = float(k["TVT_input"].iloc[-1]) + float(k["Z"].iloc[-1])
    t = k.tail(30)
    ir = 0.0
    if len(t) >= 10:
        dt = np.diff(t["TVT_input"].values)
        dz = np.diff(t["Z"].values)
        dm = np.diff(t["MD"].values)
        m = dm > 0
        if m.sum() >= 3:
            ir = float(np.median((dt[m] + dz[m]) / dm[m]))
    gr_v = ev["GR"].to_numpy(np.float64)
    pts, std_ = _pf_ancc_jit(
        ev["MD"].to_numpy(np.float64),
        ev["Z"].to_numpy(np.float64),
        gr_v,
        gg, vmin, step, gs, ls, ir, ANCC_N,
        ANCC_A, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, 0.5,
    )
    return pts, std_


def run_pf_z(hw, tw_tvt, tw_gr):
    gg_p, vmin, step = _make_grid(tw_tvt, tw_gr)
    tw_sm = pd.Series(tw_gr).rolling(PF_GW, center=True, min_periods=1).mean().values
    gg_s, _, _ = _make_grid(tw_tvt, tw_sm)
    k = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    k2 = k[k["GR"].notna()]
    gs = PF_GS_DEF
    if len(k2) >= 20:
        gs = float(np.clip(
            np.std(k2["GR"].values - np.interp(k2["TVT_input"].values, tw_tvt, tw_gr)),
            PF_GS_MIN, PF_GS_MAX,
        ))
    ktvt = k["TVT_input"].values
    kmd = k["MD"].values
    kz = k["Z"].values
    dz = np.diff(kz)
    dtvt = np.diff(ktvt)
    dmd_ = np.diff(kmd)
    m = dmd_ > 0
    beta, intc, zsig = -1.0, 0.0, 0.1
    if m.sum() >= 10:
        vz = dz[m] / dmd_[m]
        vt = dtvt[m] / dmd_[m]
        c, _, _, _ = np.linalg.lstsq(np.column_stack([vz, np.ones_like(vz)]), vt, rcond=None)
        beta, intc = float(c[0]), float(c[1])
        zsig = max(float(np.std(vt - (c[0] * vz + c[1]))), 0.001)
    iv = 0.0
    if len(k) >= 10:
        t = k.tail(20)
        dt2 = np.diff(t["TVT_input"].values)
        dm2 = np.diff(t["MD"].values)
        m2 = dm2 > 0
        if m2.sum() >= 3:
            iv = float(np.median(dt2[m2] / dm2[m2]))
    ip = float(k["TVT_input"].iloc[-1])
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(
        float(np.nanmean(tw_gr))
    )
    hw_gsm = gr_full.rolling(PF_GW, center=True, min_periods=1).mean()
    gr_v = ev["GR"].to_numpy(np.float64)
    gr_sm_v = hw_gsm.iloc[ev.index].to_numpy(np.float64)
    pts, std_ = _pf_z_jit(
        ev["MD"].to_numpy(np.float64),
        ev["Z"].to_numpy(np.float64),
        gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv,
        beta, intc, zsig, PF_N, PF_MOM, PF_VN, PF_PN, PF_GWT, PF_RP, PF_RV, 0.5,
    )
    return pts, std_
"""))

    # ── Cell 9: Imputers section header ──
    cells.append(md("## Formation plane and dense ANCC imputers"))

    # ── Cell 10: Imputers (IDENTICAL to V5) ──
    cells.append(code("""\
class FormationPlaneKNN:
    def __init__(self, wids, data_dir):
        rows = []
        for wid in wids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in FORMATIONS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df["wid"])}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df["x"].to_numpy()
        self.ya = self.df["y"].to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q / self.scale
        nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xa[ik]
        yn = self.ya[ik]
        fn = self.fa[ik]
        wx = w * xn
        wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1)
        A[:, 0, 1] = (wx * yn).sum(1)
        A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]
        A[:, 1, 1] = (wy * yn).sum(1)
        A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]
        A[:, 2, 1] = A[:, 1, 2]
        A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9
        A[:, 1, 1] += 1e-9
        A[:, 2, 2] += 1e-9
        rhs = np.stack([
            (wx[:, :, None] * fn).sum(1),
            (wy[:, :, None] * fn).sum(1),
            (w[:, :, None] * fn).sum(1),
        ], 1)
        try:
            coef = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception:
                    pass
        Xq = xy_q[:, 0]
        Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)


class DenseANCCImputer:
    def __init__(self, wids, data_dir, spw=DENSE_SPW):
        xs, ys, anccs, wids_ = [], [], [], []
        for wid in wids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            s = df.iloc[ix]
            xs.append(s["X"].values)
            ys.append(s["Y"].values)
            anccs.append(s["ANCC"].values)
            wids_.extend([wid] * len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32)
        self.wids = np.array(wids_)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q = np.atleast_2d(xy_q)
        q = xy_q / self.scale
        nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0)
        sw = w.sum(1)
        safe = np.where(sw < 1e-9, 1.0, sw)
        an = self.ancc[ik]
        ap = (an * w).sum(1) / safe
        ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((an - ap[:, None]) ** 2 * w).sum(1) / safe
        return (
            ap.astype(np.float32),
            np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
            np.where(vk, dk, np.inf).min(1).astype(np.float32),
        )


_t0 = time.time()
hw_paths = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
train_wids = [p.stem.replace("__horizontal_well", "") for p in hw_paths]
print(f"Building imputers ({len(train_wids)} wells)...")
FI = FormationPlaneKNN(train_wids, TRAIN_DIR)
DI = DenseANCCImputer(train_wids, TRAIN_DIR)
print(f"  FPK rows={len(FI.df)} | Dense rows={len(DI.ancc):,}  ({time.time() - _t0:.0f}s)")
"""))

    # ── Cell 11: Feature builder section header ──
    cells.append(md("## Per-well feature builder"))

    # ── Cell 12: Feature builder (MODIFIED: V6 feature changes) ──
    cells.append(code("""\
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def robust_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6:
        return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


def affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1.0, float(np.nanmean(kgr[v]) - np.nanmean(tw_at_k[v])) if v.any() else 0.0
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)


def seg_b_well(ktvt, kz, form_col):
    """Five calibration offsets: full, early-third, mid-third, late-50, WLS."""
    bv = ktvt + kz - form_col
    n = len(bv)
    b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n - 50):])) if n >= 5 else b_full
    t1, t2 = n // 3, 2 * n // 3
    b_early = float(np.median(bv[:max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1:max(t1 + 1, t2)])) if t2 > t1 else b_full
    w = np.exp(0.02 * np.arange(n))
    w /= w.sum()
    b_wls = float(np.dot(w, bv))
    return b_full, b_early, b_mid, b_late, b_wls


def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw_sc in hws:
        win = 2 * hw_sc + 1
        nk = len(kgr)
        nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        if len(sts) == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw_sc, mode="edge")
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw_sc, 0, nk - 1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1)
    scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3.0 * scores)
    sw /= sw.sum(1, keepdims=True) + 1e-9
    sc_ens = (tvts * sw).sum(1).astype(np.float32)
    return out, sc_ens


_FI = FI
_DI = DI


def build_well(hw_path, tw_path, is_train):
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path).sort_values("TVT")
    except Exception:
        return None
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    if is_train and ("TVT" not in hw.columns or hw["TVT"].isna().all()):
        return None
    tw_tvt = tw["TVT"].to_numpy(np.float32)
    tw_gr = tw["GR"].to_numpy(np.float32)
    if len(tw_tvt) < 3:
        return None

    np.random.seed(SEED)
    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0:
        return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)
    pf_use = pf_a.astype(np.float32)
    std_use = std_a.astype(np.float32)
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))

    lk = kn.iloc[-1]
    last_tvt = float(lk["TVT_input"])
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(
        float(np.nanmean(tw_gr))
    )
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr = gr_full.iloc[:len(kn)].to_numpy(np.float32)

    # Seven beam searches.
    bpaths = {}
    for (bs, mc, es, r, tag) in BEAMS:
        bpaths[tag] = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
    beam_ref = (bpaths["cons"] + bpaths["sm5"]) / 2.0

    # Multi-scale NCC + softmax ensemble.
    ktvt = kn["TVT_input"].to_numpy(np.float32)
    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, sc8s = sc_res[0]
    sc15, sc15s = sc_res[1]
    sc25, sc25s = sc_res[2]
    sc_cons = (sc8 + sc15 + sc25) / 3.0
    sc_trust = float(np.clip(len(kn) / 200.0, 0.0, 0.6))
    hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * sc_ens

    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
    a_cal, b_cal = affine_cal(kgr, tw_at_k)
    kmd = kn["MD"].to_numpy(np.float32)
    kz_local = kn["Z"].to_numpy(np.float32)
    pfx_rmse = float(np.sqrt(np.mean((kgr - tw_at_k) ** 2)))
    slp_all = robust_slope(kmd, ktvt)
    slp_50 = robust_slope(kmd[-50:], ktvt[-50:])
    slp_z = robust_slope(kn["Z"].to_numpy(), ktvt)

    swid = wid if is_train else None
    xy_ev = ev[["X", "Y"]].to_numpy(np.float64)
    xy_kn = kn[["X", "Y"]].to_numpy(np.float64)
    form_ev, knn_d = _FI.impute(xy_ev, self_wid=swid)
    form_kn, _ = _FI.impute(xy_kn, self_wid=swid)
    z_kn = kn["Z"].to_numpy(np.float32)
    z_ev = ev["Z"].to_numpy(np.float32)

    tvt_fs = {}
    form_rmse = {}
    form_list = []
    for fi2, fn in enumerate(FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, z_kn, form_kn[:, fi2])
        tvt_fs[f"tvtF_{fn}"] = (-z_ev + form_ev[:, fi2] + b_full).astype(np.float32)
        tvt_fs[f"tvtFw_{fn}"] = (-z_ev + form_ev[:, fi2] + b_wls).astype(np.float32)
        tvt_fs[f"tvtF50_{fn}"] = (-z_ev + form_ev[:, fi2] + b_late).astype(np.float32)
        tvt_fs[f"bw_{fn}"] = np.float32(b_full)
        tvt_fs[f"bww_{fn}"] = np.float32(b_wls)
        tvt_fs[f"bw50_{fn}"] = np.float32(b_late)
        tvt_fs[f"bw_early_{fn}"] = np.float32(b_early)
        tvt_fs[f"bw_mid_{fn}"] = np.float32(b_mid)
        form_rmse[fn] = float(np.sqrt(np.mean((ktvt - (-z_kn + form_kn[:, fi2] + b_full)) ** 2)))
        form_list.append(tvt_fs[f"tvtF_{fn}"])

    fs = np.stack(form_list, 1)
    form_mean_d = (fs.mean(1) - last_tvt).astype(np.float32)
    form_std_d = fs.std(1).astype(np.float32)
    form_rng_d = (fs.max(1) - fs.min(1)).astype(np.float32)

    d_ancc, d_std, d_dist = _DI.impute(xy_ev, self_wid=swid)
    d_kn, d_std_kn, _ = _DI.impute(xy_kn, self_wid=swid)
    b_vd = ktvt + z_kn - d_kn
    _, b_de, b_dm, b_dl, b_dw = seg_b_well(ktvt, z_kn, d_kn)
    b_d = float(np.median(b_vd))
    tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)
    tvt_densew = (-z_ev + d_ancc + b_dw).astype(np.float32)
    tvt_dense50 = (-z_ev + d_ancc + b_dl).astype(np.float32)
    d_rmse = float(np.sqrt(np.mean((ktvt + z_kn - d_kn - b_d) ** 2)))
    d_bias = float(np.mean(b_vd - b_d))

    all_sigs = [pf_use] + [p for p in bpaths.values()] + [sc8, sc15, sc25, sc_ens,
                                                          tvt_fs["tvtF_ANCC"], tvt_dense]
    sig_mat = np.stack(all_sigs, 1)
    sig_std = sig_mat.std(1).astype(np.float32)
    sig_mean = (sig_mat.mean(1) - last_tvt).astype(np.float32)

    gr_s = pd.Series(gr_full.values)
    rolls = {}
    for w_size in [5, 21, 51, 101]:
        r = gr_s.rolling(w_size, center=True, min_periods=1)
        rolls[f"grm{w_size}"] = r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f"grs{w_size}"] = r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1, 5, 15, 30]:
        rolls[f"glag{lag}"] = gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f"glead{lag}"] = gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1 = gr_s.diff().fillna(0.0).iloc[ev.index].values.astype(np.float32)
    gr_d2 = gr_s.diff().diff().fillna(0.0).iloc[ev.index].values.astype(np.float32)
    gr_env = gr_s.rolling(21, center=True, min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg = np.sqrt(np.maximum(
        (gr_s ** 2).rolling(21, center=True, min_periods=1).mean(), 0.0
    )).iloc[ev.index].values.astype(np.float32)

    hmd = ev["MD"].to_numpy(np.float32)
    md_since = hmd - float(lk["MD"])
    slp_b_all = (last_tvt + slp_all * md_since).astype(np.float32)
    slp_b_50 = (last_tvt + slp_50 * md_since).astype(np.float32)
    mdd = hw["MD"].diff().replace(0, np.nan)
    dzdmd = (hw["Z"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd = (hw["X"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dydmd = (hw["Y"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    nh = len(ev)
    frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)

    # ── V6: Relative trajectory features ──
    last_x = float(lk["X"])
    last_y = float(lk["Y"])
    last_z = float(lk["Z"])
    dx = (ev["X"].to_numpy(np.float32) - last_x).astype(np.float32)
    dy = (ev["Y"].to_numpy(np.float32) - last_y).astype(np.float32)
    dz = (z_ev - last_z).astype(np.float32)
    dxy = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)

    # ── V6: Known-zone TVT statistics ──
    ktvt_range = float(np.ptp(ktvt))
    ktvt_std = float(ktvt.std())

    # ── V6: GR vs slope baseline ──
    gr_vs_slp = hgr - np.interp(slp_b_all, tw_tvt, tw_gr).astype(np.float32)

    def sc(v):
        return np.full(nh, np.float32(v), np.float32)

    feats = {
        "well": wid,
        "id": [f"{wid}_{i}" for i in ev.index],
        "last_known_tvt": sc(last_tvt),
        "pf_ancc": pf_use,
        "pf_ancc_std": std_use,
        "pf_ancc_delta": (pf_use - last_tvt).astype(np.float32),
        "pf_z": (pf_z.astype(np.float32) if has_z else sc(last_tvt)),
        "pf_z_delta": ((pf_z - last_tvt).astype(np.float32) if has_z else sc(0.0)),
        "pf_vs_z": ((pf_use - pf_z.astype(np.float32)) if has_z else sc(0.0)),
        **{f"beam_{t}_d": (p - np.float32(last_tvt)).astype(np.float32) for t, p in bpaths.items()},
        "beam_mean_d": np.stack([(p - last_tvt) for p in bpaths.values()], 1).mean(1).astype(np.float32),
        "beam_std_d": np.stack([(p - last_tvt) for p in bpaths.values()], 1).std(1).astype(np.float32),
        "beam_med_d": np.median(np.stack([(p - last_tvt) for p in bpaths.values()], 1), 1).astype(np.float32),
        "sc8_d": (sc8 - np.float32(last_tvt)).astype(np.float32),
        "sc8_sc": sc8s,
        "sc15_d": (sc15 - np.float32(last_tvt)).astype(np.float32),
        "sc15_sc": sc15s,
        "sc25_d": (sc25 - np.float32(last_tvt)).astype(np.float32),
        "sc25_sc": sc25s,
        "sc_cons_d": (sc_cons - np.float32(last_tvt)).astype(np.float32),
        "sc_ens_d": (sc_ens - np.float32(last_tvt)).astype(np.float32),
        "sc_trust": sc(sc_trust),
        "hyb_d": (hyb_ref - np.float32(last_tvt)).astype(np.float32),
        "sig_std": sig_std,
        "sig_mean_d": sig_mean,
        **tvt_fs,
        **{f"frm_rmse_{fn}": sc(form_rmse[fn]) for fn in FORMATIONS},
        "form_mean_d": form_mean_d,
        "form_std_d": form_std_d,
        "form_rng_d": form_rng_d,
        "spatial_knn_dist": knn_d,
        "dense_ancc": d_ancc,
        "dense_std": d_std,
        "dense_dist": d_dist,
        "tvt_dense_d": (tvt_dense - last_tvt).astype(np.float32),
        "tvt_densew_d": (tvt_densew - last_tvt).astype(np.float32),
        "tvt_dense50_d": (tvt_dense50 - last_tvt).astype(np.float32),
        "dense_rmse": sc(d_rmse),
        "dense_bias": sc(d_bias),
        "pf_vs_spatial": (pf_use - tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "pf_vs_dense": (pf_use - tvt_dense).astype(np.float32),
        "spatial_vs_dense": (tvt_fs["tvtF_ANCC"] - tvt_dense).astype(np.float32),
        "beam_vs_spatial": (bpaths["cons"] - tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "sc_vs_beam": (sc_ens - bpaths["cons"]).astype(np.float32),
        "cal_a": sc(a_cal),
        "cal_b": sc(b_cal),
        "pfx_rmse": sc(pfx_rmse),
        "known_len": sc(len(kn)),
        "eval_len": sc(nh),
        "slp_all": sc(slp_all),
        "slp_50": sc(slp_50),
        "slp_z": sc(slp_z),
        "slp_b_d_all": (slp_b_all - last_tvt).astype(np.float32),
        "slp_b_d_50": (slp_b_50 - last_tvt).astype(np.float32),
        "dzdmd": dzdmd,
        "dxdmd": dxdmd,
        "dydmd": dydmd,
        "md_since": md_since,
        "frac": frac,
        "frac2": frac ** 2,
        # ── V6: absolute coords (kept) + relative trajectory (new) ──
        "z": z_ev,
        "x": ev["X"].to_numpy(np.float32),
        "y": ev["Y"].to_numpy(np.float32),
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "dxy": dxy,
        # ── V6: known-zone statistics (new) ──
        "ktvt_range": sc(ktvt_range),
        "ktvt_std": sc(ktvt_std),
        **rolls,
        "gr_d1": gr_d1,
        "gr_d2": gr_d2,
        "gr_env": gr_env,
        "gr_nrg": gr_nrg,
        "gr_minus_tw_last": (
            gr_full.iloc[ev.index].values.astype(np.float32)
            - float(np.interp(last_tvt, tw_tvt, tw_gr))
        ).astype(np.float32),
        # ── V6: GR vs slope baseline (new) ──
        "gr_vs_slp": gr_vs_slp,
        "anchor_t_pos": sc(float(
            (last_tvt - float(tw_tvt.min())) / max(float(tw_tvt.max() - tw_tvt.min()), 1e-3)
        )),
        "tw_tvt_range": sc(float(tw_tvt.max() - tw_tvt.min())),
        "tw_gr_mean": sc(float(tw_gr.mean())),
        "tw_gr_std": sc(float(tw_gr.std())),
    }
    hgr_fill = gr_full.iloc[ev.index].values.astype(np.float32)
    for o in ANCH_OFFS:
        feats[f"anch_diff_{int(o)}"] = hgr_fill - float(np.interp(last_tvt + float(o), tw_tvt, tw_gr))
    for o in BEAM_OFFS:
        feats[f"beam_diff_{int(o)}"] = hgr_fill - np.interp(bpaths["cons"] + float(o), tw_tvt, tw_gr).astype(np.float32)
    for o in SC_OFFS:
        feats[f"sc_diff_{int(o)}"] = hgr_fill - np.interp(sc_ens + float(o), tw_tvt, tw_gr).astype(np.float32)
    for o in PF_OFFS:
        feats[f"pf_diff_{int(o)}"] = hgr_fill - np.interp(pf_use + float(o), tw_tvt, tw_gr).astype(np.float32)
    if is_train:
        feats["target"] = (ev["TVT"].to_numpy(np.float32) - np.float32(last_tvt))
    df = pd.DataFrame(feats)
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype(np.float32)
    return df


def build_dataset(paths, is_train, label):
    tw_dir = TRAIN_DIR if is_train else TEST_DIR
    print(f"  {label}: {len(paths)} wells | {NCPU} threads")
    t_start = time.time()
    results = Parallel(n_jobs=NCPU, backend="threading", verbose=5)(
        delayed(build_well)(
            str(p),
            str(tw_dir / p.name.replace("__horizontal_well.csv", "__typewell.csv")),
            is_train,
        )
        for p in paths
    )
    ok = [r for r in results if r is not None]
    print(f"  {label}: OK={len(ok)} skipped={len(paths) - len(ok)} | {time.time() - t_start:.0f}s")
    return pd.concat(ok, ignore_index=True)
"""))

    # ── Cell 13: Build section header ──
    cells.append(md("## Build train and test feature tables"))

    # ── Cell 14: Build tables (IDENTICAL to V5) ──
    cells.append(code("""\
print("Building train features...")
_t_build = time.time()
train_df = build_dataset(hw_paths, is_train=True, label="train")
print(f"train: {train_df.shape}  ({time.time() - _t_build:.0f}s)")

test_paths = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
print("Building test features...")
test_df = build_dataset(test_paths, is_train=False, label="test")
print(f"test: {test_df.shape}")

SKIP = {"well", "id", "target"}
feature_cols = [c for c in train_df.columns if c not in SKIP]
print(f"#features: {len(feature_cols)}")

X = train_df[feature_cols].astype(np.float32)
y = train_df["target"]
g = train_df["well"]
Xt = test_df[feature_cols].astype(np.float32)
gc.collect()
"""))

    # ── Cell 15: Training section header (MODIFIED) ──
    cells.append(md("## Shuffled GroupKFold training (3 LGB + 1 XGB + 1 CatBoost)"))

    # ── Cell 16: Training (MODIFIED: Shuffled CV + CatBoost) ──
    cells.append(code("""\
# ── V6: Shuffled GroupKFold ──
# Plain GroupKFold assigns folds in sorted well-id order, which clusters
# wells geographically inside each fold -> inflated fold variance.
# Shuffling unique well IDs before modulo assignment breaks this correlation.
_unique_wells = np.unique(g.values)
_rng_cv = np.random.RandomState(SEED)
_shuffled = _rng_cv.permutation(_unique_wells)
_fold_map = {w: i % N_SPLITS for i, w in enumerate(_shuffled)}
_fold_ids = np.array([_fold_map[w] for w in g.values], dtype=np.int32)

# Build (tr_idx, va_idx) split list from _fold_ids
splits = []
for fold in range(N_SPLITS):
    tr = np.where(_fold_ids != fold)[0]
    va = np.where(_fold_ids == fold)[0]
    splits.append((tr, va))
    print(f"  Fold {fold}: train={len(tr):,}  valid={len(va):,}  ({len(np.unique(g.iloc[va]))} wells)")


def _lgb_train_one(params, cfg_idx):
    """Train one LightGBM config across all folds, falling back GPU -> CPU."""
    p = dict(LGB_BASE, **params)
    n_est = p.pop("n_estimators")
    oof = np.zeros(len(train_df), np.float32)
    tp = np.zeros(len(test_df), np.float32)
    use_gpu = p.get("device", "cpu") == "gpu"
    for fold, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        dva = lgb.Dataset(X.iloc[va], label=y.iloc[va], reference=dtr)
        try:
            m = lgb.train(
                p, dtr, valid_sets=[dva], num_boost_round=n_est,
                callbacks=[lgb.early_stopping(150, verbose=False),
                           lgb.log_evaluation(600)],
            )
        except Exception as exc:
            if use_gpu:
                print(f"  LGB GPU failed ({type(exc).__name__}: {exc}); retrying on CPU.")
                p = dict(p, device="cpu")
                use_gpu = False
                m = lgb.train(
                    p, dtr, valid_sets=[dva], num_boost_round=n_est,
                    callbacks=[lgb.early_stopping(150, verbose=False),
                               lgb.log_evaluation(600)],
                )
            else:
                raise
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration).astype(np.float32)
        tp += m.predict(Xt, num_iteration=m.best_iteration).astype(np.float32) / N_SPLITS
        print(f"  LGB{cfg_idx} f{fold}: "
              f"{root_mean_squared_error(y.iloc[va], oof[va]):.4f} "
              f"iter={m.best_iteration}")
    r = root_mean_squared_error(y, oof)
    print(f"  LGB{cfg_idx} OOF={r:.4f}")
    return oof, tp, r


def _xgb_train():
    oof = np.zeros(len(train_df), np.float32)
    tp = np.zeros(len(test_df), np.float32)
    params = dict(XGB_PARAMS)
    use_gpu = params.get("device") == "cuda"
    for fold, (tr, va) in enumerate(splits):
        try:
            m = xgb.XGBRegressor(**params)
            m.fit(
                X.iloc[tr].values, y.iloc[tr].values,
                eval_set=[(X.iloc[va].values, y.iloc[va].values)],
                verbose=500,
            )
        except Exception as exc:
            if use_gpu:
                print(f"  XGB GPU failed ({type(exc).__name__}: {exc}); retrying on CPU.")
                params = dict(params)
                params.pop("device", None)
                params["tree_method"] = "hist"
                use_gpu = False
                m = xgb.XGBRegressor(**params)
                m.fit(
                    X.iloc[tr].values, y.iloc[tr].values,
                    eval_set=[(X.iloc[va].values, y.iloc[va].values)],
                    verbose=500,
                )
            else:
                raise
        oof[va] = m.predict(
            X.iloc[va].values, iteration_range=(0, m.best_iteration)
        ).astype(np.float32)
        tp += m.predict(
            Xt.values, iteration_range=(0, m.best_iteration)
        ).astype(np.float32) / N_SPLITS
        print(f"  XGB f{fold}: {root_mean_squared_error(y.iloc[va], oof[va]):.4f}")
    r = root_mean_squared_error(y, oof)
    print(f"  XGB OOF={r:.4f}")
    return oof, tp, r


# ── V6: CatBoost trainer (NEW) ──
def _cb_train():
    oof = np.zeros(len(train_df), np.float32)
    tp = np.zeros(len(test_df), np.float32)
    for fold, (tr, va) in enumerate(splits):
        m = CatBoostRegressor(**CB_PARAMS)
        m.fit(
            X.iloc[tr].values, y.iloc[tr].values,
            eval_set=(X.iloc[va].values, y.iloc[va].values),
            verbose=500,
        )
        oof[va] = m.predict(X.iloc[va].values).astype(np.float32)
        tp += m.predict(Xt.values).astype(np.float32) / N_SPLITS
        print(f"  CB f{fold}: {root_mean_squared_error(y.iloc[va], oof[va]):.4f}")
        del m; gc.collect()
    r = root_mean_squared_error(y, oof)
    print(f"  CB OOF={r:.4f}")
    return oof, tp, r


results = {}
for i, cfg in enumerate(LGB_CONFIGS):
    oof, tp, r = _lgb_train_one(cfg, i)
    results[f"lgb{i}"] = {"oof": oof, "test": tp, "rmse": r}

oof, tp, r = _xgb_train()
results["xgb"] = {"oof": oof, "test": tp, "rmse": r}

oof, tp, r = _cb_train()
results["cb"] = {"oof": oof, "test": tp, "rmse": r}
"""))

    # ── Cell 17: Stacking section header ──
    cells.append(md("## Positive-weight Ridge stacking"))

    # ── Cell 18: Stacking (MODIFIED: now 5 models with cb) ──
    cells.append(code("""\
Sx = np.column_stack([v["oof"] for v in results.values()])
St = np.column_stack([v["test"] for v in results.values()])
ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)
ridge.fit(Sx, y.values)
oof_s = ridge.predict(Sx)
test_s = ridge.predict(St)
r_avg = root_mean_squared_error(y, Sx.mean(1))
r_stk = root_mean_squared_error(y, oof_s)
wts = ridge.coef_ / max(ridge.coef_.sum(), 1e-9)
print(f"\\nAvg OOF: {r_avg:.4f} | Ridge OOF: {r_stk:.4f}")
print(f"Ridge weights: {dict(zip(results.keys(), wts.round(4)))}")

final_oof = oof_s if r_stk < r_avg else Sx.mean(1)
final_test = test_s if r_stk < r_avg else St.mean(1)
"""))

    # ── Cell 19: Post-processing section header (MODIFIED) ──
    cells.append(md("## Post-processing (alpha x tau only — no SavGol, no w_pf)"))

    # ── Cell 20: Post-processing (MODIFIED: no w_pf, no SavGol) ──
    cells.append(code("""\
# ── V6: Clean post-processing ──
# Removed: w_pf blending (extra OOF tuning dimension -> overfitting risk)
# Removed: Savitzky-Golay smoothing (initial-row bias, confirmed harmful by v10)
# Only alpha * tau grid search with exponential distance decay.

base = train_df["last_known_tvt"].values
ytrue = y.values + base

print("Grid search alpha x tau...")
best_cfg, best_r = (None, None), np.inf
for alpha in np.arange(0.60, 1.01, 0.05):
    for tau in [None, 25.0, 50.0, 100.0, 200.0, 400.0]:
        d = final_oof.copy()
        if tau:
            d = d * (1.0 - np.exp(-np.maximum(train_df["md_since"].values, 0.0) / tau))
        d = d * alpha
        r = root_mean_squared_error(ytrue, base + d)
        if r < best_r:
            best_r, best_cfg = r, (alpha, tau)
ALPHA, TAU = best_cfg
print(f"Best: alpha={ALPHA:.2f} tau={TAU} | absolute TVT OOF RMSE={best_r:.4f}")


def apply_pp(df, md, alpha, tau):
    d = md.copy()
    if tau:
        d = d * (1.0 - np.exp(-np.maximum(df["md_since"].values, 0.0) / tau))
    return d * alpha


test_df2 = test_df.copy()
test_df2["pred"] = (
    test_df2["last_known_tvt"].values
    + apply_pp(test_df2, final_test, ALPHA, TAU)
)
"""))

    # ── Cell 21: Submission section header ──
    cells.append(md("## Submission"))

    # ── Cell 22: Submission (IDENTICAL pattern to V5) ──
    cells.append(code("""\
sample = pd.read_csv(SAMPLE)
sub = sample[["id"]].merge(
    test_df2[["id", "pred"]].rename(columns={"pred": "tvt"}),
    on="id", how="left",
)
fallback = float(train_df["last_known_tvt"].mean() + train_df["target"].mean())
sub["tvt"] = sub["tvt"].fillna(fallback)

assert set(sub.columns) == {"id", "tvt"}, f"Unexpected columns: {sub.columns.tolist()}"
assert len(sub) == len(sample), f"Row count mismatch: {len(sub)} vs {len(sample)}"
assert sub["tvt"].notna().all(), "NaN values remain in tvt"

sub[["id", "tvt"]].to_csv(OUT, index=False)
print(f"\\nWrote {OUT}  rows={len(sub)}")

print("\\n--- Summary ---")
for k, v in results.items():
    print(f"  {k}: OOF residual RMSE = {v['rmse']:.4f}")
print(f"  Stack (residual): {min(r_avg, r_stk):.4f}")
print(f"  Post-proc (absolute TVT OOF): {best_r:.4f}")
print(sub.head().to_string(index=False))
"""))

    # ── Cell 23: Design notes (UPDATED for V6) ──
    cells.append(md("""\
## V6 Design Notes

**Why v6 improves over v5 (anti-overfitting):**

1. **Shuffled GroupKFold** — Plain GroupKFold assigns wells to folds in
   sorted order, causing geographic/temporal clustering. Shuffling well IDs
   before modulo assignment produces more realistic OOF estimates.

2. **Stronger regularization** — `num_leaves=127` (was 255), `reg_lambda=5.0`
   (was 3.0), `reg_alpha=0.1` (was 0.05), `min_child_samples=20` (was 15).
   These constrain tree capacity and force the model to learn robust patterns.

3. **CatBoost diversity** — Oblivious trees provide a structurally different
   hypothesis space from LGBM/XGBoost, improving ensemble robustness without
   adding many parameters.

4. **Clean post-processing** — Only alpha * tau exponential decay. Removed:
   - w_pf blending (third grid dimension -> OOF double-descent)
   - Savitzky-Golay smoothing (initial-row bias confirmed by v10)

5. **Refined features** — Absolute X/Y/Z kept for macro geology; relative
   dx/dy/dz/dxy added for local trajectory context; ktvt_range and ktvt_std
   for known-zone characterization; gr_vs_slp for GR-vs-slope comparison;
   ease_frac removed (redundant with frac, frac2).

**What v6 deliberately omits:**

- Exact (X, Y, Z) coordinate replacement — label leakage (discouraged by AGENTS.md).
- TabICL — external artifact dependency, complexity without stable OOF gain.
- Cal-zone augmentation — risk of same-well data leakage across augment splits.
- Hill-climb stacking — greedy OOF weight optimization, high overfitting risk.

**Adversarial validation note:**

An adversarial classifier (train=0, test=1) on basic features achieves AUC
~0.998, driven almost entirely by absolute coordinates (X, Y, Z). The test
wells occupy a distinct spatial region. This confirms that coordinates encode
well identity. We keep them for macro-geology signal but rely on:
- L1/L2 regularization to penalize over-reliance on any single coordinate
- Shuffled CV to prevent geographic fold clustering
- Ensemble diversity to reduce single-model memorization

**On KNN imputer fold leakage:**

The FormationPlaneKNN and DenseANCCImputer are built from all 773 training
wells before CV. During CV, a validation well can see OTHER validation wells
through the KNN neighbors, causing a small optimistic bias in OOF.
Fixing it completely (per-fold imputers) would 5x the feature-building time.
The `self_wid` exclusion already handles the worst case (self-leakage)."""))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(cells)} cells)")


if __name__ == "__main__":
    main()
