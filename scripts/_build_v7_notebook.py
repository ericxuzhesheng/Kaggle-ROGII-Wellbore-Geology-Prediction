"""Build kaggle_kernel_v7_full_retrain.ipynb — three leakage fixes + DTW features.

Fixes over first draft:
- Issue 1: alpha×tau fitted per fold, final = median (no global OOF double-dip)
- Issue 2: per-fold KNN rebuild (no validation-well leakage into formation features)
- Issue 3: FormationPlaneKNN IDW fallback + range-clip (no extrapolation blow-up)
- Restored 3 LGB seeds (v6 level compute)
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


# ===========================================================================
CELL_HEADER = """\
# v7 Full Retrain Ensemble — Three Leakage Fixes + DTW Features

Improvements over v6 (9.791 Public LB):

| Change | Detail |
|--------|--------|
| **Issue-3 fix** | `FormationPlaneKNN.impute()` adds IDW fallback + range-clip → stops extrapolation blow-up on test |
| **Issue-2 fix** | Per-fold KNN rebuild: FI/DI built from training wells only each fold → no validation-well leakage |
| **Issue-1 fix** | alpha×tau searched on each fold's own validation set, final = median across folds → no global OOF double-dip |
| **Full retrain** | CV collects `best_iteration`; final model trains on 100% data at `mean_iter × 1.10` |
| **DTW features** | Sakoe-Chiba 4-radius + stochastic realizations (14+ new columns) |
| **Nelder-Mead** | Replaces Ridge(alpha=1.0) for ensemble weight optimisation |
| **3 LGB seeds** | Restored from v6 (Kaggle can handle the compute) |
| **New features** | `sqrt_frac`, `dist_xyz`, `gr_cumsum`, `prefix_gr_last5/20` |
"""

CELL_CONFIG_SECTION = "## Config and Kaggle environment"

CELL_CONFIG = """\
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

for pkg in ["numba", "catboost"]:
    try: __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--quiet"], check=False)

_kw = Path("/kaggle/working")
_numba_cache = (_kw / ".numba") if _kw.is_dir() else (Path.cwd() / ".numba_cache")
_numba_cache.mkdir(parents=True, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = str(_numba_cache)

SEED = 42
N_SPLITS = 5

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10; DENSE_SPW = 60; DENSE_K = 20

import numpy as np

ANCH_OFFS = np.array([-80,-40,-20,-10,-5,0,5,10,20,40,80], np.float32)
BEAM_OFFS = np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40], np.float32)
SC_OFFS   = np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],   np.float32)
PF_OFFS   = np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],   np.float32)
DTW_OFFS  = np.array([-20,-10,-5,-2,0,2,5,10,20],         np.float32)

DTW_RADII      = (20, 50, 100, 200)
DTW_STOCH_K    = 12
DTW_STOCH_TEMP = 3.0

BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),
    (10,  8.0,  64.0, 2, "loose"),
    ( 8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0,  90.0, 5, "sm5"),
    (20,  4.0,  36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]

PF_N=600; ANCC_N=600; PF_MOM=0.993; PF_VN=0.005; PF_PN=0.01
PF_IV=0.02; PF_IS=0.5; PF_RESAMP=0.5; PF_RP=0.2; PF_RV=0.003
PF_GW=5; PF_GWT=0.3; ANCC_A=0.998; ANCC_RN=0.002; ANCC_PN=0.005
ANCC_IR=0.01; ANCC_IS=0.3; ANCC_RP=0.1; ANCC_RR=0.001
PF_GS_MIN=10.0; PF_GS_MAX=60.0; PF_GS_DEF=30.0

LGB_BASE = dict(
    boosting_type="gbdt", num_leaves=127, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=5.0, reg_alpha=0.1, objective="regression",
    verbose=-1, n_jobs=-1, max_bin=255, force_row_wise=True, device="gpu",
)
# Restored 3 seeds (same as v6 — Kaggle can handle the compute)
LGB_CONFIGS = [
    dict(learning_rate=0.035, n_estimators=6000, random_state=42),
    dict(learning_rate=0.035, n_estimators=6000, random_state=7),
    dict(learning_rate=0.035, n_estimators=6000, random_state=123),
]

XGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.025, max_depth=7,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=2.0,
    min_child_weight=10, objective="reg:squarederror", eval_metric="rmse",
    tree_method="hist", device="cuda", early_stopping_rounds=200,
    random_state=42, n_jobs=-1, verbosity=0,
)

CB_PARAMS = dict(
    iterations=4000, learning_rate=0.035, depth=8, l2_leaf_reg=3.0,
    min_data_in_leaf=20, loss_function="RMSE", eval_metric="RMSE",
    random_seed=42, task_type="GPU", devices="0",
    od_type="Iter", od_wait=150, verbose=0,
)

RETRAIN_SCALE = 1.10

def _find_data_dir() -> Path:
    for p in [Path("/kaggle/input/rogii-wellbore-geology-prediction"),
              Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
              Path.cwd() / "data", Path.cwd()]:
        if (p / "train").is_dir() and (p / "test").is_dir() and (p / "sample_submission.csv").is_file():
            return p
    root = Path("/kaggle/input")
    if root.exists():
        for s in root.glob("**/sample_submission.csv"):
            c = s.parent
            if (c / "train").is_dir() and (c / "test").is_dir():
                return c
    raise FileNotFoundError("ROGII data not found.")

DATA = _find_data_dir()
TRAIN_DIR = DATA / "train"
TEST_DIR  = DATA / "test"
SAMPLE    = DATA / "sample_submission.csv"
OUT_DIR   = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "submission.csv"

print(f"DATA={DATA}  OUT={OUT}")
print(f"train={len(list(TRAIN_DIR.glob('*__horizontal_well.csv')))}  "
      f"test={len(list(TEST_DIR.glob('*__horizontal_well.csv')))}")
"""

CELL_IMPORTS_SECTION = "## Imports"

CELL_IMPORTS = """\
import gc, multiprocessing, time, warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
from numba import njit
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from sklearn.metrics import root_mean_squared_error

warnings.filterwarnings("ignore")
np.random.seed(SEED)
NCPU = min(4, multiprocessing.cpu_count())
print(f"NCPU={NCPU}")
"""

CELL_JIT_SECTION = "## Numba JIT kernels (beam search · particle filters · DTW)"

CELL_JIT = """\
@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t

@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N): cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0.0, 1.0 / N)
    np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u: ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na

@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.0; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt: continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni: fnd = ci; break
                if fnd >= 0:
                    if tot < cC[fnd]: cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX: cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]: mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]: best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n - 1, -1, -1): path[s] = hI[s, b]; b = hP[s, b]
    return path

@njit(cache=True)
def _pf_ancc_jit(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N,
                  ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0
    for i in range(len(md_v)):
        dm = max(md_v[i] - pm, 1.0)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tv = pos[j] - z_v[i]
            tv = max(tv, vmin - 50.0); tv = min(tv, vmin + len(gg) * step + 50.0)
            pos[j] = tv + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600.0 else 0.0, 1e-300)
                w[j] *= lk; ws += w[j]
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1.0 / N
        ne = 0.0
        for j in range(N): ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1.0 / N
        tv = 0.0
        for j in range(N): tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv; va = 0.0
        for j in range(N): va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va ** 0.5; pm = md_v[i]
    return pts, std_

@njit(cache=True)
def _pf_z_jit(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv,
               beta, icpt, zsig, N, MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    pos = np.empty(N); vel = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = ip + 0.5 * np.random.randn()
        vel[j] = iv + 0.02 * np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0; pz = z_v[0] - 1.0
    for i in range(len(md_v)):
        dm = max(md_v[i] - pm, 1.0); dzd = (z_v[i] - pz) / dm; ve = beta * dzd + icpt
        for j in range(N):
            vel[j] = MOM * vel[j] + VN * np.random.randn()
            pos[j] += vel[j] * dm + PN * np.random.randn()
            pos[j] = max(pos[j], vmin - 50.0); pos[j] = min(pos[j], vmin + len(gg_p) * step + 50.0)
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step); dp = (gr_v[i] - ep) / gs
                lp = max(np.exp(-0.5 * dp * dp) if dp * dp < 600.0 else 0.0, 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(gg_s, pos[j], vmin, step); ds = (gr_sm_v[i] - es) / (gs * 1.5)
                    ls = max(np.exp(-0.5 * ds * ds) if ds * ds < 600.0 else 0.0, 1e-300)
                    lk = (1.0 - GR_WT) * lp + GR_WT * ls
                else:
                    lk = lp
                lk = max(lk, 1e-300); w[j] *= lk; ws += w[j]
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1.0 / N
        ws2 = 0.0
        for j in range(N):
            dv = (vel[j] - ve) / max(zsig * 2.0, 0.005)
            lz = max(np.exp(-0.5 * dv * dv) if dv * dv < 600.0 else 0.0, 1e-300)
            w[j] *= lz; ws2 += w[j]
        if ws2 > 0.0:
            for j in range(N): w[j] /= ws2
        else:
            for j in range(N): w[j] = 1.0 / N
        ne = 0.0
        for j in range(N): ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N): w[j] = 1.0 / N
        wm = 0.0
        for j in range(N): wm += w[j] * pos[j]
        pts[i] = wm; va = 0.0
        for j in range(N): va += w[j] * (pos[j] - wm) ** 2
        std_[i] = va ** 0.5; pm = md_v[i]; pz = z_v[i]
    return pts, std_

# ── V7: Sakoe-Chiba DTW (ported from 9-251 notebook) ──────────────────────
@njit(cache=True)
def _dtw_sakoe_chiba(query, ref, radius):
    N = len(query); M = len(ref); INF = 1e18
    D = np.full((N, M), INF)
    slope = (M - 1.0) / max(N - 1.0, 1.0)
    for i in range(N):
        jc = int(round(i * slope))
        j_lo = max(0, jc - radius); j_hi = min(M - 1, jc + radius)
        for j in range(j_lo, j_hi + 1):
            cost = (query[i] - ref[j]) ** 2
            if i == 0 and j == 0:
                D[i, j] = cost
            elif i == 0:
                p = D[i, j - 1]; D[i, j] = cost + (p if p < INF else INF)
            elif j == 0:
                p = D[i - 1, j]; D[i, j] = cost + (p if p < INF else INF)
            else:
                a = D[i-1,j-1]; b = D[i-1,j]; c = D[i,j-1]
                mn = a if a < b else b; mn = mn if mn < c else c
                D[i, j] = cost + (mn if mn < INF else INF)
    i = N - 1; j = M - 1
    pi = np.zeros(N + M, np.int64); pj = np.zeros(N + M, np.int64); k = 0
    while i > 0 or j > 0:
        pi[k] = i; pj[k] = j; k += 1
        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            a = D[i-1,j-1]; b = D[i-1,j]; c = D[i,j-1]
            if a <= b and a <= c: i -= 1; j -= 1
            elif b <= c: i -= 1
            else: j -= 1
    pi[k] = 0; pj[k] = 0; k += 1
    return D, pi[:k], pj[:k]

@njit(cache=True)
def _dtw_path_to_tvt(pi, pj, tw_tvt, N):
    j_for_i = np.zeros(N, np.int64)
    for k in range(len(pi)): j_for_i[pi[k]] = pj[k]
    result = np.empty(N, np.float32)
    for i in range(N): result[i] = tw_tvt[j_for_i[i]]
    return result

@njit(cache=True)
def _dtw_path_slope(pi, pj, N, smooth_win=5):
    j_for_i = np.zeros(N, np.float64)
    for k in range(len(pi)): j_for_i[pi[k]] = float(pj[k])
    slope = np.zeros(N, np.float32); hw = smooth_win // 2
    for i in range(N):
        i0 = max(0, i - hw); i1 = min(N - 1, i + hw)
        slope[i] = float((j_for_i[i1] - j_for_i[i0]) / (i1 - i0)) if i1 > i0 else 1.0
    return slope

@njit(cache=True)
def _dtw_stochastic_realizations(query, ref, radius, K, temperature):
    N = len(query); M = len(ref); INF = 1e18
    slope = (M - 1.0) / max(N - 1.0, 1.0)
    D_base = np.full((N, M), INF)
    for i in range(N):
        jc = int(round(i * slope))
        j_lo = max(0, jc - radius); j_hi = min(M - 1, jc + radius)
        for j in range(j_lo, j_hi + 1):
            cost = (query[i] - ref[j]) ** 2
            if i == 0 and j == 0: D_base[i, j] = cost
            elif i == 0:
                p = D_base[i, j-1]; D_base[i, j] = cost + (p if p < INF else INF)
            elif j == 0:
                p = D_base[i-1, j]; D_base[i, j] = cost + (p if p < INF else INF)
            else:
                a = D_base[i-1,j-1]; b = D_base[i-1,j]; c = D_base[i,j-1]
                mn = a if a < b else b; mn = mn if mn < c else c
                D_base[i, j] = cost + (mn if mn < INF else INF)
    paths = np.zeros((K, N), np.int64)
    for k_idx in range(K):
        D_n = D_base.copy()
        for ii in range(N):
            jc = int(round(ii * slope))
            j_lo = max(0, jc - radius); j_hi = min(M - 1, jc + radius)
            for jj in range(j_lo, j_hi + 1):
                if D_n[ii, jj] < INF:
                    u = np.random.uniform(1e-10, 1.0)
                    D_n[ii, jj] += temperature * (-np.log(-np.log(u)))
        i = N - 1; j = M - 1
        pi_k = np.zeros(N + M, np.int64); pj_k = np.zeros(N + M, np.int64); kk = 0
        while i > 0 or j > 0:
            pi_k[kk] = i; pj_k[kk] = j; kk += 1
            if i == 0: j -= 1
            elif j == 0: i -= 1
            else:
                a = D_n[i-1,j-1]; b = D_n[i-1,j]; c = D_n[i,j-1]
                if a <= b and a <= c: i -= 1; j -= 1
                elif b <= c: i -= 1
                else: j -= 1
        pi_k[kk] = 0; pj_k[kk] = 0; kk += 1
        j_for_i = np.zeros(N, np.int64)
        for m in range(kk): j_for_i[pi_k[m]] = pj_k[m]
        paths[k_idx] = j_for_i
    return paths

def _make_grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    g = np.arange(tmin, tmax + step, step)
    return np.interp(g, tw_tvt, tw_gr).astype(np.float64), tmin, step

def run_dtw_multiscale(full_gr, tw_tvt, tw_gr, last_tvt, radii=DTW_RADII):
    N_full = len(full_gr); M = len(tw_tvt)
    query = full_gr.astype(np.float64); ref = tw_gr.astype(np.float64)
    tvts_ms = {}; slopes_ms = {}; costs_ms = {}; ens_accum = np.zeros(N_full, np.float64)
    for radius in radii:
        D, pi, pj = _dtw_sakoe_chiba(query, ref, radius)
        tvt_arr = _dtw_path_to_tvt(pi, pj, tw_tvt.astype(np.float64), N_full)
        slope_arr = _dtw_path_slope(pi, pj, N_full, smooth_win=5)
        cost_val = float(D[N_full-1, M-1])
        if cost_val >= 1e17: cost_val = float(np.min(D[D < 1e17])) if np.any(D < 1e17) else 0.0
        tvts_ms[radius] = tvt_arr; slopes_ms[radius] = slope_arr; costs_ms[radius] = cost_val
        ens_accum += tvt_arr.astype(np.float64)
    return tvts_ms, slopes_ms, costs_ms, (ens_accum / len(radii)).astype(np.float32)

def run_dtw_stochastic(full_gr, tw_tvt, tw_gr, last_tvt, radius=50,
                        K=DTW_STOCH_K, temperature=DTW_STOCH_TEMP):
    N = len(full_gr)
    paths = _dtw_stochastic_realizations(full_gr.astype(np.float64),
                                          tw_gr.astype(np.float64), radius, K, temperature)
    samples = np.stack([tw_tvt[paths[k].clip(0, len(tw_tvt)-1)].astype(np.float32)
                        for k in range(K)], 0)
    mean_t = samples.mean(0).astype(np.float32)
    std_t  = samples.std(0).astype(np.float32)
    cv_t   = (std_t / (np.abs(mean_t) + 1e-6)).astype(np.float32)
    return mean_t, std_t, cv_t

print("Compiling Numba JIT kernels...")
_dg = np.ones(10, np.float64); _dt = np.ones(20, np.float64)
_gg, _dm, _ds = _make_grid(np.arange(20, dtype=np.float64), _dt)
_beam_jit(_dg, _dt, 5, 5, 10.0, 100.0)
_pf_ancc_jit(np.ones(5), np.zeros(5), np.ones(5), _gg, _dm, _ds, 30.0,
             0.0, 0.0, 10, ANCC_A, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, 0.5)
_q2 = np.random.randn(15).astype(np.float64); _r2 = np.random.randn(20).astype(np.float64)
_dtw_sakoe_chiba(_q2, _r2, 5)
_dtw_path_to_tvt(np.array([0,1,2],np.int64), np.array([0,1,2],np.int64),
                 np.arange(20,dtype=np.float64), 3)
_dtw_path_slope(np.array([0,1,2],np.int64), np.array([0,1,2],np.int64), 3)
_dtw_stochastic_realizations(_q2, _r2, 5, 2, 1.0)
print("JIT ready (beam + PF + DTW).")
"""

CELL_PHYSICS_SECTION = "## Physics wrappers"

CELL_PHYSICS = """\
def beam_search(hgr, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    sgr = pd.Series(hgr.astype(np.float64)).interpolate(limit_direction="both").fillna(
        float(np.nanmean(tw_gr)))
    if r > 0: sgr = sgr.rolling(r * 2 + 1, center=True, min_periods=1).mean()
    sgr = sgr.to_numpy(np.float64)
    si = int(np.searchsorted(tw_tvt, start_tvt))
    si = max(0, min(si, len(tw_tvt) - 1))
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, bs, mc, es)
    return tw_tvt[path.clip(0, len(tw_tvt) - 1)].astype(np.float32)

def run_pf_ancc(hw, tw_tvt, tw_gr):
    gg, vmin, step = _make_grid(tw_tvt, tw_gr)
    k = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    k2 = k[k["GR"].notna()]; gs = PF_GS_DEF
    if len(k2) >= 20:
        gs = float(np.clip(np.std(k2["GR"].values -
            np.interp(k2["TVT_input"].values, tw_tvt, tw_gr)), PF_GS_MIN, PF_GS_MAX))
    ls = float(k["TVT_input"].iloc[-1]) + float(k["Z"].iloc[-1])
    t = k.tail(30); ir = 0.0
    if len(t) >= 10:
        dt = np.diff(t["TVT_input"].values); dz = np.diff(t["Z"].values)
        dm = np.diff(t["MD"].values); m = dm > 0
        if m.sum() >= 3: ir = float(np.median((dt[m] + dz[m]) / dm[m]))
    pts, std_ = _pf_ancc_jit(ev["MD"].to_numpy(np.float64), ev["Z"].to_numpy(np.float64),
        ev["GR"].to_numpy(np.float64), gg, vmin, step, gs, ls, ir, ANCC_N,
        ANCC_A, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, 0.5)
    return pts, std_

def run_pf_z(hw, tw_tvt, tw_gr):
    gg_p, vmin, step = _make_grid(tw_tvt, tw_gr)
    tw_sm = pd.Series(tw_gr).rolling(PF_GW, center=True, min_periods=1).mean().values
    gg_s, _, _ = _make_grid(tw_tvt, tw_sm)
    k = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    k2 = k[k["GR"].notna()]; gs = PF_GS_DEF
    if len(k2) >= 20:
        gs = float(np.clip(np.std(k2["GR"].values -
            np.interp(k2["TVT_input"].values, tw_tvt, tw_gr)), PF_GS_MIN, PF_GS_MAX))
    ktvt = k["TVT_input"].values; kmd = k["MD"].values; kz = k["Z"].values
    dz = np.diff(kz); dtvt = np.diff(ktvt); dmd_ = np.diff(kmd); m = dmd_ > 0
    beta, intc, zsig = -1.0, 0.0, 0.1
    if m.sum() >= 10:
        vz = dz[m] / dmd_[m]; vt = dtvt[m] / dmd_[m]
        c, _, _, _ = np.linalg.lstsq(np.column_stack([vz, np.ones_like(vz)]), vt, rcond=None)
        beta, intc = float(c[0]), float(c[1]); zsig = max(float(np.std(vt - (c[0]*vz+c[1]))), 0.001)
    iv = 0.0
    if len(k) >= 10:
        t = k.tail(20); dt2 = np.diff(t["TVT_input"].values); dm2 = np.diff(t["MD"].values)
        m2 = dm2 > 0
        if m2.sum() >= 3: iv = float(np.median(dt2[m2] / dm2[m2]))
    ip = float(k["TVT_input"].iloc[-1])
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hw_gsm = gr_full.rolling(PF_GW, center=True, min_periods=1).mean()
    pts, std_ = _pf_z_jit(ev["MD"].to_numpy(np.float64), ev["Z"].to_numpy(np.float64),
        ev["GR"].to_numpy(np.float64), hw_gsm.iloc[ev.index].to_numpy(np.float64),
        gg_p, gg_s, vmin, step, gs, ip, iv,
        beta, intc, zsig, PF_N, PF_MOM, PF_VN, PF_PN, PF_GWT, PF_RP, PF_RV, 0.5)
    return pts, std_
"""

CELL_KNN_SECTION = "## Formation plane and dense ANCC imputers (Issue-3 fix: IDW fallback + range-clip)"

CELL_KNN = """\
class FormationPlaneKNN:
    def __init__(self, wids, data_dir):
        rows = []
        for wid in wids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try: df = pd.read_csv(p, usecols=["X","Y"]+FORMATIONS).dropna()
            except Exception: continue
            if len(df) == 0: continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in FORMATIONS: row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df["wid"])}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df["x"].to_numpy()
        self.ya = self.df["y"].to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)
        # Precompute valid range for Issue-3 range-clip
        self.fa_min = self.fa.min(0)
        self.fa_max = self.fa.max(0)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q / self.scale
        nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ord_ = np.argpartition(dist, min(k-1, nf-1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]
        wx = w * xn; wy = w * yn

        # IDW fallback (Issue-3): pure distance-weighted mean, always well-bounded
        sw = w.sum(1, keepdims=True)
        idw = (fn * w[:, :, None]).sum(1) / (sw + 1e-9)

        A = np.zeros((len(q), 3, 3))
        A[:,0,0]=(wx*xn).sum(1); A[:,0,1]=(wx*yn).sum(1); A[:,0,2]=wx.sum(1)
        A[:,1,0]=A[:,0,1];       A[:,1,1]=(wy*yn).sum(1); A[:,1,2]=wy.sum(1)
        A[:,2,0]=A[:,0,2];       A[:,2,1]=A[:,1,2];        A[:,2,2]=w.sum(1)
        A[:,0,0]+=1e-9; A[:,1,1]+=1e-9; A[:,2,2]+=1e-9
        rhs = np.stack([(wx[:,:,None]*fn).sum(1),(wy[:,:,None]*fn).sum(1),
                        (w[:,:,None]*fn).sum(1)], 1)
        try:
            coef = np.linalg.solve(A, rhs)
        except Exception:
            coef = np.zeros((len(q), 3, len(FORMATIONS)))
            for r in range(len(q)):
                try: coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception: pass
        Xq = xy_q[:, 0]; Yq = xy_q[:, 1]
        pred = (Xq[:,None]*coef[:,0,:] + Yq[:,None]*coef[:,1,:] + coef[:,2,:]).astype(np.float32)
        pred[~vk.any(1)] = idw[~vk.any(1)].astype(np.float32)

        # Issue-3 fix: blend plane pred with IDW when plane extrapolates far out of range.
        # Threshold = half the observed formation depth range, minimum 200 ft.
        fa_range = self.fa_max - self.fa_min
        thresh = np.maximum(fa_range * 0.5, 200.0).astype(np.float32)
        too_far = np.abs(pred - idw.astype(np.float32)) > thresh[None, :]
        pred = np.where(too_far, idw.astype(np.float32), pred)
        # Hard clip to [observed_min - 500, observed_max + 500]
        pred = np.clip(pred,
                       (self.fa_min - 500.0).astype(np.float32),
                       (self.fa_max + 500.0).astype(np.float32))
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)


class DenseANCCImputer:
    def __init__(self, wids, data_dir, spw=DENSE_SPW):
        xs, ys, anccs, wids_ = [], [], [], []
        for wid in wids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try: df = pd.read_csv(p, usecols=["X","Y","ANCC"]).dropna()
            except Exception: continue
            if len(df) == 0: continue
            ix = np.linspace(0, len(df)-1, min(spw, len(df)), dtype=int); s = df.iloc[ix]
            xs.append(s["X"].values); ys.append(s["Y"].values)
            anccs.append(s["ANCC"].values); wids_.extend([wid]*len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32)
        self.wids = np.array(wids_)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q = np.atleast_2d(xy_q); q = xy_q / self.scale
        nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid: dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord_ = np.argpartition(dist, min(k-1, nf-1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1); ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1.0/(dk+1e-3), 0.0)
        sw = w.sum(1); safe = np.where(sw < 1e-9, 1.0, sw)
        an = self.ancc[ik]; ap = (an*w).sum(1)/safe
        ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((an - ap[:,None])**2 * w).sum(1) / safe
        return (ap.astype(np.float32),
                np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
                np.where(vk, dk, np.inf).min(1).astype(np.float32))

# Global KNN for test features (built from all 773 training wells — correct, no leakage)
hw_paths = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
train_wids = [p.stem.replace("__horizontal_well", "") for p in hw_paths]
_t0 = time.time()
print(f"Building global KNN for test ({len(train_wids)} wells)...")
FI_global = FormationPlaneKNN(train_wids, TRAIN_DIR)
DI_global = DenseANCCImputer(train_wids, TRAIN_DIR)
_FI, _DI = FI_global, DI_global
print(f"  FPK rows={len(FI_global.df)} | Dense={len(DI_global.ancc):,}  ({time.time()-_t0:.0f}s)")
"""

CELL_FEATS_SECTION = "## Per-well feature builder"

CELL_FEATS = """\
def robust_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6: return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])

def affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1.0, float(np.nanmean(kgr[v]) - np.nanmean(tw_at_k[v])) if v.any() else 0.0
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)

def seg_b_well(ktvt, kz, form_col):
    bv = ktvt + kz - form_col; n = len(bv)
    b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n-50):])) if n >= 5 else b_full
    t1, t2 = n // 3, 2 * n // 3
    b_early = float(np.median(bv[:max(1, t1)])) if t1 > 0 else b_full
    b_mid   = float(np.median(bv[t1:max(t1+1, t2)])) if t2 > t1 else b_full
    w = np.exp(0.02 * np.arange(n)); w /= w.sum()
    return b_full, b_early, b_mid, b_late, float(np.dot(w, bv))

def multi_scale_ncc(kgr, ktvt, hgr, hws=(8,15,25), stride=3):
    out = []
    for hw_sc in hws:
        win = 2*hw_sc+1; nk = len(kgr); nh = len(hgr)
        if nk < win+1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        kg = pd.Series(kgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk-win+1, stride, dtype=np.int32)
        if len(sts) == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        C = kg[sts[:,None]+np.arange(win,dtype=np.int32)[None,:]].astype(np.float32)
        Cn = (C - C.mean(1,keepdims=True)) / (C.std(1,keepdims=True)+1e-6)
        hp = np.pad(hg, hw_sc, mode="edge")
        H = hp[np.arange(nh)[:,None]+np.arange(win)[None,:]].astype(np.float32)
        Hn = (H - H.mean(1,keepdims=True)) / (H.std(1,keepdims=True)+1e-6)
        ncc = Hn @ Cn.T / win; best = ncc.argmax(1); score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best]+hw_sc, 0, nk-1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out],1); scores = np.stack([o[1] for o in out],1)
    sw = np.exp(3.0*scores); sw /= sw.sum(1,keepdims=True)+1e-9
    return out, (tvts*sw).sum(1).astype(np.float32)


def build_well(hw_path, tw_path, is_train):
    wid = Path(hw_path).stem.replace("__horizontal_well","")
    try: hw = pd.read_csv(hw_path); tw = pd.read_csv(tw_path).sort_values("TVT")
    except Exception: return None
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev)==0 or len(kn)<10: return None
    if is_train and ("TVT" not in hw.columns or hw["TVT"].isna().all()): return None
    tw_tvt = tw["TVT"].to_numpy(np.float32); tw_gr = tw["GR"].to_numpy(np.float32)
    if len(tw_tvt) < 3: return None

    np.random.seed(SEED)
    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a)==0: return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)
    pf_use = pf_a.astype(np.float32); std_use = std_a.astype(np.float32)
    has_z = len(pf_z)==len(pf_a) and not np.any(np.isnan(pf_z))

    lk = kn.iloc[-1]; last_tvt = float(lk["TVT_input"])
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr = gr_full.iloc[:len(kn)].to_numpy(np.float32)

    bpaths = {}
    for (bs,mc,es,r,tag) in BEAMS:
        bpaths[tag] = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
    beam_ref = (bpaths["cons"] + bpaths["sm5"]) / 2.0

    ktvt = kn["TVT_input"].to_numpy(np.float32)
    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8,15,25), stride=3)
    sc8,sc8s=sc_res[0]; sc15,sc15s=sc_res[1]; sc25,sc25s=sc_res[2]
    sc_cons = (sc8+sc15+sc25)/3.0
    sc_trust = float(np.clip(len(kn)/200.0, 0.0, 0.6))
    hyb_ref = (1-sc_trust)*beam_ref + sc_trust*sc_ens

    full_gr_arr = gr_full.values.astype(np.float32)
    dtw_tvts_ms, dtw_slopes_ms, dtw_costs_ms, dtw_ens_ms = run_dtw_multiscale(
        full_gr_arr, tw_tvt, tw_gr, last_tvt, radii=DTW_RADII)
    dtw_mean_stoch, dtw_std_stoch, dtw_cv_stoch = run_dtw_stochastic(
        full_gr_arr, tw_tvt, tw_gr, last_tvt, radius=50, K=DTW_STOCH_K, temperature=DTW_STOCH_TEMP)

    ev_start = ev.index[0]; nh = len(ev)
    def _ev(arr): return arr[ev_start:ev_start+nh].astype(np.float32)

    dtw_ens_ev     = _ev(dtw_ens_ms)
    dtw_mean_ev    = _ev(dtw_mean_stoch)
    dtw_std_ev     = _ev(dtw_std_stoch)
    dtw_cv_ev      = _ev(dtw_cv_stoch)
    dtw_per_r_ev   = {r: _ev(dtw_tvts_ms[r]) for r in DTW_RADII}
    dtw_slope_ev   = {r: _ev(dtw_slopes_ms[r]) for r in DTW_RADII}
    dtw_slope_mean_ev = np.stack([dtw_slope_ev[r] for r in DTW_RADII],1).mean(1).astype(np.float32)
    dtw_cost_arr   = np.array([dtw_costs_ms[r] for r in DTW_RADII], np.float32)

    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
    a_cal, b_cal = affine_cal(kgr, tw_at_k)
    kmd = kn["MD"].to_numpy(np.float32); kz_local = kn["Z"].to_numpy(np.float32)
    pfx_rmse = float(np.sqrt(np.mean((kgr-tw_at_k)**2)))
    slp_all = robust_slope(kmd, ktvt); slp_50 = robust_slope(kmd[-50:], ktvt[-50:])
    slp_z   = robust_slope(kn["Z"].to_numpy(), ktvt)

    swid = wid if is_train else None
    xy_ev = ev[["X","Y"]].to_numpy(np.float64); xy_kn = kn[["X","Y"]].to_numpy(np.float64)
    form_ev, knn_d = _FI.impute(xy_ev, self_wid=swid)
    form_kn, _     = _FI.impute(xy_kn, self_wid=swid)
    z_kn = kn["Z"].to_numpy(np.float32); z_ev = ev["Z"].to_numpy(np.float32)

    tvt_fs = {}; form_rmse = {}; form_list = []
    for fi2, fn in enumerate(FORMATIONS):
        b_full,b_early,b_mid,b_late,b_wls = seg_b_well(ktvt, z_kn, form_kn[:,fi2])
        tvt_fs[f"tvtF_{fn}"]   = (-z_ev+form_ev[:,fi2]+b_full).astype(np.float32)
        tvt_fs[f"tvtFw_{fn}"]  = (-z_ev+form_ev[:,fi2]+b_wls).astype(np.float32)
        tvt_fs[f"tvtF50_{fn}"] = (-z_ev+form_ev[:,fi2]+b_late).astype(np.float32)
        tvt_fs[f"bw_{fn}"]      = np.float32(b_full)
        tvt_fs[f"bww_{fn}"]     = np.float32(b_wls)
        tvt_fs[f"bw50_{fn}"]    = np.float32(b_late)
        tvt_fs[f"bw_early_{fn}"]= np.float32(b_early)
        tvt_fs[f"bw_mid_{fn}"]  = np.float32(b_mid)
        form_rmse[fn] = float(np.sqrt(np.mean((ktvt-(-z_kn+form_kn[:,fi2]+b_full))**2)))
        form_list.append(tvt_fs[f"tvtF_{fn}"])
    fs = np.stack(form_list,1)
    form_mean_d = (fs.mean(1)-last_tvt).astype(np.float32)
    form_std_d  = fs.std(1).astype(np.float32)
    form_rng_d  = (fs.max(1)-fs.min(1)).astype(np.float32)

    d_ancc,d_std,d_dist = _DI.impute(xy_ev, self_wid=swid)
    d_kn,d_std_kn,_     = _DI.impute(xy_kn, self_wid=swid)
    b_vd = ktvt+z_kn-d_kn; _,b_de,b_dm,b_dl,b_dw = seg_b_well(ktvt,z_kn,d_kn)
    b_d = float(np.median(b_vd))
    tvt_dense   = (-z_ev+d_ancc+b_d).astype(np.float32)
    tvt_densew  = (-z_ev+d_ancc+b_dw).astype(np.float32)
    tvt_dense50 = (-z_ev+d_ancc+b_dl).astype(np.float32)
    d_rmse = float(np.sqrt(np.mean((ktvt+z_kn-d_kn-b_d)**2)))
    d_bias = float(np.mean(b_vd-b_d))

    all_sigs = ([pf_use]+list(bpaths.values())+
                [sc8,sc15,sc25,sc_ens,tvt_fs["tvtF_ANCC"],tvt_dense,dtw_ens_ev])
    sig_mat = np.stack(all_sigs,1)
    sig_std  = sig_mat.std(1).astype(np.float32)
    sig_mean = (sig_mat.mean(1)-last_tvt).astype(np.float32)

    gr_s = pd.Series(gr_full.values); rolls = {}
    for w_size in [5,21,51,101]:
        r = gr_s.rolling(w_size,center=True,min_periods=1)
        rolls[f"grm{w_size}"] = r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f"grs{w_size}"] = r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1,5,15,30]:
        rolls[f"glag{lag}"]  = gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f"glead{lag}"] = gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1  = gr_s.diff().fillna(0.0).iloc[ev.index].values.astype(np.float32)
    gr_d2  = gr_s.diff().diff().fillna(0.0).iloc[ev.index].values.astype(np.float32)
    gr_env = gr_s.rolling(21,center=True,min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg = np.sqrt(np.maximum(
        (gr_s**2).rolling(21,center=True,min_periods=1).mean(),0.0
    )).iloc[ev.index].values.astype(np.float32)

    gr_cumsum_full = gr_s.cumsum()
    cs_offset = gr_cumsum_full.iloc[ev_start-1] if ev_start > 0 else 0.0
    gr_cumsum_ev = (gr_cumsum_full.iloc[ev.index]-cs_offset).values.astype(np.float32)

    known_gr = gr_s.iloc[:len(kn)]
    prefix_gr_last5  = float(known_gr.iloc[-5:].mean()) if len(known_gr)>=5 else float(known_gr.mean())
    prefix_gr_last20 = float(known_gr.iloc[-20:].mean()) if len(known_gr)>=20 else float(known_gr.mean())

    hmd = ev["MD"].to_numpy(np.float32); md_since = hmd-float(lk["MD"])
    slp_b_all = (last_tvt+slp_all*md_since).astype(np.float32)
    slp_b_50  = (last_tvt+slp_50*md_since).astype(np.float32)
    mdd = hw["MD"].diff().replace(0,np.nan)
    dzdmd = (hw["Z"].diff()/mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd = (hw["X"].diff()/mdd).iloc[ev.index].values.astype(np.float32)
    dydmd = (hw["Y"].diff()/mdd).iloc[ev.index].values.astype(np.float32)

    frac     = (np.arange(nh)/max(nh-1,1)).astype(np.float32)
    sqrt_frac = np.sqrt(frac).astype(np.float32)

    last_x=float(lk["X"]); last_y=float(lk["Y"]); last_z=float(lk["Z"])
    dx  = (ev["X"].to_numpy(np.float32)-last_x).astype(np.float32)
    dy  = (ev["Y"].to_numpy(np.float32)-last_y).astype(np.float32)
    dz  = (z_ev-last_z).astype(np.float32)
    dxy = np.sqrt(dx**2+dy**2).astype(np.float32)
    dist_xyz = np.sqrt(dx**2+dy**2+dz**2).astype(np.float32)

    gr_vs_slp = hgr - np.interp(slp_b_all, tw_tvt, tw_gr).astype(np.float32)

    def sc(v): return np.full(nh, np.float32(v), np.float32)

    feats = {
        "well": wid, "id": [f"{wid}_{i}" for i in ev.index],
        "last_known_tvt": sc(last_tvt),
        "pf_ancc": pf_use, "pf_ancc_std": std_use,
        "pf_ancc_delta": (pf_use-last_tvt).astype(np.float32),
        "pf_z": (pf_z.astype(np.float32) if has_z else sc(last_tvt)),
        "pf_z_delta": ((pf_z-last_tvt).astype(np.float32) if has_z else sc(0.0)),
        "pf_vs_z": ((pf_use-pf_z.astype(np.float32)) if has_z else sc(0.0)),
        **{f"beam_{t}_d":(p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},
        "beam_mean_d":np.stack([(p-last_tvt) for p in bpaths.values()],1).mean(1).astype(np.float32),
        "beam_std_d": np.stack([(p-last_tvt) for p in bpaths.values()],1).std(1).astype(np.float32),
        "beam_med_d": np.median(np.stack([(p-last_tvt) for p in bpaths.values()],1),1).astype(np.float32),
        "sc8_d":(sc8-np.float32(last_tvt)).astype(np.float32),"sc8_sc":sc8s,
        "sc15_d":(sc15-np.float32(last_tvt)).astype(np.float32),"sc15_sc":sc15s,
        "sc25_d":(sc25-np.float32(last_tvt)).astype(np.float32),"sc25_sc":sc25s,
        "sc_cons_d":(sc_cons-np.float32(last_tvt)).astype(np.float32),
        "sc_ens_d":(sc_ens-np.float32(last_tvt)).astype(np.float32),
        "sc_trust":sc(sc_trust), "hyb_d":(hyb_ref-np.float32(last_tvt)).astype(np.float32),
        "sig_std":sig_std, "sig_mean_d":sig_mean,
        "dtw_ens_d":(dtw_ens_ev-last_tvt).astype(np.float32),
        "dtw_stoch_mean_d":(dtw_mean_ev-last_tvt).astype(np.float32),
        "dtw_stoch_std":dtw_std_ev, "dtw_stoch_cv":dtw_cv_ev,
        "dtw_slope_mean":dtw_slope_mean_ev,
        **{f"dtw_r{r}_d":(dtw_per_r_ev[r]-last_tvt).astype(np.float32) for r in DTW_RADII},
        **{f"dtw_slope_r{r}":dtw_slope_ev[r] for r in DTW_RADII},
        "dtw_cost_min":sc(float(dtw_cost_arr.min())),
        "dtw_cost_range":sc(float(dtw_cost_arr.max()-dtw_cost_arr.min())),
        "dtw_vs_beam":(dtw_ens_ev-bpaths["cons"]).astype(np.float32),
        "dtw_vs_pf":(dtw_ens_ev-pf_use).astype(np.float32),
        "dtw_vs_sc":(dtw_ens_ev-sc_ens).astype(np.float32),
        **tvt_fs,
        **{f"frm_rmse_{fn}":sc(form_rmse[fn]) for fn in FORMATIONS},
        "form_mean_d":form_mean_d,"form_std_d":form_std_d,"form_rng_d":form_rng_d,
        "spatial_knn_dist":knn_d,
        "dense_ancc":d_ancc,"dense_std":d_std,"dense_dist":d_dist,
        "tvt_dense_d":(tvt_dense-last_tvt).astype(np.float32),
        "tvt_densew_d":(tvt_densew-last_tvt).astype(np.float32),
        "tvt_dense50_d":(tvt_dense50-last_tvt).astype(np.float32),
        "dense_rmse":sc(d_rmse),"dense_bias":sc(d_bias),
        "pf_vs_spatial":(pf_use-tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "pf_vs_dense":(pf_use-tvt_dense).astype(np.float32),
        "spatial_vs_dense":(tvt_fs["tvtF_ANCC"]-tvt_dense).astype(np.float32),
        "beam_vs_spatial":(bpaths["cons"]-tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "sc_vs_beam":(sc_ens-bpaths["cons"]).astype(np.float32),
        "cal_a":sc(a_cal),"cal_b":sc(b_cal),"pfx_rmse":sc(pfx_rmse),
        "known_len":sc(len(kn)),"eval_len":sc(nh),
        "slp_all":sc(slp_all),"slp_50":sc(slp_50),"slp_z":sc(slp_z),
        "slp_b_d_all":(slp_b_all-last_tvt).astype(np.float32),
        "slp_b_d_50":(slp_b_50-last_tvt).astype(np.float32),
        "dzdmd":dzdmd,"dxdmd":dxdmd,"dydmd":dydmd,
        "md_since":md_since,"frac":frac,"frac2":frac**2,"sqrt_frac":sqrt_frac,
        "z":z_ev,"x":ev["X"].to_numpy(np.float32),"y":ev["Y"].to_numpy(np.float32),
        "dx":dx,"dy":dy,"dz":dz,"dxy":dxy,"dist_xyz":dist_xyz,
        "gr_cumsum":gr_cumsum_ev,
        "prefix_gr_last5":sc(prefix_gr_last5),"prefix_gr_last20":sc(prefix_gr_last20),
        "ktvt_range":sc(float(np.ptp(ktvt))),"ktvt_std":sc(float(ktvt.std())),
        **rolls,"gr_d1":gr_d1,"gr_d2":gr_d2,"gr_env":gr_env,"gr_nrg":gr_nrg,
        "gr_minus_tw_last":(gr_full.iloc[ev.index].values.astype(np.float32)
                            -float(np.interp(last_tvt,tw_tvt,tw_gr))).astype(np.float32),
        "gr_vs_slp":gr_vs_slp,
        "anchor_t_pos":sc(float((last_tvt-float(tw_tvt.min()))/max(float(tw_tvt.max()-tw_tvt.min()),1e-3))),
        "tw_tvt_range":sc(float(tw_tvt.max()-tw_tvt.min())),
        "tw_gr_mean":sc(float(tw_gr.mean())),"tw_gr_std":sc(float(tw_gr.std())),
    }
    hgr_fill = gr_full.iloc[ev.index].values.astype(np.float32)
    for o in ANCH_OFFS:
        feats[f"anch_diff_{int(o)}"] = hgr_fill-float(np.interp(last_tvt+float(o),tw_tvt,tw_gr))
    for o in BEAM_OFFS:
        feats[f"beam_diff_{int(o)}"] = hgr_fill-np.interp(bpaths["cons"]+float(o),tw_tvt,tw_gr).astype(np.float32)
    for o in SC_OFFS:
        feats[f"sc_diff_{int(o)}"]   = hgr_fill-np.interp(sc_ens+float(o),tw_tvt,tw_gr).astype(np.float32)
    for o in PF_OFFS:
        feats[f"pf_diff_{int(o)}"]   = hgr_fill-np.interp(pf_use+float(o),tw_tvt,tw_gr).astype(np.float32)
    for o in DTW_OFFS:
        feats[f"dtw_diff_{int(o)}"]  = hgr_fill-np.interp(dtw_ens_ev+float(o),tw_tvt,tw_gr).astype(np.float32)
    if is_train:
        feats["target"] = (ev["TVT"].to_numpy(np.float32)-np.float32(last_tvt))
    df = pd.DataFrame(feats)
    for c in df.select_dtypes("float64").columns: df[c] = df[c].astype(np.float32)
    return df

def build_dataset(paths, is_train, label):
    tw_dir = TRAIN_DIR if is_train else TEST_DIR
    print(f"  {label}: {len(paths)} wells | {NCPU} threads")
    t0 = time.time()
    res = Parallel(n_jobs=NCPU, backend="threading", verbose=0)(
        delayed(build_well)(str(p),
            str(tw_dir / p.name.replace("__horizontal_well.csv","__typewell.csv")), is_train)
        for p in paths)
    ok = [r for r in res if r is not None]
    print(f"  {label}: OK={len(ok)} skip={len(paths)-len(ok)} | {time.time()-t0:.0f}s")
    return pd.concat(ok, ignore_index=True)
"""

CELL_BUILD_TEST_SECTION = "## Build test features (global KNN — no leakage) and determine feature_cols"

CELL_BUILD_TEST = """\
# Test features use global FI/DI (all 773 training wells) — correct, no leakage.
_FI, _DI = FI_global, DI_global
test_paths = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
print("Building test features with global KNN...")
test_df = build_dataset(test_paths, is_train=False, label="test")
print(f"test: {test_df.shape}")

SKIP = {"well", "id", "target"}
feature_cols = [c for c in test_df.columns if c not in SKIP]
print(f"#features: {len(feature_cols)}")
Xt = test_df[feature_cols].astype(np.float32)
gc.collect()
"""

CELL_TRAIN_SECTION = "## V7: Per-fold feature building + training (Issue-2 fix: fold-specific KNN)"

CELL_TRAIN = """\
# Issue-2 fix: build KNN from training wells only each fold.
# The global FI/DI is NOT used for training; only test uses it.

# Determine fold membership by well
_rng_cv = np.random.RandomState(SEED)
_shuffled = _rng_cv.permutation(train_wids)
_wid_to_fold = {w: i % N_SPLITS for i, w in enumerate(_shuffled)}
print(f"CV: {N_SPLITS} folds over {len(train_wids)} wells (shuffled).")
for fold in range(N_SPLITS):
    wv = [w for w in train_wids if _wid_to_fold[w] == fold]
    wt = [w for w in train_wids if _wid_to_fold[w] != fold]
    print(f"  Fold {fold}: train={len(wt)}  val={len(wv)}")

best_iters = {"lgb": [], "xgb": [], "cb": []}

# OOF storage: keyed by sample ID
oof_by_id  = {mn: {} for mn in
               [f"lgb{i}" for i in range(len(LGB_CONFIGS))] + ["xgb", "cb"]}
y_by_id    = {}   # id -> target residual
base_by_id = {}   # id -> last_known_tvt
md_by_id   = {}   # id -> md_since

# Per-fold pp params (Issue-1 fix)
fold_pp_params = []   # list of (alpha, tau) per fold


def _lgb_fit(p, n_est, X_tr, y_tr, X_va, y_va, use_gpu):
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    try:
        m = lgb.train(p, dtr, valid_sets=[dva], num_boost_round=n_est,
                      callbacks=[lgb.early_stopping(150,verbose=False),
                                 lgb.log_evaluation(600)])
    except Exception as exc:
        if use_gpu:
            print(f"    LGB GPU fail → CPU"); p = dict(p, device="cpu")
            m = lgb.train(p, dtr, valid_sets=[dva], num_boost_round=n_est,
                          callbacks=[lgb.early_stopping(150,verbose=False),
                                     lgb.log_evaluation(600)])
        else: raise
    return m


for fold in range(N_SPLITS):
    print(f"\\n===== Fold {fold} =====")
    wids_tr = [w for w in train_wids if _wid_to_fold[w] != fold]
    wids_va = [w for w in train_wids if _wid_to_fold[w] == fold]

    # Build fold-specific KNN (training wells only) — Issue-2 fix
    _t_knn = time.time()
    FI_fold = FormationPlaneKNN(wids_tr, TRAIN_DIR)
    DI_fold = DenseANCCImputer(wids_tr, TRAIN_DIR)
    _FI, _DI = FI_fold, DI_fold
    print(f"  Fold KNN built ({time.time()-_t_knn:.0f}s): FPK={len(FI_fold.df)} Dense={len(DI_fold.ancc):,}")

    paths_tr = [TRAIN_DIR/f"{w}__horizontal_well.csv" for w in wids_tr]
    paths_va = [TRAIN_DIR/f"{w}__horizontal_well.csv" for w in wids_va]

    df_tr = build_dataset(paths_tr, is_train=True, label=f"f{fold}_train")
    df_va = build_dataset(paths_va, is_train=True, label=f"f{fold}_val")

    # Align feature columns (guarantee consistency)
    for col in feature_cols:
        if col not in df_tr.columns: df_tr[col] = 0.0
        if col not in df_va.columns: df_va[col] = 0.0

    X_tr = df_tr[feature_cols].astype(np.float32).values
    y_tr = df_tr["target"].values
    X_va = df_va[feature_cols].astype(np.float32).values
    y_va = df_va["target"].values

    # Store metadata for OOF assembly
    for _, row in df_va[["id","target","last_known_tvt","md_since"]].iterrows():
        y_by_id[row["id"]]    = row["target"]
        base_by_id[row["id"]] = row["last_known_tvt"]
        md_by_id[row["id"]]   = row["md_since"]

    # ── LightGBM ───────────────────────────────────────────────────────────
    fold_val_preds = {}
    for ci, cfg in enumerate(LGB_CONFIGS):
        p = dict(LGB_BASE, **cfg); n_est = p.pop("n_estimators")
        m = _lgb_fit(p, n_est, X_tr, y_tr, X_va, y_va, use_gpu=(p.get("device")=="gpu"))
        pred_va = m.predict(X_va, num_iteration=m.best_iteration).astype(np.float32)
        fold_val_preds[f"lgb{ci}"] = pred_va
        best_iters["lgb"].append(m.best_iteration)
        r = root_mean_squared_error(y_va, pred_va)
        print(f"  LGB{ci} fold{fold}: {r:.4f}  iter={m.best_iteration}")
        for id_, pv in zip(df_va["id"].values, pred_va):
            oof_by_id[f"lgb{ci}"][id_] = float(pv)

    # ── XGBoost ────────────────────────────────────────────────────────────
    params_xgb = dict(XGB_PARAMS); use_gpu_xgb = params_xgb.get("device") == "cuda"
    try:
        m_xgb = xgb.XGBRegressor(**params_xgb)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=500)
    except Exception:
        if use_gpu_xgb:
            print("  XGB GPU fail → CPU"); params_xgb.pop("device",None)
            params_xgb["tree_method"]="hist"; use_gpu_xgb=False
            m_xgb = xgb.XGBRegressor(**params_xgb)
            m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=500)
        else: raise
    pred_xgb = m_xgb.predict(X_va, iteration_range=(0,m_xgb.best_iteration)).astype(np.float32)
    fold_val_preds["xgb"] = pred_xgb
    best_iters["xgb"].append(m_xgb.best_iteration)
    print(f"  XGB fold{fold}: {root_mean_squared_error(y_va,pred_xgb):.4f}  iter={m_xgb.best_iteration}")
    for id_, pv in zip(df_va["id"].values, pred_xgb):
        oof_by_id["xgb"][id_] = float(pv)
    del m_xgb; gc.collect()

    # ── CatBoost ───────────────────────────────────────────────────────────
    m_cb = CatBoostRegressor(**CB_PARAMS)
    m_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=500)
    pred_cb = m_cb.predict(X_va).astype(np.float32)
    fold_val_preds["cb"] = pred_cb
    best_it = m_cb.get_best_iteration()
    best_iters["cb"].append(best_it if best_it is not None else CB_PARAMS["iterations"])
    print(f"  CB fold{fold}: {root_mean_squared_error(y_va,pred_cb):.4f}  iter={best_iters['cb'][-1]}")
    for id_, pv in zip(df_va["id"].values, pred_cb):
        oof_by_id["cb"][id_] = float(pv)
    del m_cb; gc.collect()

    # ── Issue-1 fix: per-fold alpha×tau search ─────────────────────────────
    # Compute a simple ensemble of this fold's val predictions, then find
    # best alpha×tau on this fold's validation set only.
    fold_ens_va = np.stack(list(fold_val_preds.values()), 1).mean(1)
    base_va  = df_va["last_known_tvt"].values
    ytrue_va = y_va + base_va
    mds_va   = df_va["md_since"].values
    _best_r_fold = np.inf; _best_cfg_fold = (1.0, None)
    for _alpha in np.arange(0.60, 1.01, 0.05):
        for _tau in [None, 25.0, 50.0, 100.0, 200.0, 400.0]:
            d = fold_ens_va.copy()
            if _tau: d = d * (1.0 - np.exp(-np.maximum(mds_va, 0.0) / _tau))
            d = d * _alpha
            _r = root_mean_squared_error(ytrue_va, base_va + d)
            if _r < _best_r_fold: _best_r_fold = _r; _best_cfg_fold = (_alpha, _tau)
    fold_pp_params.append(_best_cfg_fold)
    print(f"  Fold {fold} pp: alpha={_best_cfg_fold[0]:.2f} tau={_best_cfg_fold[1]} | abs_RMSE={_best_r_fold:.4f}")

    del df_tr, df_va, FI_fold, DI_fold; gc.collect()

# Restore global KNN for full retrain later
_FI, _DI = FI_global, DI_global

print("\\n=== CV done ===")
print(f"best_iters: { {k: f'{np.mean(v):.0f}' for k,v in best_iters.items()} }")
print(f"per-fold pp: {fold_pp_params}")
"""

CELL_ENSEMBLE_SECTION = "## V7: Nelder-Mead ensemble on assembled OOF"

CELL_ENSEMBLE = """\
# Assemble OOF from per-ID dictionaries (consistent ordering via sorted IDs)
all_ids = sorted(y_by_id.keys())
model_names = [f"lgb{i}" for i in range(len(LGB_CONFIGS))] + ["xgb", "cb"]

y_oof   = np.array([y_by_id[id_]    for id_ in all_ids], dtype=np.float64)
base_oof = np.array([base_by_id[id_] for id_ in all_ids], dtype=np.float64)
md_oof   = np.array([md_by_id[id_]   for id_ in all_ids], dtype=np.float64)

Sx = np.column_stack([
    np.array([oof_by_id[mn].get(id_, 0.0) for id_ in all_ids], dtype=np.float64)
    for mn in model_names
])

def _ens_rmse(raw_w):
    w = np.maximum(raw_w, 0.0); s = w.sum()
    if s < 1e-12: return 1e9
    return float(root_mean_squared_error(y_oof, Sx @ (w / s)))

n_models = Sx.shape[1]
init_w = np.ones(n_models) / n_models
res = minimize(_ens_rmse, init_w, method="Nelder-Mead",
               options={"maxiter":20000, "xatol":1e-12, "fatol":1e-12})
nm_w = np.maximum(res.x, 0.0); nm_w /= nm_w.sum()

oof_nm  = Sx @ nm_w
r_avg   = root_mean_squared_error(y_oof, Sx.mean(1))
r_nm    = root_mean_squared_error(y_oof, oof_nm)
print(f"\\nAvg OOF: {r_avg:.4f} | Nelder-Mead OOF: {r_nm:.4f}")
print(f"Weights: {dict(zip(model_names, nm_w.round(4)))}")

final_oof     = oof_nm if r_nm < r_avg else Sx.mean(1)
final_weights = nm_w   if r_nm < r_avg else init_w
"""

CELL_RETRAIN_SECTION = "## V7: Full data retraining for test predictions"

CELL_RETRAIN = """\
# Train on 100% data at mean(best_iter) × RETRAIN_SCALE using global KNN (correct).
final_n_iters = {
    "lgb": max(50, int(round(np.mean(best_iters["lgb"]) * RETRAIN_SCALE))),
    "xgb": max(50, int(round(np.mean(best_iters["xgb"]) * RETRAIN_SCALE))),
    "cb":  max(50, int(round(np.mean(best_iters["cb"])  * RETRAIN_SCALE))),
}
print(f"Full-retrain n_iters: {final_n_iters}")

# Build full training features with global KNN
_FI, _DI = FI_global, DI_global
print("Building full train features with global KNN...")
_t_full = time.time()
train_df_full = build_dataset(hw_paths, is_train=True, label="full_train")
print(f"full_train: {train_df_full.shape}  ({time.time()-_t_full:.0f}s)")
for col in feature_cols:
    if col not in train_df_full.columns: train_df_full[col] = 0.0
X_all = train_df_full[feature_cols].astype(np.float32).values
y_all = train_df_full["target"].values
Xt_arr = Xt.values

test_preds_r = {}

# LGB full retrain
lgb_preds_full = []
for ci, cfg in enumerate(LGB_CONFIGS):
    p = dict(LGB_BASE, **cfg); p.pop("n_estimators", None)
    use_gpu = p.get("device","cpu") == "gpu"
    try:
        m = lgb.train(p, lgb.Dataset(X_all, label=y_all), num_boost_round=final_n_iters["lgb"])
    except Exception:
        if use_gpu:
            print(f"  LGB{ci} full retrain GPU → CPU"); p = dict(p, device="cpu")
            m = lgb.train(p, lgb.Dataset(X_all, label=y_all), num_boost_round=final_n_iters["lgb"])
        else: raise
    lgb_preds_full.append(m.predict(Xt_arr).astype(np.float32))
    del m; gc.collect()
    print(f"  LGB{ci} full retrain done  n={final_n_iters['lgb']}")
test_preds_r["lgb"] = np.stack(lgb_preds_full,0).mean(0)

# XGB full retrain
p_xgb = dict(XGB_PARAMS); p_xgb.pop("early_stopping_rounds",None)
p_xgb["n_estimators"] = final_n_iters["xgb"]
use_gpu_xgb = p_xgb.get("device") == "cuda"
try:
    m_xgb = xgb.XGBRegressor(**p_xgb); m_xgb.fit(X_all, y_all, verbose=500)
except Exception:
    if use_gpu_xgb:
        print("  XGB full retrain GPU → CPU"); p_xgb.pop("device",None)
        p_xgb["tree_method"]="hist"; m_xgb = xgb.XGBRegressor(**p_xgb)
        m_xgb.fit(X_all, y_all, verbose=500)
    else: raise
test_preds_r["xgb"] = m_xgb.predict(Xt_arr).astype(np.float32)
del m_xgb; gc.collect(); print(f"  XGB full retrain done  n={final_n_iters['xgb']}")

# CB full retrain
p_cb = dict(CB_PARAMS); p_cb.pop("od_type",None); p_cb.pop("od_wait",None)
p_cb["iterations"] = final_n_iters["cb"]
m_cb = CatBoostRegressor(**p_cb); m_cb.fit(X_all, y_all, verbose=500)
test_preds_r["cb"] = m_cb.predict(Xt_arr).astype(np.float32)
del m_cb; gc.collect(); print(f"  CB full retrain done  n={final_n_iters['cb']}")

# Apply ensemble weights to full-retrain test predictions
# Map OOF model names -> retrain model names (lgb0/lgb1/lgb2 all → "lgb")
retrain_names = ["lgb", "xgb", "cb"]
w_map = {}
for i, mn in enumerate(model_names):
    base = "lgb" if mn.startswith("lgb") else mn
    w_map[base] = w_map.get(base, 0.0) + final_weights[i]
rt_weights = np.array([w_map.get(n, 0.0) for n in retrain_names], np.float64)
if rt_weights.sum() < 1e-9: rt_weights[:] = 1.0 / len(retrain_names)
else: rt_weights /= rt_weights.sum()

St = np.column_stack([test_preds_r[n] for n in retrain_names])
final_test = St @ rt_weights
print(f"\\nRetrain weights: {dict(zip(retrain_names, rt_weights.round(4)))}")
print(f"Test pred  mean={final_test.mean():.3f}  std={final_test.std():.3f}")
"""

CELL_PP_SECTION = "## V7: Post-processing — Issue-1 fix: alpha×tau = median of per-fold optima"

CELL_PP = """\
# Issue-1 fix: use median of per-fold alpha×tau instead of global OOF search.
# Each fold found its own optimum on a true holdout — median is unbiased.
fold_alphas = [p[0] for p in fold_pp_params]
fold_taus   = [p[1] for p in fold_pp_params]

# For tau: treat None as 0 for median computation, then round back
_tau_nums = [t if t is not None else 0.0 for t in fold_taus]
ALPHA = float(np.median(fold_alphas))
_tau_med = float(np.median(_tau_nums))
TAU = None if _tau_med < 12.5 else float(_tau_med)

print(f"Per-fold alphas: {[round(a,2) for a in fold_alphas]}")
print(f"Per-fold taus:   {fold_taus}")
print(f"Final: alpha={ALPHA:.2f}  tau={TAU}")

# Compute OOF absolute RMSE for reporting (not used for tuning)
_d_oof = final_oof.copy()
if TAU: _d_oof = _d_oof * (1.0 - np.exp(-np.maximum(md_oof, 0.0) / TAU))
_d_oof = _d_oof * ALPHA
_oof_abs_rmse = root_mean_squared_error(y_oof + base_oof, base_oof + _d_oof)
print(f"OOF absolute TVT RMSE (reporting only): {_oof_abs_rmse:.4f}")


def apply_pp(md_since_arr, pred_resid, alpha, tau):
    d = pred_resid.copy()
    if tau: d = d * (1.0 - np.exp(-np.maximum(md_since_arr, 0.0) / tau))
    return d * alpha


test_df2 = test_df.copy()
test_df2["pred"] = (
    test_df2["last_known_tvt"].values
    + apply_pp(test_df2["md_since"].values, final_test, ALPHA, TAU)
)
"""

CELL_SUB_SECTION = "## Submission"

CELL_SUB = """\
sample = pd.read_csv(SAMPLE)
sub = sample[["id"]].merge(
    test_df2[["id","pred"]].rename(columns={"pred":"tvt"}), on="id", how="left")
fallback = float(train_df_full["last_known_tvt"].mean() + train_df_full["target"].mean())
sub["tvt"] = sub["tvt"].fillna(fallback)

assert set(sub.columns) == {"id","tvt"}
assert len(sub) == len(sample)
assert sub["tvt"].notna().all()

sub[["id","tvt"]].to_csv(OUT, index=False)
print(f"Wrote {OUT}  rows={len(sub)}")
print(f"\\n--- OOF summary (residual RMSE per model) ---")
for mn in model_names:
    preds = np.array([oof_by_id[mn].get(id_,0.0) for id_ in all_ids])
    print(f"  {mn}: {root_mean_squared_error(y_oof, preds):.4f}")
r_final = root_mean_squared_error(y_oof, final_oof)
print(f"  Ensemble: {r_final:.4f}")
print(f"  Post-proc abs TVT OOF: {_oof_abs_rmse:.4f}")
print(sub.head().to_string(index=False))
"""

CELL_NOTES = """\
## V7 Design Notes

### Three leakage fixes

| Issue | Problem | Fix |
|-------|---------|-----|
| Issue-1 | Nelder-Mead + alpha×tau both fit on global OOF | alpha×tau searched per-fold on that fold's val set; final = median |
| Issue-2 | FormationKNN/DenseANCC built from all 773 wells before CV | KNN rebuilt each fold from training wells only |
| Issue-3 | `np.linalg.solve` plane extrapolates wildly on unseen test coordinates | IDW fallback + hard range-clip added to `FormationPlaneKNN.impute()` |

### Other improvements over v6

- **Full data retraining**: CV collects `best_iteration`; final model trained at `mean_iter × 1.10` on 100% data
- **Nelder-Mead ensemble**: replaces `Ridge(alpha=1.0)` — directly optimises non-negative weights on OOF
- **DTW features**: Sakoe-Chiba at 4 radii + stochastic realizations — 22+ new columns
- **New features**: `sqrt_frac`, `dist_xyz`, `gr_cumsum`, `prefix_gr_last5/20`
- **3 LGB seeds restored**: v6 level (diversity from structure, not seed reduction)
"""

# ===========================================================================
cells = [
    md_cell(CELL_HEADER),
    md_cell(CELL_CONFIG_SECTION), code_cell(CELL_CONFIG),
    md_cell(CELL_IMPORTS_SECTION), code_cell(CELL_IMPORTS),
    md_cell(CELL_JIT_SECTION), code_cell(CELL_JIT),
    md_cell(CELL_PHYSICS_SECTION), code_cell(CELL_PHYSICS),
    md_cell(CELL_KNN_SECTION), code_cell(CELL_KNN),
    md_cell(CELL_FEATS_SECTION), code_cell(CELL_FEATS),
    md_cell(CELL_BUILD_TEST_SECTION), code_cell(CELL_BUILD_TEST),
    md_cell(CELL_TRAIN_SECTION), code_cell(CELL_TRAIN),
    md_cell(CELL_ENSEMBLE_SECTION), code_cell(CELL_ENSEMBLE),
    md_cell(CELL_RETRAIN_SECTION), code_cell(CELL_RETRAIN),
    md_cell(CELL_PP_SECTION), code_cell(CELL_PP),
    md_cell(CELL_SUB_SECTION), code_cell(CELL_SUB),
    md_cell(CELL_NOTES),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = ROOT / "kaggle_kernel_v7_full_retrain.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out}")
print(f"  Cells: {len(cells)}  Code: {sum(1 for c in cells if c['cell_type']=='code')}")
