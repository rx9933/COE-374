import contextlib
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
VC_DIR = REPO_ROOT / "Video_Camera_Processing"
K_LIST_PATH = VC_DIR / "K_list.npy"
R_LIST_PATH = VC_DIR / "R_list.npy"
T_LIST_PATH = VC_DIR / "t_list.npy"
P_LIST_PATH = VC_DIR / "P_list.npy"
EXTRINSICS_TUNE2_K_LIST_PATH = VC_DIR / "extrinsics_tune2_K_list.npy"
EXTRINSICS_TUNE2_R_LIST_PATH = VC_DIR / "extrinsics_tune2_R_list.npy"
EXTRINSICS_TUNE2_T_LIST_PATH = VC_DIR / "extrinsics_tune2_t_list.npy"
EXTRINSICS_TUNE2_P_LIST_PATH = VC_DIR / "extrinsics_tune2_P_list.npy"
EXTRINSICS_CHECKPOINT_SCORE_PATH = VC_DIR / "extrinsics_tune2_best_checkpoint_score.txt"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "MathScripts"))
sys.path.insert(0, str(VC_DIR))

import video_to_position_manual as vtm

N_CAMERAS = 2
N_RVEC = 3
N_T = 3
N_PER_CAM = N_RVEC + N_T
NDIM = N_CAMERAS * N_PER_CAM

_FIXED_K_LIST = None

MODE = "de"
# Rodrigues rvec: box [-RVEC_BOUND, RVEC_BOUND]^3 (||r||<=pi is unique except at boundary)
RVEC_BOUND = float(np.pi)
# OpenCV tvec in P = K[R|t]: search within ±T_DELTA (meters) of values in t_list.npy
T_DELTA = 1.0
DE_MAXITER = 15
DE_POPSIZE = 10
DE_POLISH = False
DE_WORKERS = None
DE_SEED = 0
DT = 1.0 / 30.0
PIXEL_SIGMA = 1.0
PHYSICS_SIGMA = 0.1
OMEGA_PHYS = 0.0
DEBUG_EVERY = 5

USE_FINAL_POSITION_TARGET = True
EXPECTED_FINAL_X = 5.4737 + 2.1336
EXPECTED_FINAL_Y = 0.0762
EXPECTED_FINAL_Z = 0.00
W_FINAL_X = 2.0
W_FINAL_Y = 1.0
W_FINAL_Z = 1.0

USE_INITIAL_POSITION_TARGET = True
EXPECTED_START_X = 0.0
EXPECTED_START_Y = 0.0
EXPECTED_START_Z = 1.6764 + 0.8382
W_START_X = 1.0
W_START_Y = 1.0
W_START_Z = 0.5
LAMBDA_REPROJ = 0.0001
W_PHYS = 0.0
LAMBDA_PARABOLIC = 0.000


def _parabolic_penalty(X_opt, dt):
    if LAMBDA_PARABOLIC <= 0:
        return 0.0
    X = np.asarray(X_opt, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 4:
        return 0.0
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0
    acc = 0.0
    for k in range(3):
        x = X[:, k]
        d3 = np.gradient(np.gradient(np.gradient(x, dt), dt), dt)
        acc += float(np.mean(d3 ** 2))
    return acc / 3.0


def _rvec_to_R(rvec: np.ndarray) -> np.ndarray:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R


def _R_to_rvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3)


def _set_fixed_intrinsics(K_list):
    global _FIXED_K_LIST
    _FIXED_K_LIST = [np.asarray(K, dtype=np.float64).reshape(3, 3).copy() for K in K_list]


def flat_to_P_list_and_extrinsics(flat: np.ndarray):
    flat = np.asarray(flat, dtype=np.float64).ravel()
    if flat.size != NDIM:
        raise ValueError("expected flat size %s, got %s" % (NDIM, flat.size))
    if _FIXED_K_LIST is None:
        raise RuntimeError("call _set_fixed_intrinsics before optimizing")
    K_list = []
    R_list = []
    t_list = []
    P_list = []
    off = 0
    for i in range(N_CAMERAS):
        K = _FIXED_K_LIST[i]
        rvec = flat[off : off + N_RVEC]
        off += N_RVEC
        t = flat[off : off + N_T].copy()
        off += N_T
        R = _rvec_to_R(rvec)
        tc = t.reshape(3, 1)
        P = K @ np.hstack([R, tc])
        K_list.append(K)
        R_list.append(R)
        t_list.append(tc)
        P_list.append(P)
    K_stack = np.stack(K_list, axis=0)
    R_stack = np.stack(R_list, axis=0)
    t_stack = np.stack(t_list, axis=0)
    return P_list, K_stack, R_stack, t_stack


def _reprojection_mse(P_list, X_opt, pixels):
    n_t, n_c = X_opt.shape[0], len(P_list)
    se = 0.0
    n_coord = 0
    for t in range(n_t):
        Xh = np.array([X_opt[t, 0], X_opt[t, 1], X_opt[t, 2], 1.0], dtype=np.float64)
        for i in range(n_c):
            obs = np.asarray(pixels[i][t], dtype=np.float64).ravel()
            uh = P_list[i] @ Xh
            w = float(uh[2])
            if abs(w) < 1e-12:
                continue
            u, v = float(uh[0] / w), float(uh[1] / w)
            if obs.size >= 2:
                se += (u - obs[0]) ** 2 + (v - obs[1]) ** 2
                n_coord += 2
    return se / max(1, n_coord)


def _physics_penalty_sq(X_opt, dt, g, physics_sigma, omega_phys):
    n = X_opt.shape[0]
    if n < 3:
        return 0.0
    g = np.asarray(g, dtype=np.float64).ravel()
    drag = 0.0
    acc = 0.0
    count = 0
    for t in range(1, n - 1):
        X_prev, X_curr, X_next = X_opt[t - 1], X_opt[t], X_opt[t + 1]
        phys_res = (X_next - 2 * X_curr + X_prev - g * dt**2 + drag * 0.5 * dt * (X_next - X_prev)) / max(
            physics_sigma, 1e-12
        )
        v = omega_phys * phys_res
        acc += float(np.dot(v, v))
        count += 3
    return acc / max(1, count)


def _final_position_penalty(x_last, y_last, z_last):
    if not USE_FINAL_POSITION_TARGET:
        return 0.0
    return (
        W_FINAL_X * (x_last - EXPECTED_FINAL_X) ** 2
        + W_FINAL_Y * (y_last - EXPECTED_FINAL_Y) ** 2
        + W_FINAL_Z * (z_last - EXPECTED_FINAL_Z) ** 2
    )

def _initial_position_penalty(x_start, y_start, z_start):
    if not USE_INITIAL_POSITION_TARGET:
        return 0.0
    return (
        W_START_X * (x_start - EXPECTED_START_X) ** 2
        + W_START_Y * (y_start - EXPECTED_START_Y) ** 2
        + W_START_Z * (z_start - EXPECTED_START_Z) ** 2
    )



def _score(P_list, X_opt, pixels, dt, g):
    xl, yl, zl = float(X_opt[-1, 0]), float(X_opt[-1, 1]), float(X_opt[-1, 2])
    xf, yf, zf = float(X_opt[0, 0]), float(X_opt[0, 1]), float(X_opt[0, 2])
    pos_pen_final = _final_position_penalty(xl, yl, zl)
    pos_pen_start = _initial_position_penalty(xf, yf, zf)
    parabolic_pen = _parabolic_penalty(X_opt, dt)
    pos_pen = pos_pen_final + pos_pen_start
    reproj = _reprojection_mse(P_list, X_opt, pixels)
    phys = _physics_penalty_sq(X_opt, dt, g, PHYSICS_SIGMA, OMEGA_PHYS) if W_PHYS > 0 else 0.0
    total = pos_pen + LAMBDA_REPROJ * reproj + W_PHYS * phys + LAMBDA_PARABOLIC * parabolic_pen
    meta = {
        "pos_pen": pos_pen,
        "pos_pen_final": pos_pen_final,
        "pos_pen_start": pos_pen_start,
        "first_xyz": (xf, yf, zf),
        "last_xyz": (xl, yl, zl),
        "reproj_mse": reproj,
        "phys_pen": phys,
        "parabolic_pen": parabolic_pen,
    }
    return total, meta


def evaluate_extrinsics(x_flat, orig_sizes, dt, g, pixel_sigma, physics_sigma, omega_phys, quiet):
    stdout_ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    try:
        P_list, K_stack, R_stack, t_stack = flat_to_P_list_and_extrinsics(x_flat)
    except ValueError:
        return 1e9, None, None, None, None
    if not np.all(np.isfinite(x_flat)) or not np.all(np.isfinite(K_stack)):
        return 1e9, None, None, None, None
    for i in range(N_CAMERAS):
        if abs(np.linalg.det(K_stack[i])) < 1e-12:
            return 1e9, None, None, None, None
    try:
        with stdout_ctx:
            out = vtm.run_pipeline_manual(
                vtm.cam_0_pos,
                vtm.cam_1_pos,
                P_list=P_list,
                orig_sizes=orig_sizes,
                dt=dt,
                g=g,
                pixel_sigma=pixel_sigma,
                physics_sigma=physics_sigma,
                omega_phys=omega_phys,
            )
            X_opt = out[0]
            pixels = out[7]
    except (ValueError, RuntimeError):
        return 1e9, None, None, None, None
    total, meta = _score(P_list, X_opt, pixels, dt, g)
    last = meta["last_xyz"]
    return total, last, X_opt, meta, (K_stack, R_stack, t_stack)


def load_initial_rt_flat():
    if not K_LIST_PATH.exists():
        raise SystemExit("missing %s" % K_LIST_PATH)
    if not R_LIST_PATH.exists():
        raise SystemExit("missing %s" % R_LIST_PATH)
    if not T_LIST_PATH.exists():
        raise SystemExit("missing %s" % T_LIST_PATH)
    K_arr = np.load(K_LIST_PATH, allow_pickle=True)
    R_arr = np.load(R_LIST_PATH, allow_pickle=True)
    T_arr = np.load(T_LIST_PATH, allow_pickle=True)
    if K_arr.shape[0] != N_CAMERAS or R_arr.shape[0] != N_CAMERAS:
        raise SystemExit("expected K_list/R_list with first dim %s" % N_CAMERAS)
    K_list = []
    t_refs = []
    parts = []
    for i in range(N_CAMERAS):
        K = np.asarray(K_arr[i], dtype=np.float64).reshape(3, 3)
        R = np.asarray(R_arr[i], dtype=np.float64).reshape(3, 3)
        t = np.asarray(T_arr[i], dtype=np.float64).ravel()
        if t.size != 3:
            raise SystemExit("camera %s t must have 3 elements, got shape %s" % (i, np.asarray(T_arr[i]).shape))
        K_list.append(K)
        t_refs.append(t.copy())
        parts.append(_R_to_rvec(R))
        parts.append(t.copy())
    _set_fixed_intrinsics(K_list)
    x0 = np.concatenate(parts, dtype=np.float64)
    return x0, t_refs


def build_rt_bounds(t_refs):
    b = float(RVEC_BOUND)
    dt = float(T_DELTA)
    bounds = []
    for c in range(N_CAMERAS):
        bounds.extend([(-b, b)] * N_RVEC)
        t = np.asarray(t_refs[c], dtype=np.float64).ravel()
        for j in range(N_T):
            v = float(t[j])
            bounds.append((v - dt, v + dt))
    assert len(bounds) == NDIM
    return bounds


def _P_stack_from_extrinsics(K_stack, R_stack, t_stack):
    return np.stack(
        [
            K_stack[i] @ np.hstack([R_stack[i], t_stack[i].reshape(3, 1)])
            for i in range(N_CAMERAS)
        ],
        axis=0,
    ).astype(np.float64)


def save_extrinsics_tune2_checkpoint(K_stack, R_stack, t_stack):
    """Write best-so-far K, R, t, P under extrinsics_tune2_*_list.npy in VC_DIR."""
    K_stack = K_stack.astype(np.float64)
    R_stack = R_stack.astype(np.float64)
    t_stack = t_stack.astype(np.float64)
    P_stack = _P_stack_from_extrinsics(K_stack, R_stack, t_stack)
    np.save(EXTRINSICS_TUNE2_K_LIST_PATH, K_stack)
    np.save(EXTRINSICS_TUNE2_R_LIST_PATH, R_stack)
    np.save(EXTRINSICS_TUNE2_T_LIST_PATH, t_stack)
    np.save(EXTRINSICS_TUNE2_P_LIST_PATH, P_stack)


def main():
    if not vtm.cam_0_pos or not vtm.cam_1_pos:
        raise SystemExit("fill video_to_position_manual cam_0_pos and cam_1_pos")
    video_paths = [
        REPO_ROOT / "Video_Camera_Processing/throws/Arushi_throw_0.mp4",
        REPO_ROOT / "Video_Camera_Processing/throws/Arushi_throw_1.mp4",
    ]
    for p in video_paths:
        if not p.exists():
            raise SystemExit("missing video: %s" % p)
    orig_sizes = [vtm.read_video_frame_size(p) for p in video_paths]
    g = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    x0, t_refs = load_initial_rt_flat()
    bounds = build_rt_bounds(t_refs)

    print(
        "optimize R (Rodrigues x3) + t (x3) per camera | K fixed from %s | init R,t from %s, %s | ndim=%s | rvec in [-%s,%s]^3 | t in [t_ref-%.2f,t_ref+%.2f] per axis (m) | USE_FINAL_POSITION_TARGET=%s final (x,y,z)=(%s,%s,%s) W_final=(%s,%s,%s) | USE_INITIAL_POSITION_TARGET=%s start (x,y,z)=(%s,%s,%s) W_start=(%s,%s,%s) | LAMBDA_REPROJ=%s W_PHYS=%s | orig_sizes=%s"
        % (
            K_LIST_PATH.name,
            R_LIST_PATH.name,
            T_LIST_PATH.name,
            NDIM,
            RVEC_BOUND,
            RVEC_BOUND,
            T_DELTA,
            T_DELTA,
            USE_FINAL_POSITION_TARGET,
            EXPECTED_FINAL_X,
            EXPECTED_FINAL_Y,
            EXPECTED_FINAL_Z,
            W_FINAL_X,
            W_FINAL_Y,
            W_FINAL_Z,
            USE_INITIAL_POSITION_TARGET,
            EXPECTED_START_X,
            EXPECTED_START_Y,
            EXPECTED_START_Z,
            W_START_X,
            W_START_Y,
            W_START_Z,
            LAMBDA_REPROJ,
            W_PHYS,
            orig_sizes,
        ),
        flush=True,
    )
    print(
        "DE: maxiter=%s popsize=%s polish=%s | checkpoints -> %s, %s, %s, %s"
        % (
            DE_MAXITER,
            DE_POPSIZE,
            DE_POLISH,
            EXTRINSICS_TUNE2_K_LIST_PATH.name,
            EXTRINSICS_TUNE2_R_LIST_PATH.name,
            EXTRINSICS_TUNE2_T_LIST_PATH.name,
            EXTRINSICS_TUNE2_P_LIST_PATH.name,
        ),
        flush=True,
    )

    b_score, b_last, _, b_meta, _ = evaluate_extrinsics(x0, orig_sizes, DT, g, PIXEL_SIGMA, PHYSICS_SIGMA, OMEGA_PHYS, True)
    print("baseline score=%s last_xyz=%s meta=%s" % (b_score, b_last, b_meta), flush=True)

    dbg = {"n": 0, "t0": None, "best": float("inf")}

    def maybe_checkpoint(x_flat, score, tag):
        if not np.isfinite(score) or score >= 1e8:
            return
        if score >= dbg["best"]:
            return
        dbg["best"] = float(score)
        _, K_stack, R_stack, t_stack = flat_to_P_list_and_extrinsics(x_flat)
        save_extrinsics_tune2_checkpoint(K_stack, R_stack, t_stack)
        P_chk = _P_stack_from_extrinsics(K_stack, R_stack, t_stack)
        with open(EXTRINSICS_CHECKPOINT_SCORE_PATH, "w") as f:
            f.write("score=%r\neval_n=%s\ntag=%s\nP_list_shape=%s\n" % (score, dbg["n"], tag, P_chk.shape))
        print(
            "[checkpoint] %s score=%s eval_n=%s -> extrinsics_tune2_*_list.npy in %s"
            % (tag, score, dbg["n"], VC_DIR),
            flush=True,
        )

    maybe_checkpoint(x0, b_score, "baseline")

    def eval_vec(d):
        d = np.asarray(d, dtype=np.float64).ravel()
        if dbg["t0"] is None:
            dbg["t0"] = time.perf_counter()
        t0 = time.perf_counter()
        s, last, _, meta, _ = evaluate_extrinsics(d, orig_sizes, DT, g, PIXEL_SIGMA, PHYSICS_SIGMA, OMEGA_PHYS, True)
        dt_eval = time.perf_counter() - t0
        dbg["n"] += 1
        if s < dbg["best"]:
            maybe_checkpoint(d, s, "de")
        if DEBUG_EVERY > 0 and (dbg["n"] % DEBUG_EVERY == 0):
            wall = time.perf_counter() - dbg["t0"]
            print(
                "[obj_eval %s] score=%s best=%s last=%s meta=%s eval_s=%.2f wall_s=%.1f"
                % (dbg["n"], s, dbg["best"], last, meta, dt_eval, wall),
                flush=True,
            )
        return s

    if MODE != "de":
        raise SystemExit("Only MODE=de is supported. Got MODE=%r." % (MODE,))

    try:
        from scipy.optimize import differential_evolution
    except ImportError as e:
        raise SystemExit("install scipy for differential evolution") from e

    print(
        "starting DE: maxiter=%s popsize=%s polish=%s ndim=%s x0=loaded rvec+t (K fixed)"
        % (DE_MAXITER, DE_POPSIZE, DE_POLISH, NDIM),
        flush=True,
    )

    def de_callback(xk, convergence=None):
        try:
            x = np.asarray(xk, dtype=np.float64).ravel()
            print("[DE iteration] |x|_max=%.6g convergence=%s" % (np.max(np.abs(x)), convergence), flush=True)
        except Exception as ex:
            print("[DE iteration] callback: %s" % ex, flush=True)
        return False

    de_kw = dict(
        x0=x0.copy(),
        maxiter=int(DE_MAXITER),
        popsize=int(DE_POPSIZE),
        seed=int(DE_SEED),
        polish=bool(DE_POLISH),
        atol=1e-5,
        tol=1e-4,
        callback=de_callback,
        disp=False,
    )
    if DE_WORKERS is not None and int(DE_WORKERS) > 1:
        de_kw["workers"] = int(DE_WORKERS)
    res = differential_evolution(eval_vec, bounds, **de_kw)
    best_x = np.asarray(res.x, dtype=np.float64).ravel()
    best_score = float(res.fun)
    _, best_last, _, best_meta, stacks = evaluate_extrinsics(best_x, orig_sizes, DT, g, PIXEL_SIGMA, PHYSICS_SIGMA, OMEGA_PHYS, False)
    K_best, R_best, t_best = stacks

    save_extrinsics_tune2_checkpoint(K_best, R_best, t_best)

    print(
        "BEST score=%s last_xyz=%s meta=%s | wrote %s %s %s %s under %s (canonical K/R/t/P on disk unchanged) | DE nfev=%s nit=%s"
        % (
            best_score,
            best_last,
            best_meta,
            EXTRINSICS_TUNE2_K_LIST_PATH.name,
            EXTRINSICS_TUNE2_R_LIST_PATH.name,
            EXTRINSICS_TUNE2_T_LIST_PATH.name,
            EXTRINSICS_TUNE2_P_LIST_PATH.name,
            VC_DIR,
            getattr(res, "nfev", "?"),
            getattr(res, "nit", "?"),
        ),
        flush=True,
    )
    print(
        "best-so-far during run: %s %s %s %s"
        % (
            EXTRINSICS_TUNE2_K_LIST_PATH.name,
            EXTRINSICS_TUNE2_R_LIST_PATH.name,
            EXTRINSICS_TUNE2_T_LIST_PATH.name,
            EXTRINSICS_TUNE2_P_LIST_PATH.name,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
