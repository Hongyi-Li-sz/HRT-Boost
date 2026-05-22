#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRT-Boost regression benchmark.

This script reproduces HRT-Boost experiments and can optionally run additional
classical tree ensembles and tabular deep-learning baselines. By default, only
HRT-Boost is evaluated. Set RUN_METHODS to enable HRT ablations or baselines.

Typical usage:
    python scripts/run_benchmark.py
    RUN_METHODS="HRT-Boost,HRT,CART,RF" python scripts/run_benchmark.py

Expected data location:
    data/raw/

Outputs:
    outputs/benchmarking_results_paper_ready_fastv2.pdf
    outputs/benchmarking_results_all_methods_fastv2_table5_xgb.tex

Environment variables:
    DATA_DIR             Directory containing dataset files. Default: data/raw
    OUTPUT_DIR           Directory for generated PDF/LaTeX/cache files. Default: outputs
    RUN_METHODS          Comma-separated method list or group: hrtboost, hrt, baselines,
                         classical/tree, deep, all. Exclusions are supported, e.g.
                         RUN_METHODS="all,-TabM,-TabNet".
    USE_GPU_ACCEL        1/0 toggle for GPU probing and acceleration. Default: 1
    N_JOBS               Global CPU parallelism. Default: -1
    GRID_N_JOBS          GridSearchCV parallelism. Default: N_JOBS
    FINAL_MODEL_N_JOBS   Final repeated-fit model parallelism. Default: N_JOBS

Notes:
    Optional baselines are skipped automatically if their libraries are not installed.
"""

import os
import time
import warnings
import inspect
import gc
import pickle
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin, clone
from joblib import Memory

warnings.filterwarnings('ignore')


# -------------------------------------------------------------------
# Optional dependency imports
# -------------------------------------------------------------------

def try_import(lib_name, class_name):
    try:
        module = __import__(lib_name, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        return None


LGBMRegressor = try_import('lightgbm', 'LGBMRegressor')
XGBRegressor = try_import('xgboost', 'XGBRegressor')


# -------------------------------------------------------------------
# 1.2 TabNet
# pip install pytorch-tabnet torch
# -------------------------------------------------------------------

TabNetRegressorRaw = try_import('pytorch_tabnet.tab_model', 'TabNetRegressor')


# -------------------------------------------------------------------
# 1.3 TabM
# pip install tabm torch
# -------------------------------------------------------------------

TabMRaw = try_import('tabm', 'TabM')


# -------------------------------------------------------------------
# 1.4 TabNet / TabM sklearn wrapper
# -------------------------------------------------------------------

def filter_supported_params(callable_obj, params):
    """Internal benchmark helper."""
    try:
        sig = inspect.signature(callable_obj)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return params
        return {k: v for k, v in params.items() if k in sig.parameters}
    except Exception:
        return params


def filter_supported_fit_params(fit_func, params):
    """Internal benchmark helper."""
    try:
        sig = inspect.signature(fit_func)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return params
        return {k: v for k, v in params.items() if k in sig.parameters}
    except Exception:
        return params


def to_dense_float32(X):
    """Internal benchmark helper."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def make_one_hot_encoder():
    """Internal benchmark helper."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def choose_virtual_batch_size(batch_size, target_virtual_batch_size):
    """Internal benchmark helper."""
    batch_size = int(batch_size)

    if batch_size <= 1:
        return 1

    target = max(2, min(int(target_virtual_batch_size), batch_size))

    for v in range(target, 1, -1):
        n_chunks = int(np.ceil(batch_size / v))
        if batch_size // n_chunks >= 2:
            return v

    return batch_size


class TabMSklearnRegressor(BaseEstimator, RegressorMixin):
    """scikit-learn wrapper for tabm.TabM used by the optional benchmark."""
    def __init__(self, arch_type='tabm', k=16, n_blocks=2, d_block=128, dropout=0.1,
                 learning_rate=2e-3, weight_decay=3e-4, max_epochs=50, batch_size=256,
                 device_name='auto', random_state=None, verbose=0):
        self.arch_type = arch_type
        self.k = k
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device_name = device_name
        self.random_state = random_state
        self.verbose = verbose

    def _resolve_device(self):
        if self.device_name not in {None, 'auto'}:
            return self.device_name
        return 'cuda' if TORCH_CUDA_AVAILABLE else 'cpu'

    def fit(self, X, y):
        if TabMRaw is None:
            raise ImportError('tabm is not installed. Install it with: pip install tabm torch')
        try:
            import torch
        except ImportError as e:
            raise ImportError('tabm requires torch. Install it with: pip install torch') from e

        X_np = to_dense_float32(X)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y_np).astype(np.float32)

        if X_np.shape[0] < 2:
            raise ValueError('TabM requires at least two training samples.')

        seed = 0 if self.random_state is None else int(self.random_state)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        device = torch.device(self._resolve_device())
        candidate_params = {
            'n_num_features': X_np.shape[1],
            'cat_cardinalities': [],
            'd_out': 1,
            'arch_type': self.arch_type,
            'k': int(self.k),
            'n_blocks': int(self.n_blocks),
            'd_block': int(self.d_block),
            'dropout': float(self.dropout),
        }
        model_params = filter_supported_params(TabMRaw.make, candidate_params)
        self.model_ = TabMRaw.make(**model_params).to(device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay))

        X_tensor = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y_scaled, dtype=torch.float32, device=device)

        n_train = X_tensor.shape[0]
        batch_size = max(1, min(int(self.batch_size), n_train))

        self.model_.train()
        for epoch in range(int(self.max_epochs)):
            perm = torch.randperm(n_train, device=device)
            epoch_loss = 0.0
            n_seen = 0
            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                xb = X_tensor[idx]
                yb = y_tensor[idx]

                optimizer.zero_grad(set_to_none=True)
                out = self.model_(xb)
                if out.ndim == 3:
                    pred = out.squeeze(-1)
                    loss = ((pred - yb.squeeze(-1).unsqueeze(1)) ** 2).mean()
                else:
                    pred = out.reshape(yb.shape)
                    loss = ((pred - yb) ** 2).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()

                n_batch = len(idx)
                epoch_loss += float(loss.detach().cpu()) * n_batch
                n_seen += n_batch

            if self.verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f'TabM epoch {epoch + 1}/{self.max_epochs}, train_mse={epoch_loss / max(n_seen, 1):.6f}')
        return self

    def predict(self, X):
        try:
            import torch
        except ImportError as e:
            raise ImportError('tabm requires torch. Install it with: pip install torch') from e

        X_np = to_dense_float32(X)
        device = next(self.model_.parameters()).device
        batch_size = max(1, int(self.batch_size))
        preds = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, X_np.shape[0], batch_size):
                xb = torch.as_tensor(X_np[start:start + batch_size], dtype=torch.float32, device=device)
                out = self.model_(xb)
                if out.ndim == 3:
                    pred = out.squeeze(-1).mean(dim=1)
                else:
                    pred = out.reshape(-1)
                preds.append(pred.detach().cpu().numpy())
        preds_scaled = np.concatenate(preds).astype(np.float32).reshape(-1, 1)
        preds = self.y_scaler_.inverse_transform(preds_scaled)
        return preds.astype(float).reshape(-1)


class TabNetSklearnRegressor(BaseEstimator, RegressorMixin):
    """Internal benchmark helper."""
    def __init__(
        self,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        lambda_sparse=1e-3,
        learning_rate=2e-2,
        max_epochs=50,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        device_name='auto',
        verbose=0,
        random_state=None
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.device_name = device_name
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        if TabNetRegressorRaw is None:
            raise ImportError("pytorch-tabnet is not installed. Install it with: pip install pytorch-tabnet")

        try:
            import torch
        except ImportError as e:
            raise ImportError("pytorch-tabnet requires torch. Install it with: pip install torch") from e

        X_np = to_dense_float32(X)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        # -------------------------------
        # -------------------------------
        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y_np).astype(np.float32)

        candidate_params = {
            "n_d": self.n_d,
            "n_a": self.n_a,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "lambda_sparse": self.lambda_sparse,
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": dict(lr=self.learning_rate),
            "seed": 0 if self.random_state is None else self.random_state,
            "verbose": self.verbose,
            "device_name": self.device_name
        }

        model_params = filter_supported_params(TabNetRegressorRaw, candidate_params)
        self.model_ = TabNetRegressorRaw(**model_params)

        n_train = X_np.shape[0]

        if n_train < 2:
            raise ValueError("TabNet requires at least two training samples because BatchNorm cannot train on a single sample.")

        batch_size = max(2, min(int(self.batch_size), n_train))
        virtual_batch_size = choose_virtual_batch_size(batch_size, self.virtual_batch_size)

        fit_params = {
            "max_epochs": int(self.max_epochs),
            "patience": int(self.patience),
            "batch_size": batch_size,
            "virtual_batch_size": virtual_batch_size,
            "num_workers": 0,
            "drop_last": True,
            "compute_importance": False
        }

        fit_params = filter_supported_fit_params(self.model_.fit, fit_params)

        self.model_.fit(X_np, y_scaled, **fit_params)

        return self

    def predict(self, X):
        X_np = to_dense_float32(X)

        preds_scaled = self.model_.predict(X_np)
        preds_scaled = np.asarray(preds_scaled, dtype=np.float32).reshape(-1, 1)

        # -------------------------------
        # -------------------------------
        preds = self.y_scaler_.inverse_transform(preds_scaled)

        return np.asarray(preds, dtype=float).reshape(-1)


# -------------------------------------------------------------------
# -------------------------------------------------------------------

DATASET_CONFIGS = [
    {"name": "abalone", "train": "abalone.data", "test": None, "cat_cols": [0]},
    #{"name": "kin8nm", "train": "kin8nm.data", "test": None, "cat_cols": []}
]

MIN_POINTS_FOR_SPLIT_BASE = 5
N_REPS = 5
RANDOM_STATE = 42
DATA_DIR = Path(os.environ.get("DATA_DIR", "data/raw"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_PDF_NAME = str(OUTPUT_DIR / "benchmarking_results_paper_ready_fastv2.pdf")
FINAL_TEX_NAME = str(OUTPUT_DIR / "benchmarking_results_all_methods_fastv2_table5_xgb.tex")


def resolve_data_path(file_name):
    """Return a dataset path from DATA_DIR, while preserving absolute paths.

    The fallback to the original relative path keeps backward compatibility for
    users who keep dataset files beside this script.
    """
    path = Path(str(file_name))
    if path.is_absolute():
        return path
    candidate = DATA_DIR / path
    return candidate if candidate.exists() else path

# -------------------------
# Speed / hardware switches
# -------------------------
#   USE_GPU_ACCEL=1 N_JOBS=32 python scripts/run_benchmark.py
USE_GPU_ACCEL = os.environ.get("USE_GPU_ACCEL", "1").lower() not in {"0", "false", "no"}
N_JOBS = int(os.environ.get("N_JOBS", "-1"))
GRID_N_JOBS = int(os.environ.get("GRID_N_JOBS", str(N_JOBS)))
FINAL_MODEL_N_JOBS = int(os.environ.get("FINAL_MODEL_N_JOBS", str(N_JOBS)))
PIPELINE_CACHE_DIR = os.environ.get("PIPELINE_CACHE_DIR", str(OUTPUT_DIR / "sklearn_pipeline_cache"))
GRID_PRE_DISPATCH = os.environ.get("GRID_PRE_DISPATCH", "2*n_jobs")


# -------------------------
# Model and baseline switches
# -------------------------
# The official default evaluates HRT-Boost only. Use RUN_METHODS to add the
# single-tree HRT ablation or optional baselines, for example:
#   RUN_METHODS="HRT-Boost,HRT,CART,RF" python scripts/run_benchmark.py
#   RUN_METHODS="all,-TabM,-TabNet" python scripts/run_benchmark.py
ALL_METHODS_TO_SHOW = [
    "HRT",
    "HRT-Boost",
    "CART",
    "RF",
    "AdaBoost",
    "Scikit-GBM",
    "XGBoost",
    "LightGBM",
    "TabM",
    "TabNet"
]


def normalize_method_name(name):
    """Normalize user-facing method names for environment-variable parsing."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


METHOD_NAME_ALIASES = {
    normalize_method_name(name): name for name in ALL_METHODS_TO_SHOW
}
METHOD_NAME_ALIASES.update(
    {
        "hrt": "HRT",
        "singlehrt": "HRT",
        "singlehrttree": "HRT",
        "hrtboost": "HRT-Boost",
        "hrtboosting": "HRT-Boost",
        "gbm": "Scikit-GBM",
        "sklearngbm": "Scikit-GBM",
        "scikitgbm": "Scikit-GBM",
        "gradientboosting": "Scikit-GBM",
        "randomforest": "RF",
        "lightgbm": "LightGBM",
        "lgbm": "LightGBM",
        "xgb": "XGBoost",
        "xgboost": "XGBoost",
        "tabm": "TabM",
        "tabnet": "TabNet",
    }
)

METHOD_GROUPS = {
    "all": ALL_METHODS_TO_SHOW,
    "everything": ALL_METHODS_TO_SHOW,
    "baseline": [
        "CART",
        "RF",
        "AdaBoost",
        "Scikit-GBM",
        "XGBoost",
        "LightGBM",
        "TabM",
        "TabNet"
    ],
    "baselines": [
        "CART",
        "RF",
        "AdaBoost",
        "Scikit-GBM",
        "XGBoost",
        "LightGBM",
        "TabM",
        "TabNet"
    ],
    "hrt": ["HRT"],
    "hrtboost": ["HRT-Boost"],
    "official": ["HRT-Boost"],
    "ours": ["HRT-Boost", "HRT"],
    "classical": ["CART", "RF", "AdaBoost", "Scikit-GBM", "XGBoost", "LightGBM"],
    "tree": ["CART", "RF", "AdaBoost", "Scikit-GBM", "XGBoost", "LightGBM"],
    "trees": ["CART", "RF", "AdaBoost", "Scikit-GBM", "XGBoost", "LightGBM"],
    "deep": ["TabM", "TabNet"],
    "new": ["TabM"],
    "newbaselines": ["TabM"],
    "one": ["HRT-Boost"],
    "two": ["HRT-Boost", "HRT"],
}


def _append_unique(target, names):
    for name in names:
        if name not in target:
            target.append(name)


def _resolve_method_token(token):
    token = str(token).strip().strip('"\'')

    if token in {"*", "all", "ALL"}:
        return list(ALL_METHODS_TO_SHOW)

    key = normalize_method_name(token)

    if key == "all":
        return list(ALL_METHODS_TO_SHOW)

    if key in METHOD_GROUPS:
        return list(METHOD_GROUPS[key])

    if key in METHOD_NAME_ALIASES:
        return [METHOD_NAME_ALIASES[key]]

    raise ValueError(
        f"Unknown method/group: {token!r}. Available methods: {', '.join(ALL_METHODS_TO_SHOW)}. "
        "Available groups: official, hrt, hrtboost, baselines, classical/tree, deep, all."
    )


def parse_methods_to_run(raw_value):
    """Parse RUN_METHODS/RUN_BASELINES into an ordered method list.

    Examples
    --------
    - RUN_METHODS=HRT-Boost
    - RUN_METHODS=HRT-Boost,HRT,CART
    - RUN_METHODS=baselines
    - RUN_METHODS=all,-TabM,-TabNet
    - RUN_METHODS=deep,-TabNet
    """
    raw_value = "HRT-Boost" if raw_value is None else str(raw_value).strip()
    if raw_value == "":
        raw_value = "HRT-Boost"

    tokens = [t.strip() for t in raw_value.replace(";", ",").split(",") if t.strip()]
    include_tokens = []
    exclude_tokens = []

    for token in tokens:
        if token[0] in {"-", "!"}:
            exclude_tokens.append(token[1:].strip())
        elif token[0] == "+":
            include_tokens.append(token[1:].strip())
        else:
            include_tokens.append(token)

    selected = []

    if not include_tokens:
        selected = list(ALL_METHODS_TO_SHOW)
    else:
        for token in include_tokens:
            _append_unique(selected, _resolve_method_token(token))

    for token in exclude_tokens:
        for name in _resolve_method_token(token):
            if name in selected:
                selected.remove(name)

    if not selected:
        raise ValueError(
            "RUN_METHODS/RUN_BASELINES did not resolve to any runnable method. "
            "Keep at least one model selected."
        )

    return selected


RUN_METHODS_RAW = os.environ.get(
    "RUN_METHODS",
    os.environ.get("RUN_BASELINES", "HRT-Boost")
)
methods_to_show = parse_methods_to_run(RUN_METHODS_RAW)
METHODS_TO_RUN_SET = set(methods_to_show)

print(f"Selected methods/baselines: {', '.join(methods_to_show)}")


def should_run_method(name):
    return name in METHODS_TO_RUN_SET


def _run_quiet_command(cmd):
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        ).returncode == 0
    except Exception:
        return False


def torch_cuda_is_usable():
    """Internal benchmark helper."""
    if not USE_GPU_ACCEL:
        return False

    try:
        import torch
        if not torch.cuda.is_available():
            return False

        _ = torch.empty((1,), device="cuda")
        torch.cuda.synchronize()
        return True
    except Exception as exc:
        print(
            "Notice: PyTorch CUDA is not usable; TabNet will use CPU. "
            f"Reason: {type(exc).__name__}: {exc}"
        )
        return False


TORCH_CUDA_AVAILABLE = torch_cuda_is_usable()
NVIDIA_SMI_AVAILABLE = _run_quiet_command(["nvidia-smi"]) if USE_GPU_ACCEL else False
GPU_AVAILABLE = TORCH_CUDA_AVAILABLE
#print(
#    f"GPU acceleration requested: {USE_GPU_ACCEL}; "
#    f"torch CUDA usable: {TORCH_CUDA_AVAILABLE}; "
#    f"nvidia-smi visible: {NVIDIA_SMI_AVAILABLE}"
#)


def safe_set_params(estimator, **params):
    """Internal benchmark helper."""
    try:
        supported = estimator.get_params(deep=False)
        valid = {k: v for k, v in params.items() if k in supported}
        if valid:
            estimator.set_params(**valid)
    except Exception:
        pass
    return estimator


def apply_final_runtime_params(estimator):
    """Internal benchmark helper."""
    safe_set_params(estimator, n_jobs=FINAL_MODEL_N_JOBS)
    safe_set_params(estimator, thread_count=FINAL_MODEL_N_JOBS)
    return estimator


def _xgb_gpu_params_for_version():
    try:
        import xgboost as xgb
        version_major = int(str(xgb.__version__).split(".")[0])
    except Exception:
        version_major = 2

    if version_major >= 2:
        return {"tree_method": "hist", "device": "cuda"}

    return {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}


def xgboost_gpu_is_usable():
    if not (USE_GPU_ACCEL and XGBRegressor is not None):
        return False

    try:
        X_probe = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y_probe = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        XGBRegressor(
            n_estimators=1,
            learning_rate=0.1,
            max_depth=1,
            n_jobs=1,
            random_state=RANDOM_STATE,
            verbosity=0,
            **_xgb_gpu_params_for_version()
        ).fit(X_probe, y_probe)
        return True
    except Exception as exc:
        print(f"text:XGBoost GPU text,text CPU.Reason: {type(exc).__name__}: {exc}")
        return False


def xgb_runtime_params():
    """Internal benchmark helper."""
    params = {
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "tree_method": "hist",
        "max_bin": XGBOOST_DEFAULT_MAX_BIN,
        "verbosity": 0
    }

    if XGBOOST_GPU_AVAILABLE:
        params.update(_xgb_gpu_params_for_version())

    return params


def lightgbm_gpu_is_usable():
    if not (USE_GPU_ACCEL and LGBMRegressor is not None):
        return False
    try:
        X_probe = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y_probe = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        LGBMRegressor(
            n_estimators=1,
            learning_rate=0.1,
            device_type="gpu",
            verbosity=-1,
            n_jobs=1,
            random_state=RANDOM_STATE
        ).fit(X_probe, y_probe)
        return True
    except Exception as exc:
        print(f"text:LightGBM GPU text,text CPU.Reason: {type(exc).__name__}: {exc}")
        return False


XGBOOST_GPU_AVAILABLE = xgboost_gpu_is_usable()
LIGHTGBM_GPU_AVAILABLE = lightgbm_gpu_is_usable()


def make_pipeline(preprocessor, estimator, memory=None):
    """Internal benchmark helper."""
    return Pipeline(
        [
            ('pre', preprocessor),
            ('reg', estimator)
        ],
        memory=memory
    )


# -------------------------------------------------------------------
# -------------------------------------------------------------------

class Node:
    def __init__(self, is_leaf, region, params=None, split_coeffs=None, children=None, stop_reason=None):
        self.is_leaf = is_leaf
        self.region = region
        self.params = params
        self.split_coeffs = split_coeffs
        self.children = children
        self.stop_reason = stop_reason

    def count_leaves(self):
        if self.is_leaf:
            return 1
        return sum(child.count_leaves() for child in self.children)


def solve_ols_nd(X_des, Z, alpha=0):
    """Internal benchmark helper."""
    n_pts, n_feats_plus_1 = X_des.shape

    if n_pts < n_feats_plus_1:
        return np.append(
            np.zeros(n_feats_plus_1 - 1),
            np.mean(Z) if n_pts > 0 else 0.0
        )

    XtX = X_des.T @ X_des
    XtZ = X_des.T @ Z

    if alpha > 0:
        I = np.eye(n_feats_plus_1)
        I[-1, -1] = 0.0
        XtX += alpha * I

    try:
        return np.linalg.solve(XtX, XtZ)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(XtX, XtZ, rcond=None)[0]


def initialize_thetas_nd(X_des, Z, seed=None, ridge_alpha=0):
    n_pts, n_dim = X_des.shape
    n_feats = n_dim - 1
    mean_z = np.mean(Z) if n_pts > 0 else 0

    if n_pts < n_dim:
        t = np.append(np.zeros(n_feats), mean_z)
        return t, t.copy()

    rng = np.random.RandomState(seed) if seed is not None else np.random

    chosen_dim = 0

    for d in range(n_feats):
        if np.max(X_des[:, d]) > np.min(X_des[:, d]):
            chosen_dim = d
            break

    median_val = np.median(X_des[:, chosen_dim])
    m_l = X_des[:, chosen_dim] < median_val

    if np.sum(m_l) < n_dim or np.sum(~m_l) < n_dim:
        t_glob = solve_ols_nd(X_des, Z, alpha=ridge_alpha)
        eps = rng.rand(n_dim) * 0.005
        return t_glob + eps, t_glob - eps

    return (
        solve_ols_nd(X_des[m_l], Z[m_l], alpha=ridge_alpha),
        solve_ols_nd(X_des[~m_l], Z[~m_l], alpha=ridge_alpha)
    )


def _rmse_eval(X_des, Z, t1, t2, flip, min_seg, min_fit):
    sv = (t2 - t1) if flip else (t1 - t2)
    m = (X_des @ sv) < 0
    sum_m = np.sum(m)

    if (
        sum_m < min_seg
        or (len(Z) - sum_m) < min_seg
        or sum_m < min_fit
        or (len(Z) - sum_m) < min_fit
    ):
        return False, np.inf, m

    p = np.empty_like(Z)
    p[m] = X_des[m] @ t1
    p[~m] = X_des[~m] @ t2

    return True, np.sqrt(np.mean((Z - p) ** 2)), m


def _run_split_dir(
    X_des,
    Z,
    t1_i,
    t2_i,
    max_iter,
    tol,
    min_seg,
    min_fit,
    step_size,
    ridge,
    flip
):
    t1, t2 = t1_i.copy(), t2_i.copy()
    prev_m, cur_rmse = None, np.inf

    v, r, _ = _rmse_eval(X_des, Z, t1, t2, flip, min_seg, min_fit)

    if v:
        cur_rmse = r
    else:
        return None, 0

    for i in range(max_iter):
        sv = (t2 - t1) if flip else (t1 - t2)
        m = (X_des @ sv) < 0

        if prev_m is not None and np.array_equal(m, prev_m):
            break

        prev_m = m
        sum_m = np.sum(m)

        if sum_m < min_fit or (len(Z) - sum_m) < min_fit:
            break

        try:
            t1_t = solve_ols_nd(X_des[m], Z[m], alpha=ridge)
            t2_t = solve_ols_nd(X_des[~m], Z[~m], alpha=ridge)

            if isinstance(step_size, (int, float)):
                t1 = (1 - step_size) * t1 + step_size * t1_t
                t2 = (1 - step_size) * t2 + step_size * t2_t
            else:
                s, accepted = 1.0, False

                for _ in range(5):
                    t1_try = (1 - s) * t1 + s * t1_t
                    t2_try = (1 - s) * t2 + s * t2_t

                    v_tr, r_tr, _ = _rmse_eval(
                        X_des,
                        Z,
                        t1_try,
                        t2_try,
                        flip,
                        min_seg,
                        min_fit
                    )

                    if v_tr and r_tr < cur_rmse - 1e-8:
                        t1, t2, cur_rmse, accepted = t1_try, t2_try, r_tr, True
                        break

                    s *= 0.5

                if not accepted:
                    break

            if np.linalg.norm(t1 - t1_t) < tol:
                break

        except Exception:
            break

    res_v, res_r, res_m = _rmse_eval(X_des, Z, t1, t2, flip, min_seg, min_fit)

    if res_v:
        sc = (t2 - t1) if flip else (t1 - t2)
        return (sc, t1, t2, res_m, ~res_m, res_r), i + 1

    return None, i + 1


def optimize_split_aligned(X_des, Z, min_seg, step, seed, ridge):
    n_pts, n_dim = X_des.shape
    min_fit = n_dim

    if n_pts < 2 * max(min_fit, min_seg):
        return None, None, None, None, None, 0

    t1_i, t2_i = initialize_thetas_nd(X_des, Z, seed=seed, ridge_alpha=ridge)

    best_res, best_rmse, best_it = None, float('inf'), 0

    for flip in [False, True]:
        res, it = _run_split_dir(
            X_des,
            Z,
            t1_i,
            t2_i,
            50,
            1e-5,
            min_seg,
            min_fit,
            step,
            ridge,
            flip
        )

        if res and res[-1] < best_rmse:
            best_rmse, best_res, best_it = res[-1], res, it

    return (*best_res[:-1], best_it) if best_res else (None, None, None, None, None, best_it)


def recursive_fit_aligned(
    X_des_f,
    Z_f,
    idx,
    threshold,
    min_seg,
    depth,
    max_depth,
    seed,
    step,
    ridge,
    iter_list
):
    X_slice = X_des_f[idx]
    Z_slice = Z_f[idx]
    n_pts, n_dim = X_slice.shape

    region = []

    if len(idx) < n_dim + 1:
        params = solve_ols_nd(X_slice, Z_slice, alpha=ridge)
        return Node(True, region, params=params, stop_reason='small')

    t_s = solve_ols_nd(X_slice, Z_slice, alpha=ridge)

    if depth >= max_depth:
        return Node(True, region, params=t_s, stop_reason='depth')

    split = None

    for att in range(2):
        res = optimize_split_aligned(
            X_slice,
            Z_slice,
            min_seg,
            step,
            (seed or 0) + depth * 100 + att,
            ridge
        )

        if res[0] is not None:
            split = res
            break

    if split:
        sc, t1, t2, ml, mr, iters = split
        iter_list.append(iters)

        return Node(
            False,
            region,
            split_coeffs=sc,
            children=[
                recursive_fit_aligned(
                    X_des_f,
                    Z_f,
                    idx[ml],
                    threshold,
                    min_seg,
                    depth + 1,
                    max_depth,
                    seed,
                    step,
                    ridge,
                    iter_list
                ),
                recursive_fit_aligned(
                    X_des_f,
                    Z_f,
                    idx[mr],
                    threshold,
                    min_seg,
                    depth + 1,
                    max_depth,
                    seed,
                    step,
                    ridge,
                    iter_list
                )
            ]
        )

    return Node(True, region, params=t_s, stop_reason='fail')


class HRTRegressor(BaseEstimator, RegressorMixin):
    """Single Hinge Regression Tree used as the HRT-Boost base learner.

    The split search follows the HRT reference implementation at
    https://github.com/Hongyi-Li-sz/Hinge-Regression-Tree.
    """

    def __init__(
        self,
        threshold=0,
        min_points=5,
        max_depth=5,
        step_size='auto',
        ridge_alpha=1.0,
        random_state=None
    ):
        self.threshold = threshold
        self.min_points = min_points
        self.max_depth = max_depth
        self.step_size = step_size
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state

    def fit(self, X, y):
        self.iter_counts_ = []

        X_des = np.column_stack([X, np.ones(X.shape[0])])

        self.root_node = recursive_fit_aligned(
            X_des,
            y.astype(float),
            np.arange(len(y)),
            self.threshold,
            self.min_points,
            0,
            self.max_depth,
            self.random_state,
            self.step_size,
            self.ridge_alpha,
            self.iter_counts_
        )

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        X_des = np.column_stack([X, np.ones(n_samples)])
        y_pred = np.zeros(n_samples)

        stack = [(self.root_node, np.arange(n_samples))]

        while stack:
            node, indices = stack.pop()

            if node.is_leaf:
                y_pred[indices] = X_des[indices] @ node.params
            else:
                mask = (X_des[indices] @ node.split_coeffs) < -1e-9
                left_idx = indices[mask]
                right_idx = indices[~mask]

                if len(left_idx) > 0:
                    stack.append((node.children[0], left_idx))

                if len(right_idx) > 0:
                    stack.append((node.children[1], right_idx))

        return y_pred


class HRTBoostingRegressor(BaseEstimator, RegressorMixin):
    """Residual boosting estimator with HRT trees as weak learners."""

    def __init__(
        self,
        n_estimators=50,
        learning_rate=0.1,
        max_depth=2,
        step_size='auto',
        ridge_alpha=1.0,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.step_size = step_size
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state

    def fit(self, X, y):
        self.models_ = []
        self.initial_pred_ = np.mean(y)

        curr_p = np.full(X.shape[0], self.initial_pred_, dtype=float)
        y_float = y.astype(float)

        for i in range(self.n_estimators):
            res = y_float - curr_p

            m = HRTRegressor(
                max_depth=self.max_depth,
                step_size=self.step_size,
                ridge_alpha=self.ridge_alpha,
                random_state=(self.random_state + i if self.random_state else None)
            )

            m.fit(X, res)

            preds = m.predict(X)
            curr_p += self.learning_rate * preds
            self.models_.append(m)

        return self

    def predict(self, X):
        p = np.full(X.shape[0], self.initial_pred_)

        for m in self.models_:
            p += self.learning_rate * m.predict(X)

        return p


# -------------------------------------------------------------------
# Model complexity utilities
# -------------------------------------------------------------------
def mean_std_ignore_nan(values):
    """Internal benchmark helper."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr))


def estimate_model_size_mb(model_obj):
    """Internal benchmark helper."""
    try:
        return len(pickle.dumps(model_obj, protocol=pickle.HIGHEST_PROTOCOL)) / (1024.0 ** 2)
    except Exception:
        return np.nan


SKLEARN_TREE_INTERNAL_PARAM_COUNT = 2.0   # feature/index + threshold/category decision
SKLEARN_TREE_SPLIT_FLOPS = 3.0           # feature access + threshold test + branch selection
SKLEARN_SPLIT_SEARCH_FLOPS_PER_SAMPLE_FEATURE = 8.0
SKLEARN_NODE_PARTITION_FLOPS_PER_SAMPLE = 2.0
SKLEARN_LEAF_VALUE_FLOPS_PER_SAMPLE = 2.0

LIGHTGBM_INTERNAL_PARAM_COUNT = 3.0      # feature + threshold/category decision + default/missing direction
LIGHTGBM_SPLIT_FLOPS = 4.0              # missing check + threshold/category test + branch selection
LIGHTGBM_GRAD_HESS_FLOPS_PER_SAMPLE = 8.0
LIGHTGBM_HIST_BUILD_FLOPS_PER_SAMPLE_FEATURE_LEVEL = 10.0
LIGHTGBM_SPLIT_SCAN_FLOPS_PER_BIN_FEATURE_NODE = 5.0
LIGHTGBM_LEAF_WEIGHT_FLOPS = 8.0
LIGHTGBM_DEFAULT_MAX_BIN = 255



XGBOOST_INTERNAL_PARAM_COUNT = 4.0
XGBOOST_SPLIT_FLOPS = 6.0
XGBOOST_ENSEMBLE_ADD_FLOPS = 2.0
ENSEMBLE_ADD_FLOPS = 2.0
XGBOOST_GRAD_HESS_FLOPS_PER_SAMPLE = 12.0
XGBOOST_HIST_BUILD_FLOPS_PER_SAMPLE_FEATURE_LEVEL = 24.0
XGBOOST_SPLIT_SCAN_FLOPS_PER_BIN_FEATURE_NODE = 28.0
XGBOOST_LEAF_WEIGHT_FLOPS = 32.0
XGBOOST_DEFAULT_MAX_BIN = 256


NEURAL_MAC_FLOPS = 2.0
TABM_ELEMENTWISE_FLOPS_PER_HIDDEN = 10.0
TABNET_DEFAULT_N_SHARED = 2
TABNET_DEFAULT_N_INDEPENDENT = 2
TABNET_ELEMENTWISE_FLOPS_PER_HIDDEN = 8.0
TABNET_MASK_FLOPS_PER_FEATURE_STEP = 8.0
ADAMW_UPDATE_FLOPS_PER_PARAM = 12.0


def _as_finite_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def dot_product_flops(n_coeffs, include_comparison=False):
    """Internal benchmark helper."""
    try:
        n_coeffs = int(n_coeffs)
    except Exception:
        return np.nan
    if n_coeffs <= 0:
        return np.nan
    flops = n_coeffs + max(n_coeffs - 1, 0)
    if include_comparison:
        flops += 1
    return float(flops)


def ols_training_flops(n_samples, n_coeffs):
    """Internal benchmark helper."""
    try:
        n = int(n_samples)
        p = int(n_coeffs)
    except Exception:
        return np.nan
    if n <= 0 or p <= 0:
        return 0.0
    if n < p:
        return float(max(n, 1))
    return float((2.0 * n * p * p) + (2.0 * n * p) + ((2.0 / 3.0) * p ** 3))


def count_hrt_node_parameters(node):
    """Internal benchmark helper."""
    if node is None:
        return np.nan
    if node.is_leaf:
        return int(np.size(node.params)) if node.params is not None else 0
    n_params = int(np.size(node.split_coeffs)) if node.split_coeffs is not None else 0
    return n_params + sum(count_hrt_node_parameters(child) for child in node.children)


def hrt_leaf_depth_stats(node, depth=0):
    """Internal benchmark helper."""
    if node is None:
        return 0.0, 0
    if node.is_leaf:
        return float(depth), 1
    depth_sum, leaf_count = 0.0, 0
    for child in node.children:
        d_sum, n_leaf = hrt_leaf_depth_stats(child, depth + 1)
        depth_sum += d_sum
        leaf_count += n_leaf
    return depth_sum, leaf_count


def _hrt_path_flops_sum(node, prefix_flops=0.0):
    """Internal benchmark helper."""
    if node is None:
        return 0.0, 0
    if node.is_leaf:
        leaf_params = int(np.size(node.params)) if node.params is not None else 0
        leaf_flops = dot_product_flops(leaf_params, include_comparison=False)
        if np.isnan(leaf_flops):
            leaf_flops = 0.0
        return float(prefix_flops + leaf_flops), 1

    split_params = int(np.size(node.split_coeffs)) if node.split_coeffs is not None else 0
    split_flops = dot_product_flops(split_params, include_comparison=True)
    if np.isnan(split_flops):
        split_flops = 0.0

    total, count = 0.0, 0
    for child in node.children:
        c_total, c_count = _hrt_path_flops_sum(child, prefix_flops + split_flops)
        total += c_total
        count += c_count
    return total, count


def estimate_hrt_node_flops(node):
    """Internal benchmark helper."""
    if node is None:
        return np.nan
    total, count = _hrt_path_flops_sum(node, 0.0)
    if count <= 0:
        return np.nan
    return float(total / count)


def estimate_hrt_training_flops_from_data(node, X_train_processed, iter_counts=None):
    """Internal benchmark helper."""
    if node is None or X_train_processed is None:
        return np.nan
    try:
        X_np = to_dense_float32(X_train_processed)
        X_des = np.column_stack([X_np, np.ones(X_np.shape[0], dtype=np.float32)])
    except Exception:
        return np.nan

    iter_values = list(iter_counts or [])
    iter_pos = [0]
    p = X_des.shape[1]

    def next_iter_count():
        if iter_pos[0] >= len(iter_values):
            return 1
        value = iter_values[iter_pos[0]]
        iter_pos[0] += 1
        try:
            return max(1, int(value))
        except Exception:
            return 1

    def rec(cur_node, indices):
        n = len(indices)
        total = ols_training_flops(n, p)
        if cur_node is None or cur_node.is_leaf:
            return total

        try:
            split_cost_per_sample = dot_product_flops(np.size(cur_node.split_coeffs), include_comparison=True)
            if np.isnan(split_cost_per_sample):
                split_cost_per_sample = 0.0
            total += n * split_cost_per_sample
            mask = (X_des[indices] @ cur_node.split_coeffs) < -1e-9
            left_idx = indices[mask]
            right_idx = indices[~mask]
        except Exception:
            left_idx = indices[: n // 2]
            right_idx = indices[n // 2:]

        iters = next_iter_count()
        total += iters * (
            ols_training_flops(len(left_idx), p)
            + ols_training_flops(len(right_idx), p)
        )

        children = cur_node.children or []
        if len(children) >= 1:
            total += rec(children[0], left_idx)
        if len(children) >= 2:
            total += rec(children[1], right_idx)
        return total

    return float(rec(node, np.arange(X_des.shape[0])))


def sklearn_tree_counts(tree_estimator):
    try:
        tree = tree_estimator.tree_
        children_left = tree.children_left
        children_right = tree.children_right
        is_leaf = children_left == children_right
        leaf_count = int(np.sum(is_leaf))
        internal_count = int(tree.node_count - leaf_count)
        return internal_count, leaf_count, int(tree.node_count)
    except Exception:
        return np.nan, np.nan, np.nan


def sklearn_tree_effective_feature_count(tree_estimator, n_features):
    """Internal benchmark helper."""
    try:
        n_features = max(1, int(n_features))
    except Exception:
        n_features = 1
    try:
        value = getattr(tree_estimator, 'max_features_', None)
        if value is None:
            value = getattr(tree_estimator, 'n_features_in_', n_features)
        return max(1, min(int(value), n_features))
    except Exception:
        return n_features


def sklearn_tree_parameter_count(tree_estimator):
    """Internal benchmark helper."""
    try:
        tree = tree_estimator.tree_
        internal_count, leaf_count, _ = sklearn_tree_counts(tree_estimator)
        output_dim = int(np.prod(tree.value.shape[1:])) if hasattr(tree, 'value') else 1
        return int(SKLEARN_TREE_INTERNAL_PARAM_COUNT * internal_count + leaf_count * max(output_dim, 1))
    except Exception:
        return np.nan


def sklearn_tree_leaf_depths(tree_estimator):
    """Internal benchmark helper."""
    try:
        tree = tree_estimator.tree_
        children_left = tree.children_left
        children_right = tree.children_right
        stack = [(0, 0)]
        leaf_depths = []
        while stack:
            node_id, depth = stack.pop()
            if children_left[node_id] == children_right[node_id]:
                leaf_depths.append(depth)
            else:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
        return np.asarray(leaf_depths, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def sklearn_tree_avg_leaf_depth(tree_estimator):
    """Internal benchmark helper."""
    leaf_depths = sklearn_tree_leaf_depths(tree_estimator)
    return float(np.mean(leaf_depths)) if leaf_depths.size else np.nan


def sklearn_tree_inference_flops(tree_estimator):
    """Internal benchmark helper."""
    avg_depth = sklearn_tree_avg_leaf_depth(tree_estimator)
    if np.isnan(avg_depth):
        return np.nan
    return float(avg_depth * SKLEARN_TREE_SPLIT_FLOPS)


def sklearn_tree_training_flops(tree_estimator, n_features, sample_fraction=1.0):
    """
    
            effective_features * n_node * log2(n_node + 1)
            """
    try:
        tree = tree_estimator.tree_
        n_features = max(1, int(n_features))
        effective_features = sklearn_tree_effective_feature_count(tree_estimator, n_features)
        sample_fraction = min(max(float(sample_fraction), 1e-6), 1.0)
        children_left = tree.children_left
        children_right = tree.children_right
        n_node_samples = tree.n_node_samples
        total = 0.0
        for node_id in range(tree.node_count):
            n_node = max(1.0, float(n_node_samples[node_id]) * sample_fraction)
            if children_left[node_id] != children_right[node_id]:
                total += (
                    SKLEARN_SPLIT_SEARCH_FLOPS_PER_SAMPLE_FEATURE
                    * effective_features
                    * n_node
                    * np.log2(n_node + 1.0)
                )
                total += SKLEARN_NODE_PARTITION_FLOPS_PER_SAMPLE * n_node
            else:
                total += SKLEARN_LEAF_VALUE_FLOPS_PER_SAMPLE * n_node
        return float(total)
    except Exception:
        return np.nan

def xgboost_dump_counts_and_depth(dump_text):
    """Internal benchmark helper."""
    lines = [line for line in str(dump_text).splitlines() if line.strip()]
    leaf_depths = [line.count('\t') for line in lines if 'leaf=' in line]
    leaf_count = len(leaf_depths)
    internal_count = max(len(lines) - leaf_count, 0)
    avg_depth = float(np.mean(leaf_depths)) if leaf_depths else np.nan
    max_depth = float(np.max(leaf_depths)) if leaf_depths else np.nan
    return internal_count, leaf_count, avg_depth, max_depth


def estimate_xgboost_training_flops(
    internal_counts,
    leaf_counts,
    max_depths,
    best_param_dict,
    n_train,
    n_features
):
    """Internal benchmark helper."""
    try:
        n_trees = int(len(leaf_counts))
        if n_trees <= 0:
            return np.nan

        n_train = max(1.0, float(n_train))
        n_features = max(1.0, float(n_features))
        internal_total = max(0.0, float(np.nansum(internal_counts)))
        leaf_total = max(0.0, float(np.nansum(leaf_counts)))

        subsample = float(best_param_dict.get('subsample', 1.0) or 1.0)
        colsample = float(best_param_dict.get('colsample_bytree', 1.0) or 1.0)
        subsample = min(max(subsample, 1e-6), 1.0)
        colsample = min(max(colsample, 1e-6), 1.0)

        effective_n = max(1.0, n_train * subsample)
        effective_features = max(1.0, n_features * colsample)

        inferred_depth = float(np.nanmax(max_depths)) if len(max_depths) else 1.0
        max_depth = float(best_param_dict.get('max_depth', inferred_depth) or inferred_depth or 1.0)
        max_depth = max(1.0, max_depth)

        max_bin = int(best_param_dict.get('max_bin', XGBOOST_DEFAULT_MAX_BIN) or XGBOOST_DEFAULT_MAX_BIN)
        max_bin = max(2, max_bin)

        grad_hess_cost = XGBOOST_GRAD_HESS_FLOPS_PER_SAMPLE * n_train * n_trees
        histogram_cost = (
            XGBOOST_HIST_BUILD_FLOPS_PER_SAMPLE_FEATURE_LEVEL
            * effective_n
            * effective_features
            * max_depth
            * n_trees
        )
        split_scan_cost = (
            XGBOOST_SPLIT_SCAN_FLOPS_PER_BIN_FEATURE_NODE
            * max_bin
            * effective_features
            * internal_total
        )
        routing_cost = XGBOOST_SPLIT_FLOPS * effective_n * max_depth * n_trees
        leaf_update_cost = XGBOOST_LEAF_WEIGHT_FLOPS * leaf_total

        return float(grad_hess_cost + histogram_cost + split_scan_cost + routing_cost + leaf_update_cost)
    except Exception:
        return np.nan


def lightgbm_tree_counts(node):
    if not isinstance(node, dict):
        return 0, 0
    if 'leaf_value' in node:
        return 0, 1
    li, ll = lightgbm_tree_counts(node.get('left_child'))
    ri, rl = lightgbm_tree_counts(node.get('right_child'))
    return 1 + li + ri, ll + rl


def lightgbm_leaf_depth_stats(node, depth=0):
    if not isinstance(node, dict):
        return 0.0, 0, 0.0
    if 'leaf_value' in node:
        return float(depth), 1, float(depth)
    left_sum, left_count, left_max = lightgbm_leaf_depth_stats(node.get('left_child'), depth + 1)
    right_sum, right_count, right_max = lightgbm_leaf_depth_stats(node.get('right_child'), depth + 1)
    return left_sum + right_sum, left_count + right_count, max(left_max, right_max)


def estimate_lightgbm_training_flops(
    internal_counts,
    leaf_counts,
    max_depths,
    best_param_dict,
    n_train,
    n_features
):
    """Internal benchmark helper."""
    try:
        n_trees = int(len(leaf_counts))
        if n_trees <= 0:
            return np.nan

        n_train = max(1.0, float(n_train))
        n_features = max(1.0, float(n_features))
        internal_total = max(0.0, float(np.nansum(internal_counts)))
        leaf_total = max(0.0, float(np.nansum(leaf_counts)))

        subsample = best_param_dict.get('subsample', best_param_dict.get('bagging_fraction', 1.0))
        feature_fraction = best_param_dict.get('feature_fraction', best_param_dict.get('colsample_bytree', 1.0))
        subsample = min(max(float(subsample or 1.0), 1e-6), 1.0)
        feature_fraction = min(max(float(feature_fraction or 1.0), 1e-6), 1.0)

        effective_n = max(1.0, n_train * subsample)
        effective_features = max(1.0, n_features * feature_fraction)

        finite_depths = [float(d) for d in max_depths if d is not None and np.isfinite(d)]
        inferred_depth = max(finite_depths) if finite_depths else 1.0
        max_depth_param = best_param_dict.get('max_depth', inferred_depth)
        max_depth = float(max_depth_param if max_depth_param not in {None, -1} else inferred_depth)
        if max_depth <= 0:
            max_depth = inferred_depth
        max_depth = max(1.0, max_depth)

        max_bin = int(best_param_dict.get('max_bin', LIGHTGBM_DEFAULT_MAX_BIN) or LIGHTGBM_DEFAULT_MAX_BIN)
        max_bin = max(2, max_bin)

        grad_hess_cost = LIGHTGBM_GRAD_HESS_FLOPS_PER_SAMPLE * n_train * n_trees
        histogram_cost = (
            LIGHTGBM_HIST_BUILD_FLOPS_PER_SAMPLE_FEATURE_LEVEL
            * effective_n
            * effective_features
            * max_depth
            * n_trees
        )
        split_scan_cost = (
            LIGHTGBM_SPLIT_SCAN_FLOPS_PER_BIN_FEATURE_NODE
            * max_bin
            * effective_features
            * internal_total
        )
        routing_cost = LIGHTGBM_SPLIT_FLOPS * effective_n * max_depth * n_trees
        leaf_update_cost = LIGHTGBM_LEAF_WEIGHT_FLOPS * leaf_total

        return float(grad_hess_cost + histogram_cost + split_scan_cost + routing_cost + leaf_update_cost)
    except Exception:
        return np.nan


def count_torch_parameters_from_obj(model_obj):
    """Internal benchmark helper."""
    candidates = [model_obj]
    for attr in ('model_', 'network', 'model', 'regressor_', 'estimator_', 'net_', 'network_'):
        if hasattr(model_obj, attr):
            candidates.append(getattr(model_obj, attr))
    if hasattr(model_obj, 'model_'):
        inner = getattr(model_obj, 'model_')
        for attr in ('network', 'model', 'net_', 'network_', 'transformer', 'encoder'):
            if hasattr(inner, attr):
                candidates.append(getattr(inner, attr))
    for candidate in candidates:
        try:
            if candidate is not None and hasattr(candidate, 'parameters'):
                return int(sum(p.numel() for p in candidate.parameters()))
        except Exception:
            pass
    return np.nan


def _fallback_parameter_count_from_size(model_size_mb):
    """Internal benchmark helper."""
    if model_size_mb is None or np.isnan(model_size_mb):
        return np.nan
    return float(model_size_mb * (1024.0 ** 2) / 4.0)


def _safe_int_from_params(params, key, default):
    try:
        value = params.get(key, default)
        if value is None:
            value = default
        return max(1, int(value))
    except Exception:
        return max(1, int(default))


def _safe_float_from_params(params, key, default):
    try:
        value = params.get(key, default)
        if value is None:
            value = default
        return float(value)
    except Exception:
        return float(default)


def estimate_tabm_inference_flops(n_params, best_param_dict, n_features):
    """Internal benchmark helper."""
    try:
        n_params = float(n_params)
        n_features = max(1.0, float(n_features or 1))
        k = _safe_int_from_params(best_param_dict, 'k', 16)
        n_blocks = _safe_int_from_params(best_param_dict, 'n_blocks', 2)
        d_block = _safe_int_from_params(best_param_dict, 'd_block', 128)
        d_out = 1.0

        param_floor = 2.5 * n_params

        input_projection = k * NEURAL_MAC_FLOPS * n_features * d_block
        block_mlp = k * n_blocks * (2.0 * NEURAL_MAC_FLOPS * d_block * d_block)
        block_elementwise = k * (n_blocks + 1) * TABM_ELEMENTWISE_FLOPS_PER_HIDDEN * d_block
        output_projection = k * NEURAL_MAC_FLOPS * d_block * d_out
        head_aggregation = max(k - 1, 0) + 1.0

        arch_estimate = (
            input_projection
            + block_mlp
            + block_elementwise
            + output_projection
            + head_aggregation
        )

        return float(max(param_floor, arch_estimate))
    except Exception:
        return np.nan


def estimate_tabnet_inference_flops(n_params, best_param_dict, n_features):
    """
    
                step-wise attention/mask/elementwise costs.
    """
    try:
        n_params = float(n_params)
        n_features = max(1.0, float(n_features or 1))
        n_steps = _safe_int_from_params(best_param_dict, 'n_steps', 3)
        n_d = _safe_int_from_params(best_param_dict, 'n_d', 8)
        n_a = _safe_int_from_params(best_param_dict, 'n_a', 8)
        dim = max(1.0, float(n_d + n_a))

        n_shared = _safe_int_from_params(best_param_dict, 'n_shared', TABNET_DEFAULT_N_SHARED)
        n_independent = _safe_int_from_params(best_param_dict, 'n_independent', TABNET_DEFAULT_N_INDEPENDENT)

        param_floor = NEURAL_MAC_FLOPS * n_params

        first_shared_glu = NEURAL_MAC_FLOPS * n_features * (2.0 * dim)
        later_shared_glu = max(n_shared - 1, 0) * NEURAL_MAC_FLOPS * dim * (2.0 * dim)
        shared_reuse = max(n_steps - 1, 0) * (first_shared_glu + later_shared_glu)

        attention_linear = n_steps * NEURAL_MAC_FLOPS * n_a * n_features
        mask_and_sparsemax = n_steps * TABNET_MASK_FLOPS_PER_FEATURE_STEP * n_features
        glu_bn_elementwise = n_steps * (n_shared + n_independent) * TABNET_ELEMENTWISE_FLOPS_PER_HIDDEN * dim
        decision_aggregation = n_steps * (2.0 * n_d)

        arch_estimate = (
            param_floor
            + shared_reuse
            + attention_linear
            + mask_and_sparsemax
            + glu_bn_elementwise
            + decision_aggregation
        )

        return float(max(param_floor, arch_estimate))
    except Exception:
        return np.nan


def estimate_neural_inference_flops(name, n_params, best_param_dict, n_train_samples, n_features=None):
    if n_params is None or np.isnan(n_params):
        return np.nan

    n_params = float(n_params)
    n_train = max(1, int(n_train_samples or 1))
    n_feat = max(1, int(n_features or 1))
    n_estimators = int(best_param_dict.get('n_estimators', getattr(best_param_dict, 'n_estimators', 1)) or 1)

    if name == "TabNet":
        return estimate_tabnet_inference_flops(n_params, best_param_dict, n_feat)

    if name == "TabM":
        return estimate_tabm_inference_flops(n_params, best_param_dict, n_feat)

    return float(2.0 * n_params)


def estimate_neural_training_flops(name, inference_flops_per_sample, best_param_dict, n_train_samples, n_features, n_params=None):
    if inference_flops_per_sample is None or np.isnan(inference_flops_per_sample):
        return np.nan

    n_train = max(1, int(n_train_samples or 1))
    n_features = max(1, int(n_features or 1))

    epochs = int(best_param_dict.get('max_epochs', getattr(best_param_dict, 'max_epochs', 1)) or 1)
    batch_size = int(best_param_dict.get('batch_size', getattr(best_param_dict, 'batch_size', 256)) or 256)
    steps_per_epoch = int(np.ceil(n_train / max(1, batch_size)))

    optimizer_cost = 0.0
    if n_params is not None and not np.isnan(n_params):
        optimizer_cost = ADAMW_UPDATE_FLOPS_PER_PARAM * float(n_params) * steps_per_epoch

    return float(max(1, epochs) * (3.0 * inference_flops_per_sample * n_train + optimizer_cost))

def _get_n_train_and_features(X_train_processed, n_train_samples, n_features):
    if X_train_processed is not None:
        try:
            return int(X_train_processed.shape[0]), int(X_train_processed.shape[1])
        except Exception:
            pass
    return int(n_train_samples or 1), int(n_features or 1)


def estimate_model_complexity(name, model_obj, best_param_dict=None, X_train_processed=None, n_train_samples=None, n_features=None):
    """Internal benchmark helper."""
    best_param_dict = best_param_dict or {}
    n_train, n_feat = _get_n_train_and_features(X_train_processed, n_train_samples, n_features)
    model_size_mb = estimate_model_size_mb(model_obj)
    n_params, inference_flops, training_flops = np.nan, np.nan, np.nan

    try:
        if name == "HRT":
            n_params = count_hrt_node_parameters(model_obj.root_node)
            inference_flops = estimate_hrt_node_flops(model_obj.root_node)
            training_flops = estimate_hrt_training_flops_from_data(
                model_obj.root_node,
                X_train_processed,
                getattr(model_obj, 'iter_counts_', None)
            )

        elif name == "HRT-Boost":
            n_params = 1 + sum(count_hrt_node_parameters(m.root_node) for m in model_obj.models_)
            inference_flops = sum(
                estimate_hrt_node_flops(m.root_node) + ENSEMBLE_ADD_FLOPS
                for m in model_obj.models_
            )
            model_train_flops = []
            for m in model_obj.models_:
                fit_flops = estimate_hrt_training_flops_from_data(
                    m.root_node,
                    X_train_processed,
                    getattr(m, 'iter_counts_', None)
                )
                pred_flops = estimate_hrt_node_flops(m.root_node)
                if not np.isnan(fit_flops):
                    if np.isnan(pred_flops):
                        pred_flops = 0.0
                    model_train_flops.append(fit_flops + n_train * (pred_flops + ENSEMBLE_ADD_FLOPS))
            training_flops = float(np.nansum(model_train_flops)) if model_train_flops else np.nan

        elif name == "CART":
            n_params = sklearn_tree_parameter_count(model_obj)
            inference_flops = sklearn_tree_inference_flops(model_obj)
            training_flops = sklearn_tree_training_flops(model_obj, n_feat)

        elif name == "RF":
            n_estimators = len(model_obj.estimators_)
            n_params = sum(sklearn_tree_parameter_count(t) for t in model_obj.estimators_)
            inference_flops = sum(sklearn_tree_inference_flops(t) for t in model_obj.estimators_)
            training_flops = sum(sklearn_tree_training_flops(t, n_feat) for t in model_obj.estimators_)
            training_flops += n_estimators * n_train * 1.0  # bootstrap/sample-index construction

        elif name == "AdaBoost":
            n_estimators = len(model_obj.estimators_)
            n_params = sum(sklearn_tree_parameter_count(t) for t in model_obj.estimators_)
            n_params += 2 * n_estimators  # estimator_weights_ + estimator_errors_
            inference_flops = sum(sklearn_tree_inference_flops(t) for t in model_obj.estimators_)
            inference_flops += n_estimators * (ENSEMBLE_ADD_FLOPS + np.log2(max(n_estimators, 2)))
            training_flops = sum(sklearn_tree_training_flops(t, n_feat) for t in model_obj.estimators_)
            training_flops += n_estimators * n_train * 8.0  # residual/error + sample-weight update

        elif name == "Scikit-GBM":
            trees = [t[0] for t in model_obj.estimators_]
            n_trees = len(trees)
            n_params = sum(sklearn_tree_parameter_count(t) for t in trees)
            n_params += 1  # init estimator/intercept
            inference_flops = sum(sklearn_tree_inference_flops(t) for t in trees)
            inference_flops += n_trees * ENSEMBLE_ADD_FLOPS + 1
            training_flops = sum(sklearn_tree_training_flops(t, n_feat) for t in trees)
            training_flops += n_trees * n_train * 8.0  # residual/negative-gradient/update

        elif name == "XGBoost":
            internal_counts, leaf_counts, avg_depths, max_depths = [], [], [], []
            for dump_text in model_obj.get_booster().get_dump():
                i_count, l_count, avg_d, max_d = xgboost_dump_counts_and_depth(dump_text)
                internal_counts.append(i_count)
                leaf_counts.append(l_count)
                avg_depths.append(avg_d)
                max_depths.append(max_d)
            n_trees = len(leaf_counts)

            n_params = int(
                XGBOOST_INTERNAL_PARAM_COUNT * np.nansum(internal_counts)
                + np.nansum(leaf_counts)
            )

            inference_flops = float(
                np.nansum(avg_depths) * XGBOOST_SPLIT_FLOPS
                + n_trees * XGBOOST_ENSEMBLE_ADD_FLOPS
            )

            training_flops = estimate_xgboost_training_flops(
                internal_counts=internal_counts,
                leaf_counts=leaf_counts,
                max_depths=max_depths,
                best_param_dict=best_param_dict,
                n_train=n_train,
                n_features=n_feat
            )

        elif name == "LightGBM":
            tree_info = model_obj.booster_.dump_model().get('tree_info', [])
            internal_counts, leaf_counts, avg_depths, max_depths = [], [], [], []
            for tree in tree_info:
                root = tree.get('tree_structure', {})
                i_count, l_count = lightgbm_tree_counts(root)
                depth_sum, leaf_count, max_depth = lightgbm_leaf_depth_stats(root)
                internal_counts.append(i_count)
                leaf_counts.append(l_count)
                avg_depths.append(depth_sum / max(leaf_count, 1))
                max_depths.append(max_depth)
            n_trees = len(tree_info)
            n_params = int(LIGHTGBM_INTERNAL_PARAM_COUNT * np.nansum(internal_counts) + np.nansum(leaf_counts))
            inference_flops = float(np.nansum(avg_depths) * LIGHTGBM_SPLIT_FLOPS + n_trees * ENSEMBLE_ADD_FLOPS)
            training_flops = estimate_lightgbm_training_flops(
                internal_counts=internal_counts,
                leaf_counts=leaf_counts,
                max_depths=max_depths,
                best_param_dict=best_param_dict,
                n_train=n_train,
                n_features=n_feat
            )

        elif name in {"TabM", "TabNet"}:
            n_params = count_torch_parameters_from_obj(model_obj)
            if np.isnan(n_params):
                n_params = _fallback_parameter_count_from_size(model_size_mb)
            inference_flops = estimate_neural_inference_flops(name, n_params, best_param_dict, n_train, n_feat)
            training_flops = estimate_neural_training_flops(name, inference_flops, best_param_dict, n_train, n_feat, n_params)

    except Exception:
        n_params, inference_flops, training_flops = np.nan, np.nan, np.nan

    return {
        "n_params": _as_finite_float(n_params),
        "model_size_mb": _as_finite_float(model_size_mb),
        "inference_flops": _as_finite_float(inference_flops),
        "training_flops": _as_finite_float(training_flops),
    }

# -------------------------------------------------------------------
# Benchmark execution
# -------------------------------------------------------------------

all_best_params_records = []
dataset_records_for_latex = []

with PdfPages(FINAL_PDF_NAME) as pdf:
    for config in DATASET_CONFIGS:
        print(f"\n>>>> Processing dataset: {config['name']} <<<<")

        try:
            sep = config.get("sep", None)

            df = pd.read_csv(resolve_data_path(config["train"]), header=None, sep=sep, engine='python')

            if df.shape[1] <= 1:
                df = pd.read_csv(resolve_data_path(config["train"]), header=None, sep=r'\s+', engine='python')

            df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')
            df = df.dropna()

            X_raw, y = df.iloc[:, :-1], df.iloc[:, -1].values.astype(float)
            n_f, n_s = X_raw.shape[1], len(y)

            if config["test"]:
                df_te = pd.read_csv(resolve_data_path(config["test"]), header=None, sep=sep, engine='python')

                if df_te.shape[1] <= 1:
                    df_te = pd.read_csv(resolve_data_path(config["test"]), header=None, sep=r'\s+', engine='python')

                df_te.iloc[:, -1] = pd.to_numeric(df_te.iloc[:, -1], errors='coerce')
                df_te = df_te.dropna()

                X_tr_raw = X_raw
                y_tr = y
                X_te_raw = df_te.iloc[:, :-1]
                y_te = df_te.iloc[:, -1].values.astype(float)
            else:
                X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
                    X_raw,
                    y,
                    test_size=0.5,
                    random_state=RANDOM_STATE
                )

        except Exception as e:
            print(f"Loading failed; skipping {config['name']}: {e}")
            continue

        pre = ColumnTransformer(
            [
                (
                    'num',
                    StandardScaler(),
                    [i for i in range(X_tr_raw.shape[1]) if i not in config["cat_cols"]]
                ),
                (
                    'cat',
                    make_one_hot_encoder(),
                    config["cat_cols"]
                )
            ],
            sparse_threshold=0.0
        )

        models_info = {
            "HRT": {
                "estimator": HRTRegressor(),
                "params": {
                    "max_depth": [2, 4, 6],
                    "ridge_alpha": [0.01, 0.1, 1.0],
                    "min_points": [5, 10]
                }
            },

            "HRT-Boost": {
                "estimator": HRTBoostingRegressor(),
                "params": {
                    "n_estimators": [50, 150],
                    "learning_rate": [0.01, 0.1, 1],
                    "max_depth": [2, 3, 4],
                    "ridge_alpha": [0, 0.1, 1.0]
                }
            },

            "CART": {
                "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
                "params": {
                    "max_depth": [5, 10, 20, None],
                    "min_samples_leaf": [1, 5, 10],
                    "criterion": ["squared_error", "friedman_mse"]
                }
            },

            "RF": {
                "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
                "params": {
                    "n_estimators": [50, 150],
                    "max_depth": [3, 5, 7],
                    "max_features": ["sqrt", "log2", None],
                    "min_samples_leaf": [1, 4]
                }
            },

            "AdaBoost": {
                "estimator": AdaBoostRegressor(random_state=RANDOM_STATE),
                "params": {
                    "n_estimators": [50, 150],
                    "learning_rate": [0.1, 0.5, 1.0],
                    "loss": ["linear", "square"]
                }
            },

            "Scikit-GBM": {
                "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
                "params": {
                    "n_estimators": [50, 150],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [2, 4],
                    "subsample": [0.8, 1.0]
                }
            },
        }

        if XGBRegressor:
            models_info["XGBoost"] = {
                "estimator": XGBRegressor(**xgb_runtime_params()),
                "params": {
                    "n_estimators": [50, 150],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 6],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "max_bin": [XGBOOST_DEFAULT_MAX_BIN]
                }
            }

        if LGBMRegressor:
            models_info["LightGBM"] = {
                "estimator": LGBMRegressor(
                    random_state=RANDOM_STATE,
                    verbosity=-1,
                    n_jobs=1,
                    **({"device_type": "gpu"} if LIGHTGBM_GPU_AVAILABLE else {})
                ),
                "params": {
                    "n_estimators": [50, 150],
                    "learning_rate": [0.01, 0.1],
                    "num_leaves": [31, 63],
                    "reg_alpha": [0.0, 0.1]
                }
            }

        if TabMRaw:
            models_info["TabM"] = {
                "estimator": TabMSklearnRegressor(
                    random_state=RANDOM_STATE,
                    verbose=0,
                    device_name=("cuda" if TORCH_CUDA_AVAILABLE else "cpu")
                ),
                "params": {
                    "arch_type": ["tabm"],
                    "k": [8, 16],
                    "n_blocks": [2],
                    "d_block": [128],
                    "dropout": [0.0, 0.1],
                    "learning_rate": [0.001, 0.002],
                    "weight_decay": [3e-4],
                    "max_epochs": [50],
                    "batch_size": [256]
                }
            }
        else:
            if should_run_method("TabM"):
                print("Notice: tabm was not detected; skipping the TabM baseline.")

        if TabNetRegressorRaw:
            models_info["TabNet"] = {
                "estimator": TabNetSklearnRegressor(
                    random_state=RANDOM_STATE,
                    verbose=0,
                    device_name=("cuda" if TORCH_CUDA_AVAILABLE else "cpu")
                ),
                "params": {
                    "n_d": [8, 16],
                    "n_a": [8, 16],
                    "n_steps": [3, 4],
                    "learning_rate": [0.01, 0.02],
                    "max_epochs": [50],
                    "patience": [10]
                }
            }
        else:
            if should_run_method("TabNet"):
                print("Notice: pytorch-tabnet was not detected; skipping the TabNet baseline.")

        unavailable_selected_methods = [
            name for name in methods_to_show
            if name not in models_info
        ]
        if unavailable_selected_methods:
            print(
                "Notice: the following selected methods will not run in this round because dependencies are missing or the methods were skipped dynamically: "
                + ", ".join(unavailable_selected_methods)
            )

        models_info = {
            name: models_info[name]
            for name in methods_to_show
            if name in models_info
        }

        if not models_info:
            raise RuntimeError(
                "The current RUN_METHODS/RUN_BASELINES selection contains no runnable models. "
                "Install the required dependencies or adjust RUN_METHODS."
            )

        cache_dir = os.path.join(PIPELINE_CACHE_DIR, config['name'])
        os.makedirs(cache_dir, exist_ok=True)
        pipeline_memory = Memory(location=cache_dir, verbose=0)

        best_params = {}

        try:
            for name, info in models_info.items():
                gpu_or_deep_models = ["TabM", "TabNet"]
                if XGBOOST_GPU_AVAILABLE:
                    gpu_or_deep_models.append("XGBoost")
                if LIGHTGBM_GPU_AVAILABLE:
                    gpu_or_deep_models.append("LightGBM")
                grid_n_jobs = 1 if name in gpu_or_deep_models else GRID_N_JOBS

                gs = GridSearchCV(
                    make_pipeline(pre, info["estimator"], memory=pipeline_memory),
                    {
                        f'reg__{k}': v
                        for k, v in info["params"].items()
                    },
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=grid_n_jobs,
                    pre_dispatch=GRID_PRE_DISPATCH,
                    return_train_score=False
                )

                gs.fit(X_tr_raw, y_tr)

                best_params[name] = {
                    k.replace('reg__', ''): v
                    for k, v in gs.best_params_.items()
                }
        finally:
            gc.collect()

        results = {
            name: {
                'RMSE': [],
                'Inference_FLOPs': [],
                'Training_FLOPs': []
            }
            for name in models_info
        }

        for i in range(N_REPS):
            for name in models_info:
                reg = clone(models_info[name]["estimator"])
                reg.set_params(**best_params[name])
                reg = apply_final_runtime_params(reg)

                if hasattr(reg, 'random_state'):
                    reg.random_state = RANDOM_STATE + i

                pipe = make_pipeline(pre, reg, memory=pipeline_memory)

                pipe.fit(X_tr_raw, y_tr)
                preds = pipe.predict(X_te_raw)

                m_obj = pipe.named_steps['reg']

                try:
                    X_train_processed_for_complexity = to_dense_float32(
                        pipe.named_steps['pre'].transform(X_tr_raw)
                    )
                except Exception:
                    X_train_processed_for_complexity = None

                complexity = estimate_model_complexity(
                    name,
                    m_obj,
                    best_params.get(name, {}),
                    X_train_processed=X_train_processed_for_complexity,
                    n_train_samples=len(y_tr),
                    n_features=n_f
                )

                results[name]['RMSE'].append(
                    np.sqrt(mean_squared_error(y_te, preds))
                )
                results[name]['Inference_FLOPs'].append(complexity['inference_flops'])
                results[name]['Training_FLOPs'].append(complexity['training_flops'])

        summary_df = pd.DataFrame(
            [
                {
                    "Model": n,
                    "RMSE": np.mean(results[n]['RMSE']),
                    "Inference FLOPs/sample": mean_std_ignore_nan(results[n]['Inference_FLOPs'])[0],
                    "Training FLOPs": mean_std_ignore_nan(results[n]['Training_FLOPs'])[0]
                }
                for n in models_info
            ]
        )

        print(f"\nDataset {config['name']} summary:")
        print(summary_df.to_string(index=False))

        dataset_records_for_latex.append(
            {
                "name": config['name'],
                "n_f": n_f,
                "n_s": n_s,
                "rmse_stats": {
                    n: (
                        np.mean(results[n]['RMSE']),
                        np.std(results[n]['RMSE'])
                    )
                    for n in models_info
                },
                "flop_stats": {
                    n: mean_std_ignore_nan(results[n]['Inference_FLOPs'])
                    for n in models_info
                },
                "training_flop_stats": {
                    n: mean_std_ignore_nan(results[n]['Training_FLOPs'])
                    for n in models_info
                }
            }
        )

        all_best_params_records.append(
            {
                "name": config['name'],
                "best_params": best_params
            }
        )

        try:
            pipeline_memory.clear(warn=False)
            shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception:
            pass
        gc.collect()


# -------------------------------------------------------------------
# -------------------------------------------------------------------



def latex_escape(text):
    """Escape a value for use in LaTeX tables."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def format_compact_number(value, decimals=2):
    """Format large numerical values using compact K/M/B suffixes."""
    if value is None or np.isnan(value):
        return "N/A"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"{value / 1e9:.{decimals}f}B"
    if abs_value >= 1e6:
        return f"{value / 1e6:.{decimals}f}M"
    if abs_value >= 1e3:
        return f"{value / 1e3:.{decimals}f}K"
    if abs_value >= 10:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}"


def format_stat_compact(mean_val, std_val, decimals=2):
    """Format mean and standard deviation for compact LaTeX tables."""
    if mean_val is None or np.isnan(mean_val):
        return "N/A"
    if std_val is None or np.isnan(std_val) or abs(float(std_val)) < 1e-12:
        return format_compact_number(mean_val, decimals=decimals)
    return f"{format_compact_number(mean_val, decimals=decimals)} \\(\\pm\\) {format_compact_number(std_val, decimals=decimals)}"


def generate_benchmark_latex_table(records):
    """Generate the official benchmark table with RMSE and FLOPs only."""
    latex_str = r"""
\begin{table*}[htbp]
\tiny
    \centering
    \caption{Official HRT-Boost benchmark summary. RMSE is reported as mean $\pm$ standard deviation over repeated runs. Inference and training FLOPs are analytical estimates.}
    \label{tab:hrt_boost_benchmark_summary}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{l l c c c}
        \toprule
        \textbf{Dataset} & \textbf{Model} & \textbf{RMSE} & \textbf{Inference FLOPs / sample} & \textbf{Training FLOPs} \\
        \midrule """
    for i, rec in enumerate(records):
        dataset_name = latex_escape(rec["name"].replace('_', ' ').capitalize())
        n_s_str = f"{rec['n_s'] // 1000}k" if rec['n_s'] >= 1000 else str(rec['n_s'])
        dataset_cell = f"{dataset_name} ({rec['n_f']}, {n_s_str})"
        available_models = [m for m in methods_to_show if m in rec.get("rmse_stats", {})]
        if not available_models:
            continue
        for j, model_name in enumerate(available_models):
            prefix = f"\n        \\multirow{{{len(available_models)}}}{{*}}{{ {dataset_cell} }}" if j == 0 else "\n        "
            r_mean, r_std = rec["rmse_stats"].get(model_name, (np.nan, np.nan))
            f_mean, f_std = rec["flop_stats"].get(model_name, (np.nan, np.nan))
            tf_mean, tf_std = rec.get("training_flop_stats", {}).get(model_name, (np.nan, np.nan))
            latex_str += (
                f"{prefix} & {latex_escape(model_name)} "
                f"& {format_stat_compact(r_mean, r_std, decimals=4)} "
                f"& {format_stat_compact(f_mean, f_std, decimals=2)} "
                f"& {format_stat_compact(tf_mean, tf_std, decimals=2)} \\\\"
            )
        if i < len(records) - 1:
            latex_str += "\n        \\midrule "
    latex_str += r"""
        \bottomrule
    \end{tabular}}
\end{table*}"""
    return latex_str


table_benchmark = generate_benchmark_latex_table(dataset_records_for_latex)

full_latex = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage[landscape, margin=1in]{geometry}

\begin{document}

\section{Benchmark Results}
""" + table_benchmark + r"""

\end{document}
"""

with open(FINAL_TEX_NAME, "w", encoding="utf-8") as f:
    f.write(full_latex)

print(f"\nOK Benchmark LaTeX source has been saved to: {FINAL_TEX_NAME}")
