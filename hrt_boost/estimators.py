"""Core scikit-learn estimators for HRT-Boost.

HRT-Boost is built from Hinge Regression Tree (HRT) base learners. The HRT
split optimizer follows the public HRT reference implementation at
https://github.com/Hongyi-Li-sz/Hinge-Regression-Tree and is packaged here as
a lightweight NumPy/scikit-learn implementation for regression experiments.

The main public estimator is :class:`HRTBoostingRegressor`. The single-tree
:class:`HRTRegressor` is exported for ablation studies and baseline comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


@dataclass
class Node:
    """A node in a hinge regression tree."""

    is_leaf: bool
    region: list
    params: Optional[np.ndarray] = None
    split_coeffs: Optional[np.ndarray] = None
    children: Optional[list["Node"]] = None
    stop_reason: Optional[str] = None

    def count_leaves(self) -> int:
        """Return the number of leaf nodes below this node."""
        if self.is_leaf:
            return 1
        return sum(child.count_leaves() for child in (self.children or []))


def solve_ols_nd(X_des: np.ndarray, z: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """Solve a ridge-regularized least-squares problem on a design matrix.

    Parameters
    ----------
    X_des:
        Dense design matrix with a final intercept column.
    z:
        Target vector.
    alpha:
        Ridge penalty applied to non-intercept coefficients.
    """
    n_points, n_coeffs = X_des.shape
    if n_points < n_coeffs:
        intercept = float(np.mean(z)) if n_points > 0 else 0.0
        return np.append(np.zeros(n_coeffs - 1), intercept)

    xtx = X_des.T @ X_des
    xtz = X_des.T @ z
    if alpha > 0:
        penalty = np.eye(n_coeffs)
        penalty[-1, -1] = 0.0
        xtx += alpha * penalty

    try:
        return np.linalg.solve(xtx, xtz)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(xtx, xtz, rcond=None)[0]


def initialize_thetas_nd(
    X_des: np.ndarray,
    z: np.ndarray,
    seed: Optional[int] = None,
    ridge_alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize the two linear predictors used by a split search."""
    n_points, n_dim = X_des.shape
    n_features = n_dim - 1
    mean_z = float(np.mean(z)) if n_points > 0 else 0.0

    if n_points < n_dim:
        theta = np.append(np.zeros(n_features), mean_z)
        return theta, theta.copy()

    rng = np.random.RandomState(seed) if seed is not None else np.random
    chosen_dim = 0
    for dim in range(n_features):
        if np.max(X_des[:, dim]) > np.min(X_des[:, dim]):
            chosen_dim = dim
            break

    median_value = np.median(X_des[:, chosen_dim])
    left_mask = X_des[:, chosen_dim] < median_value

    if np.sum(left_mask) < n_dim or np.sum(~left_mask) < n_dim:
        global_theta = solve_ols_nd(X_des, z, alpha=ridge_alpha)
        jitter = rng.rand(n_dim) * 0.005
        return global_theta + jitter, global_theta - jitter

    return (
        solve_ols_nd(X_des[left_mask], z[left_mask], alpha=ridge_alpha),
        solve_ols_nd(X_des[~left_mask], z[~left_mask], alpha=ridge_alpha),
    )


def _rmse_eval(
    X_des: np.ndarray,
    z: np.ndarray,
    theta_1: np.ndarray,
    theta_2: np.ndarray,
    flip: bool,
    min_segment: int,
    min_fit: int,
) -> tuple[bool, float, np.ndarray]:
    split_vector = (theta_2 - theta_1) if flip else (theta_1 - theta_2)
    mask = (X_des @ split_vector) < 0
    n_left = int(np.sum(mask))

    if (
        n_left < min_segment
        or (len(z) - n_left) < min_segment
        or n_left < min_fit
        or (len(z) - n_left) < min_fit
    ):
        return False, np.inf, mask

    pred = np.empty_like(z, dtype=float)
    pred[mask] = X_des[mask] @ theta_1
    pred[~mask] = X_des[~mask] @ theta_2
    return True, float(np.sqrt(np.mean((z - pred) ** 2))), mask


def _run_split_dir(
    X_des: np.ndarray,
    z: np.ndarray,
    theta_1_init: np.ndarray,
    theta_2_init: np.ndarray,
    max_iter: int,
    tol: float,
    min_segment: int,
    min_fit: int,
    step_size,
    ridge: float,
    flip: bool,
):
    theta_1, theta_2 = theta_1_init.copy(), theta_2_init.copy()
    previous_mask, current_rmse = None, np.inf

    valid, rmse, _ = _rmse_eval(X_des, z, theta_1, theta_2, flip, min_segment, min_fit)
    if valid:
        current_rmse = rmse
    else:
        return None, 0

    iteration = 0
    for iteration in range(max_iter):
        split_vector = (theta_2 - theta_1) if flip else (theta_1 - theta_2)
        mask = (X_des @ split_vector) < 0
        if previous_mask is not None and np.array_equal(mask, previous_mask):
            break
        previous_mask = mask

        n_left = int(np.sum(mask))
        if n_left < min_fit or (len(z) - n_left) < min_fit:
            break

        try:
            theta_1_target = solve_ols_nd(X_des[mask], z[mask], alpha=ridge)
            theta_2_target = solve_ols_nd(X_des[~mask], z[~mask], alpha=ridge)

            if isinstance(step_size, (int, float)):
                theta_1 = (1 - step_size) * theta_1 + step_size * theta_1_target
                theta_2 = (1 - step_size) * theta_2 + step_size * theta_2_target
            else:
                step, accepted = 1.0, False
                for _ in range(5):
                    theta_1_try = (1 - step) * theta_1 + step * theta_1_target
                    theta_2_try = (1 - step) * theta_2 + step * theta_2_target
                    valid_try, rmse_try, _ = _rmse_eval(
                        X_des, z, theta_1_try, theta_2_try, flip, min_segment, min_fit
                    )
                    if valid_try and rmse_try < current_rmse - 1e-8:
                        theta_1, theta_2 = theta_1_try, theta_2_try
                        current_rmse, accepted = rmse_try, True
                        break
                    step *= 0.5
                if not accepted:
                    break

            if np.linalg.norm(theta_1 - theta_1_target) < tol:
                break
        except Exception:
            break

    valid, rmse, mask = _rmse_eval(X_des, z, theta_1, theta_2, flip, min_segment, min_fit)
    if valid:
        split_coeffs = (theta_2 - theta_1) if flip else (theta_1 - theta_2)
        return (split_coeffs, theta_1, theta_2, mask, ~mask, rmse), iteration + 1
    return None, iteration + 1


def optimize_split_aligned(
    X_des: np.ndarray,
    z: np.ndarray,
    min_segment: int,
    step,
    seed: Optional[int],
    ridge: float,
):
    """Optimize an HRT split by alternating between routing and least squares."""
    n_points, n_dim = X_des.shape
    min_fit = n_dim
    if n_points < 2 * max(min_fit, min_segment):
        return None, None, None, None, None, 0

    theta_1_init, theta_2_init = initialize_thetas_nd(X_des, z, seed=seed, ridge_alpha=ridge)
    best_result, best_rmse, best_iterations = None, float("inf"), 0

    for flip in (False, True):
        result, iterations = _run_split_dir(
            X_des,
            z,
            theta_1_init,
            theta_2_init,
            max_iter=50,
            tol=1e-5,
            min_segment=min_segment,
            min_fit=min_fit,
            step_size=step,
            ridge=ridge,
            flip=flip,
        )
        if result and result[-1] < best_rmse:
            best_rmse, best_result, best_iterations = result[-1], result, iterations

    return (*best_result[:-1], best_iterations) if best_result else (None, None, None, None, None, best_iterations)


def recursive_fit_aligned(
    X_des_full: np.ndarray,
    z_full: np.ndarray,
    indices: np.ndarray,
    threshold: float,
    min_segment: int,
    depth: int,
    max_depth: int,
    seed: Optional[int],
    step,
    ridge: float,
    iteration_counts: list[int],
) -> Node:
    """Recursively fit an HRT tree."""
    X_slice = X_des_full[indices]
    z_slice = z_full[indices]
    n_points, n_dim = X_slice.shape
    region = []

    if len(indices) < n_dim + 1:
        params = solve_ols_nd(X_slice, z_slice, alpha=ridge)
        return Node(True, region, params=params, stop_reason="small")

    params = solve_ols_nd(X_slice, z_slice, alpha=ridge)
    if depth >= max_depth:
        return Node(True, region, params=params, stop_reason="depth")

    split = None
    for attempt in range(2):
        result = optimize_split_aligned(
            X_slice,
            z_slice,
            min_segment,
            step,
            (seed or 0) + depth * 100 + attempt,
            ridge,
        )
        if result[0] is not None:
            split = result
            break

    if split:
        split_coeffs, _theta_1, _theta_2, left_mask, right_mask, iterations = split
        iteration_counts.append(iterations)
        return Node(
            False,
            region,
            split_coeffs=split_coeffs,
            children=[
                recursive_fit_aligned(
                    X_des_full,
                    z_full,
                    indices[left_mask],
                    threshold,
                    min_segment,
                    depth + 1,
                    max_depth,
                    seed,
                    step,
                    ridge,
                    iteration_counts,
                ),
                recursive_fit_aligned(
                    X_des_full,
                    z_full,
                    indices[right_mask],
                    threshold,
                    min_segment,
                    depth + 1,
                    max_depth,
                    seed,
                    step,
                    ridge,
                    iteration_counts,
                ),
            ],
        )

    return Node(True, region, params=params, stop_reason="failed_split")


class HRTRegressor(BaseEstimator, RegressorMixin):
    """Hinge Regression Tree (HRT) regressor.

    Parameters
    ----------
    threshold:
        Reserved for compatibility with earlier experimental scripts.
    min_points:
        Minimum number of samples allowed on each side of a split.
    max_depth:
        Maximum tree depth.
    step_size:
        Numeric damping value or ``"auto"`` for backtracking.
    ridge_alpha:
        Ridge penalty used in local least-squares fits.
    random_state:
        Random seed used for split initialization.
    """

    def __init__(
        self,
        threshold: float = 0.0,
        min_points: int = 5,
        max_depth: int = 5,
        step_size="auto",
        ridge_alpha: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.threshold = threshold
        self.min_points = min_points
        self.max_depth = max_depth
        self.step_size = step_size
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the regression tree."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.iter_counts_ = []
        X_des = np.column_stack([X, np.ones(X.shape[0])])
        self.root_node = recursive_fit_aligned(
            X_des,
            y,
            np.arange(len(y)),
            self.threshold,
            self.min_points,
            0,
            self.max_depth,
            self.random_state,
            self.step_size,
            self.ridge_alpha,
            self.iter_counts_,
        )
        return self

    def predict(self, X):
        """Predict target values for samples in ``X``."""
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        X_des = np.column_stack([X, np.ones(n_samples)])
        y_pred = np.zeros(n_samples, dtype=float)
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

    def count_leaves(self) -> int:
        """Return the number of fitted leaves."""
        return self.root_node.count_leaves()


# Backward-compatible descriptive alias. HRTRegressor is the preferred public name.
HingeRegressionTreeRegressor = HRTRegressor


class HRTBoostingRegressor(BaseEstimator, RegressorMixin):
    """Residual boosting with HRT trees as weak learners."""

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 2,
        step_size="auto",
        ridge_alpha: float = 1.0,
        random_state: Optional[int] = None,
        min_points: int = 5,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.step_size = step_size
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self.min_points = min_points

    def fit(self, X, y):
        """Fit the boosted ensemble."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.models_ = []
        self.initial_pred_ = float(np.mean(y))
        current_pred = np.full(X.shape[0], self.initial_pred_, dtype=float)

        for i in range(int(self.n_estimators)):
            residual = y - current_pred
            seed = self.random_state + i if self.random_state is not None else None
            model = HRTRegressor(
                max_depth=self.max_depth,
                step_size=self.step_size,
                ridge_alpha=self.ridge_alpha,
                random_state=seed,
                min_points=self.min_points,
            )
            model.fit(X, residual)
            current_pred += self.learning_rate * model.predict(X)
            self.models_.append(model)
        return self

    def predict(self, X):
        """Predict target values for samples in ``X``."""
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.initial_pred_, dtype=float)
        for model in self.models_:
            pred += self.learning_rate * model.predict(X)
        return pred


# Backward-compatible aliases used by the original benchmark script.
HRTRegressor = HingeRegressionTreeRegressor
