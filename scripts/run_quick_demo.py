#!/usr/bin/env python3
"""Run a small synthetic-data smoke test for the official HRT-Boost estimator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from hrt_boost import HRTBoostingRegressor, HRTRegressor


def make_data(seed: int = 42, n_samples: int = 3000):
    """Create a small piecewise-linear regression problem."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-2.0, 2.0, size=(n_samples, 2))
    y = np.where(
        x[:, 0] + 0.5 * x[:, 1] < 0,
        2.0 * x[:, 0] - x[:, 1],
        -x[:, 0] + 1.5 * x[:, 1],
    )
    y += rng.normal(scale=0.1, size=n_samples)
    return x, y


def evaluate(name, model, x_train, x_test, y_train, y_test):
    """Fit one estimator and print standard regression metrics."""
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"{name}: RMSE={rmse:.4f}, R2={r2:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run the HRT-Boost synthetic-data demo.")
    parser.add_argument(
        "--include-hrt",
        action="store_true",
        help="Also run the single HRT tree ablation.",
    )
    args = parser.parse_args()

    x, y = make_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=42
    )

    evaluate(
        "HRT-Boost",
        HRTBoostingRegressor(n_estimators=25, learning_rate=0.1, max_depth=2, random_state=42),
        x_train,
        x_test,
        y_train,
        y_test,
    )

    if args.include_hrt:
        evaluate(
            "HRT",
            HRTRegressor(max_depth=3, ridge_alpha=0.1, random_state=42),
            x_train,
            x_test,
            y_train,
            y_test,
        )


if __name__ == "__main__":
    main()
