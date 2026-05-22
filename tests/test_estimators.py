"""Basic estimator smoke tests."""

import numpy as np

from hrt_boost import HRTBoostingRegressor, HRTRegressor


def test_hrt_tree_predicts_shape():
    rng = np.random.RandomState(0)
    x = rng.randn(80, 3)
    y = x[:, 0] - 2 * x[:, 1] + rng.normal(scale=0.01, size=80)
    model = HRTRegressor(max_depth=2, min_points=5, random_state=0)
    model.fit(x, y)
    pred = model.predict(x[:7])
    assert pred.shape == (7,)
    assert np.isfinite(pred).all()


def test_hrt_boost_predicts_shape():
    rng = np.random.RandomState(1)
    x = rng.randn(60, 2)
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1]
    model = HRTBoostingRegressor(n_estimators=3, learning_rate=0.1, max_depth=1, random_state=1)
    model.fit(x, y)
    pred = model.predict(x[:5])
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
