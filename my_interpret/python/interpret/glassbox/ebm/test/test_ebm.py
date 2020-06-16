# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ....test.utils import (
    synthetic_multiclass,
    synthetic_classification,
    adult_classification,
    iris_classification,
)
from ....test.utils import synthetic_regression
from ..ebm import ExplainableBoostingRegressor, ExplainableBoostingClassifier

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_validate,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.metrics import accuracy_score
import pytest

import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn


@pytest.mark.slow
def test_ebm_synthetic_multiclass():
    data = synthetic_multiclass()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=0, n_estimators=2)
    clf.fit(X, y)

    prob_scores = clf.predict_proba(X)

    within_bounds = (prob_scores >= 0.0).all() and (prob_scores <= 1.0).all()
    assert within_bounds

    valid_ebm(clf)


@pytest.mark.slow
def test_ebm_synthetic_multiclass_pairwise():
    data = synthetic_multiclass()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=1, n_estimators=2)
    with pytest.raises(RuntimeError):
        clf.fit(X, y)


@pytest.mark.slow
def test_ebm_multiclass():
    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    clf = ExplainableBoostingClassifier()
    clf.fit(X_train, y_train)

    assert accuracy_score(y_test, clf.predict(X_test)) > 0.9


def test_ebm_synthetic_pairwise():
    a = np.random.randint(low=0, high=50, size=10000)
    b = np.random.randint(low=0, high=20, size=10000)

    df = pd.DataFrame(np.c_[a, b], columns=["a", "b"])
    df["y"] = [
        1 if (x > 35 and y > 15) or (x < 15 and y < 5) else 0
        for x, y in zip(df["a"], df["b"])
    ]

    X = df[["a", "b"]]
    y = df["y"]

    seed = 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    clf = ExplainableBoostingClassifier(interactions=1)
    clf.fit(X_train, y_train)

    clf_global = clf.explain_global()

    # Low/Low and High/High should learn high scores
    assert clf_global.data(2)["scores"][-1][-1] > 5
    assert clf_global.data(2)["scores"][0][0] > 5


def test_prefit_ebm():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=1, interactions=0, data_n_episodes=0)
    clf.fit(X, y)

    for _, attrib_set_model in enumerate(clf.attribute_set_models_):
        has_non_zero = np.any(attrib_set_model)
        assert not has_non_zero


@pytest.mark.slow
def test_ebm_synthetic_regression():
    data = synthetic_regression()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingRegressor(n_jobs=-2, interactions=0)
    clf.fit(X, y)
    clf.predict(X)

    valid_ebm(clf)


def valid_ebm(ebm):
    assert ebm.attribute_sets_[0]["n_attributes"] == 1
    assert ebm.attribute_sets_[0]["attributes"] == [0]

    for _, attrib_set_model in enumerate(ebm.attribute_set_models_):
        all_finite = np.isfinite(attrib_set_model).all()
        assert all_finite


@pytest.mark.slow
def test_ebm_synthetic_classfication():
    data = synthetic_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=0)
    clf.fit(X, y)
    prob_scores = clf.predict_proba(X)

    within_bounds = (prob_scores >= 0.0).all() and (prob_scores <= 1.0).all()
    assert within_bounds

    valid_ebm(clf)


@pytest.mark.visual
@pytest.mark.slow
def test_ebm_adult():
    from .... import preserve, show, shutdown_show_server, set_show_addr

    data = adult_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=3)
    n_splits = 3
    ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=1337)
    cross_validate(
        clf, X, y, scoring="roc_auc", cv=ss, n_jobs=None, return_estimator=True
    )
    clf.fit(X, y)
    prob_scores = clf.predict_proba(X)

    within_bounds = (prob_scores >= 0.0).all() and (prob_scores <= 1.0).all()
    assert within_bounds

    valid_ebm(clf)

    set_show_addr(("127.0.0.1", 6000))
    global_exp = clf.explain_global()
    local_exp = clf.explain_local(X[:5, :], y[:5])

    # Smoke test: should run without crashing.
    preserve(global_exp)
    preserve(local_exp)
    show(global_exp)
    show(local_exp)

    # Check all features for global (including interactions).
    for selector_key in global_exp.selector[global_exp.selector.columns[0]]:
        preserve(global_exp, selector_key)

    shutdown_show_server()
