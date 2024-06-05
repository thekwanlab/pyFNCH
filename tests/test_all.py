import pytest
from pyFNCH.FNCHyper import FNCHyper, FastEstimation, ExactEstimation
import numpy as np

test_fnch = FNCHyper(4,2,3,weight=2)

def test_mean():
    actual = 2.27272
    observed = test_fnch.mean()
    assert abs(actual - observed) < 1e-3

def test_variance():
    actual = 0.380165
    observed = test_fnch.variance()
    assert abs(actual - observed) < 1e-3

def test_mode():
    actual = 2
    observed = test_fnch.mode()
    assert actual == observed

N_WHITE_TEST = 500
N_BLACK_TEST = 500
M_TEST = 500

X = 400
ws = np.arange(1, 50, 0.01)

exact_estimation = ExactEstimation(N_WHITE_TEST, N_BLACK_TEST, M_TEST)
fast_estimation = FastEstimation(N_WHITE_TEST, N_BLACK_TEST, M_TEST)

scores_exact = np.vectorize(lambda w: exact_estimation.score(w, X))(ws)
scores_fast = np.vectorize(lambda w: fast_estimation.score(w, X))(ws)

informations_exact = np.vectorize(lambda w: exact_estimation.information(w, X))(ws)
informations_fast = np.vectorize(lambda w: fast_estimation.information(w, X))(ws)

means_exact = np.vectorize(lambda w: exact_estimation.factory(w).mean())(ws)
means_fast = np.vectorize(lambda w: fast_estimation.factory(w).mean())(ws)

variances_exact = np.vectorize(lambda w: exact_estimation.factory(w).variance())(ws)
variances_fast = np.vectorize(lambda w: fast_estimation.factory(w).variance())(ws)

def is_all_within_threshold(a1, a2, threshold=0.05):
    return np.all(np.abs(a1-a2) < threshold)

def test_score_agreement():
    fast = np.vectorize(lambda w: fast_estimation.score(w, X))(ws)
    exact = np.vectorize(lambda w: exact_estimation.score(w, X))(ws)
    assert is_all_within_threshold(fast, exact), "Fast Score and Exact Score do not agree"

def test_information_agreement():
    fast = np.vectorize(lambda w: fast_estimation.information(w, X))(ws)
    exact = np.vectorize(lambda w: exact_estimation.information(w, X))(ws)
    assert is_all_within_threshold(fast, exact), "Fast Information and Exact Information do not agree"

def test_mean_agreement():
    exact = np.vectorize(lambda w: exact_estimation.factory(w).mean())(ws)
    fast = np.vectorize(lambda w: fast_estimation.factory(w).mean())(ws)
    assert is_all_within_threshold(fast, exact, threshold=0.05), "Fast Mean and Exact Mean do not agree"

def test_variance_agreement():
    exact = np.vectorize(lambda w: exact_estimation.factory(w).mean())(ws)
    fast = np.vectorize(lambda w: fast_estimation.factory(w).mean())(ws)
    assert is_all_within_threshold(fast, exact, threshold=0.05), "Fast Variance and Exact Variance do not agree"

@pytest.mark.parametrize('n_success, n_fail, m_draws, x, expected_mle', [
    (500, 500, 500, 400, 16.0),
    (500, 500, 500, 450, 81.0),
    (186, 51912-186, 71, 37, 377.537308),
    (2, 2, 2, 1, 1.0),
    (2, 2, 2, 2, np.inf),
    (2, 2, 2, 0, 0.0),
])
def test_mle(n_success, n_fail, m_draws, x, expected_mle):
    observed_mle = FastEstimation(n_success=n_success, n_failure=n_fail, m_draws=m_draws).fit(x)._mle
    assert expected_mle == observed_mle or abs(expected_mle - observed_mle) < 1e-3

