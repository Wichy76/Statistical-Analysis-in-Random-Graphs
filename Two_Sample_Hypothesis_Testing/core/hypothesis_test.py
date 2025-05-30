import numpy as np
from sklearn.covariance import EllipticEnvelope
from scipy.stats import chi2


def fit_elliptic_envelope(S, contamination=0.05):
    ee = EllipticEnvelope(contamination=contamination).fit(S)
    return ee


def test_statistic(ee, x):
    m2 = ee.mahalanobis(x.reshape(1, -1))[0]
    thr = chi2.ppf(0.95, df=x.shape[0])
    return m2, thr, m2 <= thr


def run_test(G1_stats, G2_stat, contamination=0.05):
    ee = fit_elliptic_envelope(G1_stats, contamination)
    return test_statistic(ee, G2_stat)


def run_bivariate_test(G1_stats, G2_stat, contamination=0.05):
    ee = EllipticEnvelope(contamination=contamination).fit(G1_stats[:, :2])
    return test_statistic(ee, G2_stat[:2])
