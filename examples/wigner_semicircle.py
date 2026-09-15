"""Linear algebra: GOE eigenvalues against the analytic semicircle law.

Problem
    Eigenvalues of a large symmetric Gaussian random matrix, rescaled by
    sqrt(N), should follow the Wigner semicircle density on [-2, 2].

Verification (independent_method)
    A one-sample Kolmogorov-Smirnov test compares the empirical eigenvalue
    distribution from a dense symmetric eigensolver against the closed-form
    semicircle CDF. The histogram and the CDF come from entirely different
    places: LAPACK, and calculus.
"""

import numpy as np
from scipy.stats import kstest

N = 600
rng = np.random.default_rng(20260916)

raw = rng.standard_normal((N, N))
matrix = (raw + raw.T) / np.sqrt(2.0)  # off-diagonal variance 1
eigenvalues = np.linalg.eigvalsh(matrix) / np.sqrt(N)


def semicircle_cdf(x):
    """Integral of sqrt(4 - x**2) / (2*pi) from -2 to x."""
    x = np.clip(np.asarray(x, dtype=float), -2.0, 2.0)
    return 0.5 + x * np.sqrt(4.0 - x**2) / (4.0 * np.pi) + np.arcsin(x / 2.0) / np.pi


result = kstest(eigenvalues, semicircle_cdf)
print("N =", N)
print("eigenvalue range =", repr(float(eigenvalues.min())), repr(float(eigenvalues.max())))
print("KS statistic =", repr(float(result.statistic)))
print("KS p-value =", repr(float(result.pvalue)))

# the CDF is a probability distribution, not just a curve that fits
edges = semicircle_cdf(np.array([-2.0, 0.0, 2.0]))
print("CDF at -2, 0, 2 =", repr(edges.tolist()))

assert abs(edges[0]) < 1e-12 and abs(edges[2] - 1.0) < 1e-12, "semicircle CDF is not normalised"
assert abs(edges[1] - 0.5) < 1e-12, "semicircle CDF is not symmetric"
assert result.statistic < 0.05, f"eigenvalues depart from the semicircle law: D={result.statistic}"
assert eigenvalues.max() < 2.3, "spectrum spills well outside the semicircle support"
print("VERIFIED: empirical spectrum matches the analytic semicircle law")
