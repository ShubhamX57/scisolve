"""Quadrature: a singular integral with a closed form, at three tolerances.

Problem
    integral of ln(sin x) from 0 to pi/2, which is -(pi/2) * ln 2. The
    integrand diverges at the left endpoint, so this is not a soft target.

Verification (convergence + analytic)
    Adaptive quadrature is run at a crude, a loose and a tight tolerance. All
    three are compared against the closed form, the accuracy must improve
    monotonically, and each run's own error estimate must bound the true error
    it had no way of knowing.
"""

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad

EXACT = -(np.pi / 2.0) * np.log(2.0)
SETTINGS = [
    ("crude", {"epsabs": 1e-4, "epsrel": 1e-4, "limit": 3}),
    ("loose", {"epsabs": 1e-8, "limit": 200}),
    ("tight", {"epsabs": 1e-13, "limit": 400}),
]


def integrand(x):
    return np.log(np.sin(x))


print("closed form -(pi/2)*ln2 =", repr(float(EXACT)))

true_errors = []
for label, options in SETTINGS:
    with warnings.catch_warnings():  # the crude run is meant to struggle
        warnings.simplefilter("ignore", IntegrationWarning)
        value, reported = quad(integrand, 0.0, np.pi / 2.0, **options)
    true_error = abs(value - EXACT)
    true_errors.append(true_error)
    print(f"{label}: value = {value!r}  reported error = {reported!r}  true error = {true_error!r}")
    assert true_error <= reported, f"{label}: the reported error estimate was optimistic"

crude, loose, tight = true_errors
assert crude > 1e-5, "the crude run was supposed to be visibly wrong; the comparison is empty"
assert loose < 1e-8, f"loose quadrature missed the closed form by {loose}"
assert tight < 1e-12, f"tight quadrature missed the closed form by {tight}"
assert tight <= loose <= crude, "tightening the tolerance did not improve the answer"
print("VERIFIED: accuracy improves with tolerance and every error estimate was honest")
