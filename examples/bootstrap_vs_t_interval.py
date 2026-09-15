"""Statistics: a bootstrap CI for a Gaussian mean against the closed form.

Problem
    A 95% confidence interval for the mean of a normal sample.

Verification (independent_method)
    The percentile bootstrap resamples the data thousands of times and never
    touches the t distribution; the closed form uses the t quantile and never
    resamples. For Gaussian data they must agree, and the bootstrap standard
    error must match s/sqrt(n).
"""

import numpy as np
from scipy import stats

N = 400
B = 8000
TRUE_MEAN, TRUE_SD = 5.0, 2.0
rng = np.random.default_rng(20260916)

sample = rng.normal(TRUE_MEAN, TRUE_SD, size=N)
mean = float(sample.mean())
sd = float(sample.std(ddof=1))

# Closed form: mean +/- t_{0.975, n-1} * s / sqrt(n)
t_crit = float(stats.t.ppf(0.975, N - 1))
half_width = t_crit * sd / np.sqrt(N)
t_low, t_high = mean - half_width, mean + half_width

# Bootstrap: resample, take percentiles, use no distributional theory at all.
indices = rng.integers(0, N, size=(B, N))
boot_means = sample[indices].mean(axis=1)
b_low, b_high = (float(v) for v in np.percentile(boot_means, [2.5, 97.5]))

print("n =", N, " bootstrap resamples =", B)
print("sample mean =", repr(mean), " sample sd =", repr(sd))
print("t interval         =", repr(t_low), repr(t_high))
print("bootstrap interval =", repr(b_low), repr(b_high))

analytic_se = sd / np.sqrt(N)
boot_se = float(boot_means.std(ddof=1))
print("analytic standard error =", repr(float(analytic_se)))
print("bootstrap standard error =", repr(boot_se))

se_gap = abs(boot_se - analytic_se) / analytic_se
low_gap = abs(b_low - t_low) / half_width
high_gap = abs(b_high - t_high) / half_width
print("relative standard-error gap =", repr(float(se_gap)))
print(
    "endpoint gaps as a fraction of the half-width =",
    repr(float(low_gap)),
    repr(float(high_gap)),
)

assert se_gap < 0.05, f"bootstrap SE disagrees with s/sqrt(n): {se_gap}"
assert low_gap < 0.05 and high_gap < 0.05, "the two intervals do not agree"
assert t_low < TRUE_MEAN < t_high, "the interval missed the true mean (1 run in 20 will)"
print("VERIFIED: resampling and the t distribution give the same interval")
