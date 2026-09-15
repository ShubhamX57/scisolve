"""Root finding: convergence order at a root of known multiplicity.

Problem
    f(x) = (x - 1)**3 * exp(x) has a triple root at x = 1 and no other roots,
    since exp is never zero.

Verification (convergence)
    Theory says plain Newton at a root of multiplicity m converges linearly
    with ratio 1 - 1/m, and that the modified iteration x - m*f/f' recovers
    quadratic convergence. Both rates are measured from the iterates and
    compared with the prediction, which came from calculus rather than from
    running the iteration again.
"""

import numpy as np

MULTIPLICITY = 3
ROOT = 1.0


def f(x):
    return (x - 1.0) ** 3 * np.exp(x)


def df(x):
    return (x - 1.0) ** 2 * np.exp(x) * (x + 2.0)


def newton(x0, factor, steps):
    xs = [x0]
    x = x0
    for _ in range(steps):
        fx, dfx = f(x), df(x)
        if dfx == 0.0:  # landed exactly on the root; the next step is 0/0
            break
        x = x - factor * fx / dfx
        xs.append(x)
    return np.array(xs)


plain_errors = np.abs(newton(1.5, 1.0, 40) - ROOT)
modified_errors = np.abs(newton(1.5, MULTIPLICITY, 10) - ROOT)

ratios = plain_errors[1:11] / plain_errors[:10]
measured_rate = float(np.mean(ratios[-5:]))
predicted_rate = 1.0 - 1.0 / MULTIPLICITY

print("plain Newton errors (first 6) =", repr([float(e) for e in plain_errors[:6]]))
print("measured linear rate =", repr(measured_rate))
print("predicted rate 1 - 1/m =", repr(predicted_rate))
print("modified Newton errors (first 6) =", repr([float(e) for e in modified_errors[:6]]))

quadratic = modified_errors[1:5] / modified_errors[:4] ** 2
print("modified e_next / e**2 =", repr([float(q) for q in quadratic]))

reached = np.flatnonzero(modified_errors < 1e-10)
assert reached.size, "modified Newton never reached 1e-10"
steps_to_1e10 = int(reached[0])
print("modified Newton steps to 1e-10 =", steps_to_1e10)
print(
    "plain Newton error after the same number of steps =",
    repr(float(plain_errors[steps_to_1e10])),
)

assert abs(measured_rate - predicted_rate) < 0.01, f"linear rate is {measured_rate}, not 2/3"
assert np.all(np.diff(plain_errors[:25]) < 0), "plain Newton is not converging monotonically"
assert np.ptp(quadratic) < 1.0, "modified Newton is not converging quadratically"
assert steps_to_1e10 <= 6, "multiplicity-aware Newton should get there in a handful of steps"
assert plain_errors[steps_to_1e10] > 1e-3, "plain Newton should still be far away at that point"
print("VERIFIED: both convergence orders match the theory for multiplicity 3")
