"""ODE: damped oscillator, numerical solution vs the analytic solution.

Problem
    x'' + 2*zeta*omega*x' + omega**2 * x = 0,  x(0) = 1,  x'(0) = 0.

Verification (analytic)
    An adaptive RK45 integration is compared against the closed-form
    underdamped solution, and the ratio of successive peak heights is compared
    against the logarithmic decrement 2*pi*zeta/sqrt(1-zeta**2). The stepper and
    the formula share no code.
"""

import numpy as np
from scipy.integrate import solve_ivp

ZETA, OMEGA = 0.08, 3.0
OMEGA_D = OMEGA * np.sqrt(1.0 - ZETA**2)
T_END = 12.0


def rhs(t, y):
    return [y[1], -2.0 * ZETA * OMEGA * y[1] - OMEGA**2 * y[0]]


def exact(t):
    decay = np.exp(-ZETA * OMEGA * t)
    return decay * (np.cos(OMEGA_D * t) + (ZETA * OMEGA / OMEGA_D) * np.sin(OMEGA_D * t))


t_eval = np.linspace(0.0, T_END, 6001)
sol = solve_ivp(rhs, (0.0, T_END), [1.0, 0.0], t_eval=t_eval, rtol=1e-11, atol=1e-13)
assert sol.success, sol.message

x = sol.y[0]
max_error = float(np.max(np.abs(x - exact(t_eval))))
print("max |RK45 - analytic| =", repr(max_error))

peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
heights = x[peaks]
log_decrements = np.log(heights[:-1] / heights[1:])
predicted = 2.0 * np.pi * ZETA / np.sqrt(1.0 - ZETA**2)

print("peaks found =", len(peaks))
print("measured log decrement =", repr(float(np.mean(log_decrements))))
print("predicted log decrement =", repr(float(predicted)))
print("worst deviation =", repr(float(np.max(np.abs(log_decrements - predicted)))))

assert max_error < 1e-8, f"RK45 disagrees with the closed form: {max_error}"
assert len(peaks) >= 4, "not enough peaks to measure the decay"
assert np.max(np.abs(log_decrements - predicted)) < 1e-3, "envelope does not decay as predicted"
print("VERIFIED: trajectory and decay rate both match the closed form")
