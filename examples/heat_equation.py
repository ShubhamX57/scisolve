"""PDE: the heat equation against its separation-of-variables series.

Problem
    u_t = alpha * u_xx on [0, 1] with u(0, t) = u(1, t) = 0 and
    u(x, 0) = sin(pi x) + 0.5 sin(3 pi x).

Verification (analytic + invariant)
    The initial condition is exactly two Fourier modes, so the series solution
    is exact and finite: each mode decays as exp(-alpha (n pi)**2 t). A
    Crank-Nicolson march is compared against it, and the decay rate of each
    mode is recovered from the numerical solution by projection and compared
    with the analytic eigenvalue.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

ALPHA = 0.5
NX = 201
T_END = 0.1
NT = 400

x = np.linspace(0.0, 1.0, NX)
dx = x[1] - x[0]
dt = T_END / NT
interior = x[1:-1]


def initial(xs):
    return np.sin(np.pi * xs) + 0.5 * np.sin(3.0 * np.pi * xs)


def exact(xs, t):
    return np.exp(-ALPHA * (np.pi**2) * t) * np.sin(np.pi * xs) + 0.5 * np.exp(
        -ALPHA * (9.0 * np.pi**2) * t
    ) * np.sin(3.0 * np.pi * xs)


n = interior.size
laplacian = (
    np.diag(-2.0 * np.ones(n))
    + np.diag(np.ones(n - 1), 1)
    + np.diag(np.ones(n - 1), -1)
) / dx**2
identity = np.eye(n)
r = ALPHA * dt / 2.0
left = lu_factor(identity - r * laplacian)
right = identity + r * laplacian

u = initial(interior)
mode1, mode3 = np.sin(np.pi * interior), np.sin(3.0 * np.pi * interior)
norm1, norm3 = mode1 @ mode1, mode3 @ mode3
history = [(0.0, float(u @ mode1 / norm1), float(u @ mode3 / norm3))]

for step in range(NT):
    u = lu_solve(left, right @ u)
    t = (step + 1) * dt
    history.append((t, float(u @ mode1 / norm1), float(u @ mode3 / norm3)))

max_error = float(np.max(np.abs(u - exact(interior, T_END))))
print("grid points =", NX, " time steps =", NT)
print("max |Crank-Nicolson - series| at t =", T_END, "is", repr(max_error))

times = np.array([h[0] for h in history])
amp1 = np.array([h[1] for h in history])
amp3 = np.array([h[2] for h in history])

rate1 = float(np.polyfit(times, np.log(amp1), 1)[0])
rate3 = float(np.polyfit(times, np.log(amp3 / 0.5), 1)[0])
print("measured decay rate, mode 1 =", repr(rate1), " analytic =", repr(-ALPHA * np.pi**2))
print("measured decay rate, mode 3 =", repr(rate3), " analytic =", repr(-ALPHA * 9.0 * np.pi**2))
print("final amplitudes =", repr(float(amp1[-1])), repr(float(amp3[-1])))

assert max_error < 1e-5, f"the march departs from the exact series by {max_error}"
assert abs(rate1 / (-ALPHA * np.pi**2) - 1.0) < 2e-3, "mode 1 decays at the wrong rate"
assert abs(rate3 / (-ALPHA * 9.0 * np.pi**2) - 1.0) < 2e-2, "mode 3 decays at the wrong rate"
assert amp1[-1] > 0 and 0 < amp3[-1] < amp1[-1], "mode ordering is wrong: mode 3 must die faster"
print("VERIFIED: numerical march and the exact series agree, mode by mode")
