"""Optimization: a constrained QP solved twice, and its KKT residuals.

Problem
    minimise 0.5 * x' Q x - b' x  subject to  A x = c.

Verification (invariant)
    SLSQP, an iterative nonlinear programming method, is checked against the
    direct solution of the KKT linear system, and the stationarity and
    feasibility residuals of the SLSQP point are driven to zero. An iterative
    optimiser and a linear solve are genuinely different computations.
"""

import numpy as np
from scipy.optimize import minimize

Q = np.array([[2.0, 0.5], [0.5, 1.0]])
B = np.array([1.0, 2.0])
A = np.array([[1.0, 1.0]])
C = np.array([1.0])


def objective(x):
    return 0.5 * x @ Q @ x - B @ x


def gradient(x):
    return Q @ x - B


# Direct route: solve the KKT system [[Q, A'], [A, 0]] [x; lam] = [b; c].
n, m = Q.shape[0], A.shape[0]
kkt = np.block([[Q, A.T], [A, np.zeros((m, m))]])
rhs = np.concatenate([B, C])
kkt_solution = np.linalg.solve(kkt, rhs)
x_kkt, lam_kkt = kkt_solution[:n], kkt_solution[n:]

# Iterative route: SLSQP, which knows nothing about the linear system above.
result = minimize(
    objective,
    x0=np.array([5.0, -5.0]),
    jac=gradient,
    constraints=[{"type": "eq", "fun": lambda x: A @ x - C, "jac": lambda x: A}],
    method="SLSQP",
    options={"ftol": 1e-12, "maxiter": 200},
)
assert result.success, result.message
x_slsqp = result.x

print("x from KKT linear system =", repr(x_kkt.tolist()))
print("x from SLSQP             =", repr(x_slsqp.tolist()))
print("objective at KKT point   =", repr(float(objective(x_kkt))))
print("objective at SLSQP point =", repr(float(objective(x_slsqp))))

disagreement = float(np.max(np.abs(x_kkt - x_slsqp)))
stationarity = float(np.linalg.norm(Q @ x_slsqp - B + A.T @ lam_kkt))
feasibility = float(np.linalg.norm(A @ x_slsqp - C))
print("max |x_kkt - x_slsqp| =", repr(disagreement))
print("stationarity residual =", repr(stationarity))
print("primal feasibility residual =", repr(feasibility))

assert disagreement < 1e-7, "the two methods disagree about the minimiser"
assert stationarity < 1e-7, "KKT stationarity is not satisfied"
assert feasibility < 1e-10, "the constraint is not satisfied"
print("VERIFIED: two independent methods agree and the KKT residuals vanish")
