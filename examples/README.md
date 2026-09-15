# Examples

Each script solves a problem and then checks the answer some way that could
actually fail. Every one ends in an `assert`, so `tests/test_examples.py` can
re-run them all through the `Executor` in CI, with no API key, and a wrong
answer stops the build.

The rule these are written to: **the verification must be a different
computation, not the same one twice.** Solving an ODE twice with the same
solver and getting the same number verifies nothing.

| Example | Domain | What checks it |
|---|---|---|
| `damped_oscillator.py` | ODE | Closed-form underdamped solution, plus the logarithmic decrement measured from the peaks |
| `wigner_semicircle.py` | Linear algebra | KS test of LAPACK eigenvalues against the analytic semicircle CDF |
| `constrained_qp_kkt.py` | Optimization | An iterative NLP solver against a direct KKT linear solve; stationarity and feasibility residuals |
| `newton_multiplicity.py` | Root finding | Measured convergence rates against the theoretical 1 - 1/m and quadratic recovery |
| `bootstrap_vs_t_interval.py` | Statistics | Resampling against the closed-form t interval, which share no theory |
| `quadrature_log_sin.py` | Quadrature | A singular integral with a closed form, at three tolerances, each estimate checked for honesty |
| `heat_equation.py` | PDE | Crank-Nicolson against an exact two-mode separation-of-variables series, mode by mode |

`failures/` holds the opposite: cases that are wrong, or checks that pass when
they should not. See its README.

## Adding one

It needs a real check. Closed form, conservation law, independent algorithm, or
a limit with a known value. If the only way you can think of to verify it is to
run the same computation again, it is not an example yet.

Run one directly:

```bash
python examples/damped_oscillator.py
```
