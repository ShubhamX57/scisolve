"""CLOSED: an answer could claim more precision than was ever printed.

Found by running the CLI end to end, not by reading the code. numpy truncates
array display by default, so stdout held eight significant figures while the
answer quoted seventeen. check_grounding matched to a relative tolerance of
1e-3, so nine digits that nothing had computed were reported as grounded.

First mitigated in the system prompt, which asks for full-precision printing.
That reduced how often it arose and closed nothing. Now closed in the checker:
a literal within rtol of a printed number but claiming more significant figures
than that number was printed to is reported in GroundingReport.overprecise and
the answer is not grounded. The value is traceable; the extra digits are not.

This file stays as a regression guard. If it starts failing, the hole is open
again.
"""

import numpy as np

from scisolve.grounding import check_grounding

matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
eigenvalues = np.linalg.eigvalsh(matrix)

printed = f"eigenvalues {eigenvalues}"  # numpy's default display
print("what the session printed:", printed)

overclaimed = (
    f"The eigenvalues are {float(eigenvalues[0])!r} and {float(eigenvalues[1])!r}."
)
print("what the answer claimed:", overclaimed)

report = check_grounding(overclaimed, [printed])
print("grounded =", report.grounded)
print("overprecise =", report.overprecise)
print("unmatched =", report.unmatched)

# rounding to what was actually printed is accepted
rounded = "The eigenvalues are 1.38196601 and 3.61803399."
rounded_report = check_grounding(rounded, [printed])
print("rounded answer grounded =", rounded_report.grounded)

# and so is the full-precision answer, once the code prints full precision
full = f"eigenvalues {[float(v) for v in eigenvalues]!r}"
print(
    "printing at full precision grounds it again =",
    check_grounding(overclaimed, [full]).grounded,
)

assert not report.grounded, "REGRESSION: the overprecise answer was accepted again"
assert len(report.overprecise) == 2, "REGRESSION: the extra digits were not flagged"
assert report.unmatched == (), "the values are traceable; only the precision was not"
assert rounded_report.grounded, "rounding to the printed digits should still pass"
assert check_grounding(overclaimed, [full]).grounded, "full precision should pass"
print("CLOSED: over-claimed precision is caught, and both honest forms still pass")
