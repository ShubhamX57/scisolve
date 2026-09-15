"""KNOWN FAILURE: an answer can claim more precision than was ever printed.

Found while running the CLI end to end. numpy truncates array display by
default, so stdout held eight significant figures while the answer quoted
seventeen. check_grounding matches to a relative tolerance of 1e-3, so the
extra digits matched something that was never computed to that precision.

The check is doing what it documents -- it verifies provenance to within rtol,
not digit by digit. But "grounded" reads to a user as "these digits came from
the session", and here nine of them did not.

Mitigation in place: the system prompt now asks for full-precision printing.
That reduces how often this arises; it does not close it.
Closing it properly means comparing significant digits against the matched
output value, which changes what grounded means and is a decision, not a patch.
"""

import numpy as np

from scisolve.grounding import check_grounding

matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
eigenvalues = np.linalg.eigvalsh(matrix)

printed = f"eigenvalues {eigenvalues}"  # numpy's default display
print("what the session actually printed:", printed)

overclaimed = (
    f"The eigenvalues are {float(eigenvalues[0])!r} and {float(eigenvalues[1])!r}."
)
print("what the answer claimed:", overclaimed)

report = check_grounding(overclaimed, [printed])
print("grounded =", report.grounded)
print("matched =", report.matched)

digits_printed = len("1.38196601".replace(".", ""))
digits_claimed = len(repr(float(eigenvalues[0])).replace(".", ""))
print("significant digits printed =", digits_printed)
print("significant digits claimed =", digits_claimed)

assert report.grounded, "this entry is out of date: the hole appears to be closed"
assert digits_claimed > digits_printed
print(
    "STILL FAILING: an answer with",
    digits_claimed - digits_printed,
    "unverified digits was accepted",
)
