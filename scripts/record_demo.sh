#!/usr/bin/env bash
# Record the terminal demo for the README.
#
# Needs asciinema to record and agg to turn the cast into a GIF:
#   brew install asciinema agg        # macOS
#   pipx install asciinema            # elsewhere; agg from its GitHub releases
#
# This makes a real API call. ANTHROPIC_API_KEY must be set.

set -euo pipefail

PROBLEM="${1:-Integrate ln(sin x) from 0 to pi/2 and verify the result against the closed form.}"
OUT_DIR="docs"
CAST="${OUT_DIR}/demo.cast"
GIF="${OUT_DIR}/demo.gif"

if ! command -v asciinema >/dev/null 2>&1; then
  echo "asciinema is not installed; see the header of this script." >&2
  exit 2
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ANTHROPIC_API_KEY is not set; this records a real run." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
rm -f "${CAST}"

asciinema rec "${CAST}" \
  --cols 100 --rows 34 \
  --title "scisolve" \
  --command "scisolve \"${PROBLEM}\""

echo
echo "recorded ${CAST}"

if command -v agg >/dev/null 2>&1; then
  agg --font-size 15 --theme asciinema "${CAST}" "${GIF}"
  echo "wrote ${GIF}"
else
  echo "agg not found; convert with: agg ${CAST} ${GIF}"
fi

echo
echo "Watch it back before committing. A demo that shows a run you would not"
echo "stand behind is worse than no demo."
