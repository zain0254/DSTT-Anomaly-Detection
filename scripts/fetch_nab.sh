#!/usr/bin/env bash
# Fetch NAB data, labels, scorer config and selected detector results at a
# pinned commit. Uses a sparse, blob-filtered checkout so only the needed files
# are downloaded. Output: data/nab/ (a git checkout of numenta/NAB).
#
# results/ is fetched only for the detectors used to validate our scorer port
# against results/final_results.json (see pipeline/nab_scorer.py).
set -euo pipefail

NAB_COMMIT="ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"   # numenta/NAB master as of 2026-11-18
VALIDATION_DETECTORS=(numenta windowedGaussian relativeEntropy null)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/nab"

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$DEST"
  git -C "$DEST" init -q
  git -C "$DEST" remote add origin https://github.com/numenta/NAB.git
fi

sparse=(data labels config nab)
for d in "${VALIDATION_DETECTORS[@]}"; do sparse+=("results/$d"); done
git -C "$DEST" sparse-checkout set --cone "${sparse[@]}"
git -C "$DEST" fetch -q --depth 1 --filter=blob:none origin "$NAB_COMMIT"
git -C "$DEST" -c advice.detachedHead=false checkout -q FETCH_HEAD

got="$(git -C "$DEST" rev-parse HEAD)"
if [ "$got" != "$NAB_COMMIT" ]; then
  echo "checked out $got, expected $NAB_COMMIT" >&2
  exit 1
fi

n_csv=$(find "$DEST/data" -name '*.csv' | wc -l)
echo "NAB $got"
echo "data files: $n_csv (expected 58)"
echo "$got" > "$DEST/PINNED_COMMIT"
