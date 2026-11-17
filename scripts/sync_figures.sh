#!/usr/bin/env bash
# Mirror figures/ into the images repository (same folder structure under
# figures/), commit there and push. The images repo defaults to the sibling
# folder ../DSTT-Anomaly-Detection-images; override with IMAGES_REPO.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMG="${IMAGES_REPO:-$ROOT/../DSTT-Anomaly-Detection-images}"
IMG="$(cd "$IMG" && pwd)"

[ -d "$IMG/.git" ] || { echo "not a git repo: $IMG" >&2; exit 1; }
[ -d "$ROOT/figures" ] || { echo "no figures/ in $ROOT" >&2; exit 1; }

src_rev="$(git -C "$ROOT" rev-parse --short HEAD)"
git -C "$IMG" pull -q --ff-only
rm -rf "$IMG/figures"
cp -r "$ROOT/figures" "$IMG/figures"

git -C "$IMG" add -A figures
if git -C "$IMG" diff --cached --quiet; then
  echo "images repo already up to date"
  exit 0
fi
phases="$(cd "$ROOT/figures" && ls -d */ | tr -d '/' | paste -sd, -)"
git -C "$IMG" commit -q -m "Sync figures ($phases) from main repo at $src_rev"
git -C "$IMG" push -q origin HEAD
echo "pushed: $(git -C "$IMG" log --oneline -1)"
