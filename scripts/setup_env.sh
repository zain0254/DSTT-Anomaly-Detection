#!/usr/bin/env bash
# Create the pipeline venv in envs/pipeline from requirements.txt and save the
# resolved package list to results/environment/pip_freeze.txt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV="$ROOT/envs/pipeline"

if command -v py >/dev/null 2>&1; then PY=(py -3.12); else PY=(python3.12); fi
[ -d "$ENV" ] || "${PY[@]}" -m venv "$ENV"
if [ -x "$ENV/Scripts/python.exe" ]; then VPY="$ENV/Scripts/python.exe"; else VPY="$ENV/bin/python"; fi

"$VPY" -m pip install -q --upgrade pip
"$VPY" -m pip install -q -r "$ROOT/requirements.txt"
mkdir -p "$ROOT/results/environment"
{
  echo "# pip freeze of envs/pipeline, $("$VPY" --version)"
  "$VPY" -m pip freeze
} > "$ROOT/results/environment/pip_freeze.txt"
echo "env ready: $VPY"
