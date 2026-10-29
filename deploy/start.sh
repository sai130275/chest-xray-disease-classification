#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${RADX_MODEL_PATH:-models/chest_xray_finetuned_best.pth}"
if [[ ! -f "$MODEL_PATH" ]]; then
  if [[ -z "${RADX_MODEL_URL:-}" ]]; then
    echo "RADX_MODEL_URL is required when the model checkpoint is not present at $MODEL_PATH" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$MODEL_PATH")"
  python - "$RADX_MODEL_URL" "$MODEL_PATH" <<'PY'
import sys
import urllib.request

url, path = sys.argv[1:]
request = urllib.request.Request(url, headers={"User-Agent": "RAD-X-render-deploy/1.0"})
with urllib.request.urlopen(request, timeout=180) as response, open(path, "wb") as output:
    while chunk := response.read(1024 * 1024):
        output.write(chunk)
PY
fi

exec uvicorn backend.app:app --host 0.0.0.0 --port "${PORT:-10000}"
