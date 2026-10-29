#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${RADX_MODEL_PATH:-models/chest_xray_finetuned_best.pth}"
if [[ ! -f "$MODEL_PATH" ]]; then
  if [[ -z "${DROPBOX_APP_KEY:-}" || -z "${DROPBOX_APP_SECRET:-}" || -z "${DROPBOX_REFRESH_TOKEN:-}" ]]; then
    echo "DROPBOX_APP_KEY, DROPBOX_APP_SECRET, and DROPBOX_REFRESH_TOKEN are required to download the private checkpoint." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$MODEL_PATH")"
  python - "$DROPBOX_FILE_PATH" "$MODEL_PATH" "$DROPBOX_APP_KEY" "$DROPBOX_APP_SECRET" "$DROPBOX_REFRESH_TOKEN" <<'PY'
import base64
import hashlib
import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request

dropbox_path, path, app_key, app_secret, refresh_token = sys.argv[1:]
token_body = urllib.parse.urlencode({
    "grant_type": "refresh_token",
    "refresh_token": refresh_token,
}).encode()
basic = base64.b64encode(f"{app_key}:{app_secret}".encode()).decode()
token_request = urllib.request.Request(
    "https://api.dropboxapi.com/oauth2/token",
    data=token_body,
    headers={
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
    },
    method="POST",
)
with urllib.request.urlopen(token_request, timeout=30) as response:
    access_token = json.load(response)["access_token"]

download_request = urllib.request.Request(
    "https://content.dropboxapi.com/2/files/download",
    headers={
        "Authorization": f"Bearer {access_token}",
        "Dropbox-API-Arg": json.dumps({"path": dropbox_path}),
    },
)
digest = hashlib.sha256()
temporary_path = None
try:
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as output:
        temporary_path = output.name
        with urllib.request.urlopen(download_request, timeout=180) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
    expected_sha256 = "e795b4256134fcaedf0748ef30a951c7700647c6233d6868e95c2002871ee4d3"
    if digest.hexdigest() != expected_sha256:
        raise ValueError("Downloaded checkpoint failed its SHA-256 integrity check.")
    os.replace(temporary_path, path)
except Exception:
    if temporary_path and os.path.exists(temporary_path):
        os.unlink(temporary_path)
    raise
PY
fi

exec uvicorn backend.app:app --host 0.0.0.0 --port "${PORT:-10000}"
