"""Create an offline Dropbox refresh token for the Render model downloader.

The app key and secret are entered locally and are never saved by this script.
The printed refresh token should be copied directly into Render's secret field.
"""

from __future__ import annotations

import base64
import getpass
import json
import urllib.error
import urllib.parse
import urllib.request


def main() -> None:
    app_key = input("Dropbox app key: ").strip()
    app_secret = getpass.getpass("Dropbox app secret (input hidden): ").strip()
    if not app_key or not app_secret:
        raise SystemExit("Both app key and app secret are required.")

    auth_url = "https://www.dropbox.com/oauth2/authorize?" + urllib.parse.urlencode(
        {
            "client_id": app_key,
            "response_type": "code",
            "token_access_type": "offline",
            "scope": "files.content.read",
        }
    )
    print("\nOpen this Dropbox authorization URL, sign in, and approve files.content.read:")
    print(auth_url)
    code = input("\nPaste the one-time authorization code shown by Dropbox: ").strip()
    if not code:
        raise SystemExit("An authorization code is required.")

    payload = urllib.parse.urlencode(
        {"grant_type": "authorization_code", "code": code}
    ).encode()
    basic = base64.b64encode(f"{app_key}:{app_secret}".encode()).decode()
    request = urllib.request.Request(
        "https://api.dropboxapi.com/oauth2/token",
        data=payload,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"Dropbox rejected the authorization exchange (HTTP {error.code}). "
            "Check the app key, secret, and one-time code, then retry."
        ) from None

    refresh_token = result.get("refresh_token")
    if not refresh_token:
        raise SystemExit(
            "Dropbox did not return a refresh token. Check that offline access "
            "was requested and granted."
        )

    print("\nCopy this value directly into Render's DROPBOX_REFRESH_TOKEN secret:")
    print(refresh_token)
    print("\nThe refresh token is sensitive. Do not commit it or paste it into chat.")


if __name__ == "__main__":
    main()
