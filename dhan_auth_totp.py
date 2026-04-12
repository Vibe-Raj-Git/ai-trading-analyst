# dhan_auth_totp.py
# 09-Feb: This new file is created to handle Dhan access token using TOTP
# 28-Mar: Updated with better error handling and debug output

import os
import time
import requests
import pyotp
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# Load environment variables - try multiple paths
load_dotenv()  # loads from current directory

# Also try to load from the same directory as this file
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"[DHAN_AUTH] Loaded .env from: {env_path}")

IST = ZoneInfo("Asia/Kolkata")

# Get credentials
DHAN_CLIENT_ID = os.environ.get("DHAN_CLIENT_ID") or ""
DHAN_PIN = os.environ.get("DHAN_PIN") or ""
DHAN_TOTP_SECRET = os.environ.get("DHAN_TOTP_SECRET") or ""

# Debug: Show what we loaded
print("=" * 50)
print("[DHAN_AUTH] Environment Variables Status:")
print(f"  DHAN_CLIENT_ID: {'✓ SET' if DHAN_CLIENT_ID else '✗ MISSING'} (value: {DHAN_CLIENT_ID[:4] + '...' if DHAN_CLIENT_ID else 'None'})")
print(f"  DHAN_PIN: {'✓ SET' if DHAN_PIN else '✗ MISSING'} (length: {len(DHAN_PIN) if DHAN_PIN else 0})")
print(f"  DHAN_TOTP_SECRET: {'✓ SET' if DHAN_TOTP_SECRET else '✗ MISSING'} (length: {len(DHAN_TOTP_SECRET) if DHAN_TOTP_SECRET else 0})")
print("=" * 50)

# Validate credentials
if not DHAN_CLIENT_ID:
    print("[DHAN_AUTH] WARNING: DHAN_CLIENT_ID not set in environment!")
if not DHAN_PIN:
    print("[DHAN_AUTH] WARNING: DHAN_PIN not set in environment!")
if not DHAN_TOTP_SECRET:
    print("[DHAN_AUTH] WARNING: DHAN_TOTP_SECRET not set in environment!")

DHAN_AUTH_URL = "https://auth.dhan.co/app/generateAccessToken"

_cached_token: str | None = None
_cached_expiry: float | None = None  # epoch seconds


def _generate_totp() -> str:
    """Generate TOTP code from secret"""
    if not DHAN_TOTP_SECRET:
        raise ValueError("DHAN_TOTP_SECRET is not set. Cannot generate TOTP code.")
    
    try:
        totp = pyotp.TOTP(DHAN_TOTP_SECRET)
        code = totp.now()
        print(f"[DHAN_AUTH] Generated TOTP code: {code}")
        return code
    except Exception as e:
        print(f"[DHAN_AUTH] Error generating TOTP: {e}")
        raise


def _fetch_new_access_token() -> str:
    """Fetch new access token from Dhan"""
    if not DHAN_CLIENT_ID:
        raise ValueError("DHAN_CLIENT_ID is not set. Cannot authenticate.")
    if not DHAN_PIN:
        raise ValueError("DHAN_PIN is not set. Cannot authenticate.")
    
    totp_code = _generate_totp()

    # Use params (query parameters) as per Dhan API documentation
    params = {
        "dhanClientId": DHAN_CLIENT_ID,
        "pin": DHAN_PIN,
        "totp": totp_code,
    }
    
    print(f"[DHAN_AUTH] Requesting token with Client ID: {DHAN_CLIENT_ID[:4]}...")
    print(f"[DHAN_AUTH] PIN length: {len(DHAN_PIN)}")
    print(f"[DHAN_AUTH] TOTP length: {len(totp_code)}")
    
    try:
        resp = requests.post(DHAN_AUTH_URL, params=params, timeout=30)
        print(f"[DHAN_AUTH] HTTP status: {resp.status_code}")
        
        # Check if response is JSON
        try:
            data = resp.json()
        except Exception as e:
            print(f"[DHAN_AUTH] Failed to parse JSON response: {e}")
            print(f"[DHAN_AUTH] Raw response: {resp.text[:500]}")
            raise RuntimeError(f"Invalid response from Dhan: {resp.text[:200]}")
        
        if resp.status_code != 200:
            print(f"[DHAN_AUTH] Error response: {data}")
            raise RuntimeError(f"Dhan auth error (status {resp.status_code}): {data}")

        if "accessToken" not in data:
            print(f"[DHAN_AUTH] Response missing accessToken: {data}")
            raise RuntimeError(f"Dhan auth missing accessToken field: {data}")

        access_token = data["accessToken"]
        expiry_iso = data.get("expiryTime")
        print(f"[DHAN_AUTH] Received access token (truncated): {access_token[:20]}...")
        print(f"[DHAN_AUTH] Expiry (broker time): {expiry_iso}")

        global _cached_token, _cached_expiry
        _cached_token = access_token
        # 23 hour safety window (tokens typically last 24 hours)
        _cached_expiry = time.time() + 23 * 60 * 60
        
        print(f"[DHAN_AUTH] Token cached until: {datetime.fromtimestamp(_cached_expiry).strftime('%Y-%m-%d %H:%M:%S')}")

        return access_token
        
    except requests.exceptions.RequestException as e:
        print(f"[DHAN_AUTH] Request failed: {e}")
        raise RuntimeError(f"Dhan auth request failed: {e}")
    except Exception as e:
        print(f"[DHAN_AUTH] Unexpected error: {e}")
        raise


def get_access_token() -> str:
    """
    Return a valid Dhan access token, refreshing via TOTP if needed.
    """
    global _cached_token, _cached_expiry
    
    # Check if we have a valid cached token
    if _cached_token and _cached_expiry:
        time_left = _cached_expiry - time.time()
        if time_left > 0:
            print(f"[DHAN_AUTH] Using cached token (expires in {time_left/3600:.1f} hours)")
            return _cached_token
        else:
            print(f"[DHAN_AUTH] Cached token expired (expired {abs(time_left/3600):.1f} hours ago)")
    
    print("[DHAN_AUTH] Fetching new access token...")
    return _fetch_new_access_token()


# Test function to verify credentials
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Testing Dhan Authentication")
    print("=" * 50)
    try:
        token = get_access_token()
        print(f"\n✓ SUCCESS! Token obtained: {token[:20]}...")
        print(f"✓ Full token length: {len(token)} characters")
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that your .env file exists in the project root")
        print("2. Verify DHAN_CLIENT_ID is correct (should be a number)")
        print("3. Verify DHAN_PIN is correct (your Dhan login PIN)")
        print("4. Verify DHAN_TOTP_SECRET is correct (from Google Authenticator setup)")
        print("5. Make sure you have internet connectivity")