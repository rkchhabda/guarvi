#!/usr/bin/env python
"""Pre-flight tool for Groww API's static-IP requirement.

Groww only accepts API requests coming from the IP you whitelist in the
Groww developer portal. This script:

1. Detects the current outbound (public) IP of this machine.
2. Compares it against GROWW_STATIC_IP (from .env / environment / config).
3. Optionally writes the detected IP into .env (--set).

Usage:
    python scripts/check_static_ip.py            # check only
    python scripts/check_static_ip.py --set      # save detected IP to .env
    python scripts/check_static_ip.py --json     # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

# Windows consoles default to cp1252 and crash on emoji/unicode output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Same endpoints as src/data/groww_loader.py; tried in order.
PUBLIC_IP_SERVICES = [
    "https://api.ipify.org",
    "https://ifconfig.me/ip",
    "https://checkip.amazonaws.com",
    "https://ipinfo.io/ip",
]

PLACEHOLDER_VALUES = {"", "your_static_ip_here", "your_static_ip"}


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from .env without overriding real env vars."""
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=path, override=False)
        return
    except ImportError:
        pass
    # Minimal fallback parser when python-dotenv is not installed.
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def get_public_ip(timeout: float = 5.0) -> str | None:
    """Detect the outbound public IP using PUBLIC_IP_SERVICES."""
    for url in PUBLIC_IP_SERVICES:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                ip = resp.read().decode("utf-8").strip()
            if ip:
                return ip
        except Exception:
            continue
    return None


def read_configured_static_ip() -> tuple[str, str]:
    """Return (ip, source) for the configured static IP, if any."""
    ip = os.getenv("GROWW_STATIC_IP", "").strip()
    if ip and ip not in PLACEHOLDER_VALUES:
        return ip, "environment/.env"
    # Fall back to config/data.yaml -> groww.static_ip
    try:
        import yaml

        with open(PROJECT_ROOT / "config" / "data.yaml", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        ip = (config.get("groww", {}).get("static_ip") or "").strip()
        if ip and ip not in PLACEHOLDER_VALUES:
            return ip, "config/data.yaml"
    except Exception:
        pass
    return "", "not configured"


def set_static_ip_in_env(ip: str, env_path: Path) -> Path:
    """Write GROWW_STATIC_IP=<ip> into .env (creating the file if needed)."""
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = [
            "# Groww API configuration",
            "# NEVER commit this file to version control!",
        ]

    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith("GROWW_STATIC_IP="):
            lines[i] = f"GROWW_STATIC_IP={ip}"
            replaced = True
            break
    if not replaced:
        lines.append("")
        lines.append("# Static IP whitelisted in the Groww developer portal")
        lines.append(f"GROWW_STATIC_IP={ip}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Groww API static-IP pre-flight check")
    parser.add_argument(
        "--set",
        action="store_true",
        help="Save the detected public IP to .env as GROWW_STATIC_IP",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text",
    )
    args = parser.parse_args()

    load_env_file(PROJECT_ROOT / ".env")

    public_ip = get_public_ip()
    configured_ip, source = read_configured_static_ip()
    match = bool(public_ip and configured_ip and public_ip == configured_ip)

    if args.set:
        if not public_ip:
            print("❌ Could not detect public IP; nothing to save.")
            return 1
        env_path = set_static_ip_in_env(public_ip, PROJECT_ROOT / ".env")
        print(f"💾 Saved GROWW_STATIC_IP={public_ip} to {env_path}")
        configured_ip, source = public_ip, ".env"
        match = True

    if args.json:
        print(
            json.dumps(
                {
                    "public_ip": public_ip,
                    "configured_static_ip": configured_ip,
                    "source": source,
                    "match": match,
                }
            )
        )
        return 0 if match or not configured_ip else 1

    print("=" * 60)
    print("Groww API — Static IP Pre-flight Check")
    print("=" * 60)
    print(f"  Current public IP : {public_ip or 'DETECTION FAILED'}")
    print(f"  Configured IP     : {configured_ip or '(not configured)'}  [{source}]")
    print()

    if not configured_ip:
        print("ℹ️  No static IP configured yet.")
        print(f"   → Whitelist {public_ip} in the Groww developer portal:")
        print("     Groww Developer Portal → API keys → Whitelist IP")
        print("   → Then save it locally:  python scripts/check_static_ip.py --set")
        return 0

    if match:
        print("✅ MATCH — outbound IP equals the whitelisted static IP.")
        print("   Groww API calls from this machine should be accepted.")
        return 0

    print("❌ MISMATCH — Groww will reject API calls from this machine.")
    print("   Fix options:")
    print(f"   a) Whitelist the current IP {public_ip} in the Groww developer portal.")
    print("   b) Run the bot from the machine/network whose IP is whitelisted")
    print("      (e.g., your cloud VPS with a static/elastic IP).")
    print("   c) Route traffic through a static-IP proxy/VPN and set HTTPS_PROXY.")
    print("   d) Update the saved IP if your ISP assigned you a new one:")
    print("      python scripts/check_static_ip.py --set")
    return 1


if __name__ == "__main__":
    sys.exit(main())
