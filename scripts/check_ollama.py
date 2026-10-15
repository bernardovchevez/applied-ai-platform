#!/usr/bin/env python3
"""Verify Ollama server is running before executing platform commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from shared.config import get_env
from shared.ollama_client import base_url, chat_model, embed_model


def main() -> int:
    url = base_url()
    print(f"Checking Ollama at {url} ...")

    try:
        r = httpx.get(f"{url}/api/tags", timeout=3.0)
        r.raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        print("\n❌ Could not connect to Ollama server.\n")
        print("Fix (in order):")
        print("  1. Install: https://ollama.com/download/mac")
        print("  2. Open the Ollama app from Applications (menu bar icon must appear)")
        print("  3. Wait ~10 seconds, then run:  ollama list")
        print("  4. Pull models:")
        print(f"       ollama pull {chat_model()}")
        print(f"       ollama pull {embed_model()}")
        print(f"\nDetail: {exc}")
        return 1

    models = [m["name"] for m in r.json().get("models", [])]
    print(f"✅ Ollama server is running ({len(models)} model(s) installed)")

    need = {chat_model(), embed_model()}
    missing = [m for m in need if not any(name == m or name.startswith(f"{m}:") for name in models)]
    if missing:
        print("\n⚠️  Missing models. Run:")
        for m in missing:
            print(f"     ollama pull {m}")
        return 1

    print(f"✅ Models ready: {chat_model()}, {embed_model()}")
    return 0


if __name__ == "__main__":
    # Try to open Ollama app on macOS if installed
    if len(sys.argv) > 1 and sys.argv[1] == "--launch":
        app = Path("/Applications/Ollama.app")
        if app.exists():
            subprocess.run(["open", "-a", "Ollama"], check=False)
            print("Launched Ollama.app — wait a few seconds and re-run check.")
        else:
            print("Ollama.app not found in /Applications. Download from https://ollama.com/download/mac")
        sys.exit(0)

    sys.exit(main())
