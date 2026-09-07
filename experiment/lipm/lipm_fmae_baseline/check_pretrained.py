#!/usr/bin/env python3
"""Check whether public LiPM pretrained weights are available."""

import json
import os
import sys
import urllib.request
from pathlib import Path


LIPM_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CLONE = LIPM_ROOT / "LiPM"
RELEASE_API = "https://api.github.com/repos/JuRenGithub/LiPM/releases"
WEIGHT_SUFFIXES = (".pth", ".pt", ".ckpt", ".bin", ".safetensors")


def local_weight_files():
    if not OFFICIAL_CLONE.exists():
        return []
    return [
        p for p in OFFICIAL_CLONE.rglob("*")
        if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES and ".git" not in p.parts
    ]


def release_assets():
    try:
        with urllib.request.urlopen(RELEASE_API, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"Could not query GitHub releases: {exc}")
        return None

    assets = []
    for release in payload:
        for asset in release.get("assets", []):
            assets.append({
                "release": release.get("tag_name"),
                "name": asset.get("name"),
                "url": asset.get("browser_download_url"),
                "size": asset.get("size"),
            })
    return assets


def main():
    print(f"Official clone: {OFFICIAL_CLONE}")
    weights = local_weight_files()
    if weights:
        print("Local LiPM weight files found:")
        for path in weights:
            print(f"  {path}")
    else:
        print("No local LiPM weight files found in the official clone.")

    assets = release_assets()
    if assets is None:
        return 2
    if assets:
        print("GitHub release assets:")
        for asset in assets:
            print(f"  [{asset['release']}] {asset['name']} {asset['size']} {asset['url']}")
    else:
        print("GitHub release assets: none.")

    if not weights and not assets:
        print("Result: no public pretrained LiPM checkpoint is currently available from the checked sources.")
        return 1
    print("Result: candidate checkpoint asset found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
