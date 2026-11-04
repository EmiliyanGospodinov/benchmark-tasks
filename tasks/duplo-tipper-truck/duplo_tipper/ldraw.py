"""Fetch a small attributed LDraw library (official, then unofficial) and flatten its triangle/quad geometry.

LDraw coordinates are in LDU (0.4 mm), with Y pointing down. Returned vertices
use metres and a right-handed Z-up frame: (x, y, z) = (X, Z, -Y) * 0.0004.
Edges and conditional edges are omitted. The selected untextured parts use a
uniform task colour; this is not a general importer for printed/textured parts.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # task folder
CACHE = ROOT / "assets" / "ldraw"
BASES = ("https://library.ldraw.org/library/official/", "https://library.ldraw.org/library/unofficial/")
_VERIFIED = set()
ORIGIN = {}  # cached file -> URL it was downloaded from (this session)


def fetch(name, stack=()):
    """Download a part and referenced subparts/primitives, retaining original headers."""
    name = name.replace("\\", "/").lower()
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or name in stack:
        raise ValueError(f"Invalid or cyclic LDraw dependency: {name}")
    candidates = [f"parts/{name}", f"p/{name}"]
    for candidate in candidates:
        path = CACHE / candidate
        if path.is_file():
            # A download may have stopped after writing its parent. Check the
            # cached parent's children as well so an interrupted fetch resumes.
            if name not in _VERIFIED:
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    tokens = line.split()
                    if tokens and tokens[0] == "1":
                        fetch(" ".join(tokens[14:]), (*stack, name))
                _VERIFIED.add(name)
            return path
    for base, candidate in ((b, c) for b in BASES for c in candidates):
        url = base + candidate
        raw = None
        for attempt in range(8):
            try:
                with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as response:
                    raw = response.read()
                break
            except HTTPError as error:
                if error.code == 404:
                    break
                if error.code != 429:
                    raise
                time.sleep(5 * (attempt + 1))  # the library rate-limits bursts
        if raw is None:
            continue
        text = raw.decode("utf-8-sig")
        if not text.lstrip().startswith("0 "):
            raise ValueError(f"Unexpected response for {url}")
        path = CACHE / candidate
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        ORIGIN[candidate] = url
        print(f"Fetched {candidate}", flush=True)
        for line in text.splitlines():
            tokens = line.split()
            if tokens and tokens[0] == "1":
                fetch(" ".join(tokens[14:]), (*stack, name))
        _VERIFIED.add(name)
        return path
    raise FileNotFoundError(name)


def geometry(name):
    """Return vertices (N,3) and triangular faces (M,3) without importing Isaac."""
    triangles = []

    def walk(part, matrix, offset, invert=False):
        paths = [CACHE / "parts" / part, CACHE / "p" / part]
        path = next((p for p in paths if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Fetch LDraw dependency first: {part}")
        ccw, invert_next = True, False
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "0":
                if "BFC" in tokens:
                    if "INVERTNEXT" in tokens:
                        invert_next = True
                    elif "CCW" in tokens:
                        ccw = True
                    elif "CW" in tokens:
                        ccw = False
                continue
            kind = int(tokens[0])
            if kind == 1:
                values = np.array(tokens[2:14], dtype=float)
                translation, rotation = values[:3], values[3:].reshape(3, 3)
                mirror = np.linalg.det(rotation) < 0
                walk(" ".join(tokens[14:]).replace("\\", "/").lower(),
                     matrix @ rotation, matrix @ translation + offset,
                     invert ^ invert_next ^ bool(mirror))
                invert_next = False
            elif kind in (3, 4):
                points = np.array(tokens[2:], dtype=float).reshape(kind, 3)
                points = points @ matrix.T + offset
                if invert ^ (not ccw):
                    points = points[::-1]
                triangles.append(points[[0, 1, 2]])
                if kind == 4:
                    triangles.append(points[[0, 2, 3]])

    walk(name.lower(), np.eye(3), np.zeros(3))
    vertices = np.concatenate(triangles)
    vertices = vertices[:, [0, 2, 1]] * np.array([0.0004, 0.0004, -0.0004])
    unique, inverse = np.unique(np.round(vertices, 10), axis=0, return_inverse=True)
    return unique, inverse.reshape(-1, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", default=["3437.dat", "15314.dat", "65084.dat", "14094.dat"])
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in args.parts:
        fetch(name)
    manifest = CACHE.parent / "ldraw-sources.json"
    previous = {e["file"]: e["url"] for e in json.loads(manifest.read_text())} if manifest.exists() else {}
    sources = []
    for path in sorted(CACHE.rglob("*.dat")):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        sources.append({
            "file": str(path.relative_to(CACHE)),
            "url": ORIGIN.get(path.relative_to(CACHE).as_posix(),
                              previous.get(path.relative_to(CACHE).as_posix(), "")),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "attribution": [s for s in lines if s.startswith(("0 Author:", "0 !LICENSE", "0 !HISTORY"))],
        })
    (CACHE.parent / "ldraw-sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    for name in args.parts:
        vertices, faces = geometry(name)
        destination = CACHE.parent / (Path(name).stem + ".npz")
        np.savez_compressed(destination, vertices=vertices, faces=faces)
        print(name, "triangles", len(faces), "bounds_mm", (vertices.min(0)*1000).tolist(),
              (vertices.max(0)*1000).tolist(), flush=True)


if __name__ == "__main__":
    main()
