"""Content-addressed object store: var/objects/<sha256>."""

import hashlib
import os
import threading
from pathlib import Path

from gdpsolve.schemas import ObjectRef

ROOT = Path("var/objects")


def put(data: bytes, media_type: str, root: Path = ROOT) -> ObjectRef:
    sha = hashlib.sha256(data).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    path = root / sha
    if not path.exists():
        path.write_bytes(data)
    return ObjectRef(sha256=sha, media_type=media_type, size=len(data))


def get(ref: ObjectRef, root: Path = ROOT) -> bytes:
    return (root / ref.sha256).read_bytes()
