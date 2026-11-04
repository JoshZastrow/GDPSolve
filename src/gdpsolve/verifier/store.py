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
    if not path.exists() or path.stat().st_size != len(data):
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)  # atomic: a failed write never leaves a partial object under its sha256 name
    return ObjectRef(sha256=sha, media_type=media_type, size=len(data))


def get(ref: ObjectRef, root: Path = ROOT) -> bytes:
    return (root / ref.sha256).read_bytes()
