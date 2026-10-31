"""Canonical serialisation and content hashing.

Every experiment, configuration and journal record is identified by the SHA-256 of its
canonical JSON form. Two objects that mean the same thing hash the same; any change to a
parameter changes the hash. This is what binds a result to the exact configuration that
produced it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

import numpy as np


def canonical(obj: Any) -> Any:
    """Convert ``obj`` into plain JSON types, deterministically.

    Dataclasses become dicts, tuples/sets become lists (sets sorted), numpy scalars become
    Python scalars, dates become ISO strings. NaN and infinities are rejected: they have no
    canonical JSON form and usually indicate a bug upstream.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: canonical(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return canonical(obj.value)
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"mapping keys must be str for canonical form, got {type(k)}")
            out[k] = canonical(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(canonical(v) for v in obj)
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not math.isfinite(f):
            raise ValueError(f"non-finite float {f!r} has no canonical form")
        return f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return canonical(obj.tolist())
    if isinstance(obj, PurePath):
        return obj.as_posix()
    raise TypeError(f"no canonical form for {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    return json.dumps(canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def content_hash(obj: Any) -> str:
    """SHA-256 of the canonical JSON form of ``obj``."""
    return sha256_hex(canonical_json(obj))


def array_digest(*arrays: np.ndarray) -> str:
    """Order-sensitive digest of numeric arrays (shape, dtype and bytes).

    NaNs are normalised to one bit pattern so that two NaN-equal arrays digest equally.
    """
    h = hashlib.sha256()
    for a in arrays:
        arr = np.ascontiguousarray(a)
        if arr.dtype.kind == "f":
            arr = np.where(np.isnan(arr), np.nan, arr)
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    return h.hexdigest()
