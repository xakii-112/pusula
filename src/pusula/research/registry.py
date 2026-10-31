"""Pre-registration ledger: an append-only, hash-chained record of every experiment.

Rules enforced in code (they were prose in NihaiAlgoritma's CLAUDE.md, and got forgotten):

1. A result can only be recorded against a pre-registration made **before** it.
2. A configuration's identity is the SHA-256 of its canonical form; editing a parameter after
   seeing results creates a new trial, it does not overwrite the old one.
3. Every pre-registered configuration in a family counts as a trial in the deflated Sharpe
   ratio, whether or not its result was ever looked at, so trials cannot be hidden.
4. Re-running a study that already has results requires an explicit ``rerun_reason``,
   which is itself recorded.
5. Each record carries the hash of the previous one; ``verify`` detects any edit.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pusula.core.errors import RegistryError
from pusula.core.hashing import canonical, content_hash

GENESIS = "0" * 64


def code_manifest(src_root: Path | None = None) -> dict[str, str]:
    """SHA-256 per ``pusula`` source file (line endings normalised), keyed by relative path.

    Stored with every pre-registration and result so that the exact files that changed between
    the two can be listed, not just "the digest differs".
    """
    root = src_root or Path(__file__).resolve().parents[1]
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(_normalised(p)).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


def _normalised(p: Path) -> bytes:
    return p.read_bytes().replace(b"\r\n", b"\n")


def code_digest(src_root: Path | None = None) -> str:
    """SHA-256 over all ``pusula`` source files (path + bytes), sorted."""
    root = src_root or Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(_normalised(p))
    return h.hexdigest()


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    keys = set(before) | set(after)
    return sorted(k for k in keys if before.get(k) != after.get(k))


@dataclass(frozen=True, slots=True)
class Entry:
    kind: str
    body: dict[str, Any]
    prev: str
    hash: str


class Registry:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[Entry] = []
        self._reload()

    def _reload(self) -> None:
        """Re-read the ledger from disk and verify it (another process may have appended)."""
        entries = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    entries.append(Entry(rec["kind"], rec["body"], rec["prev"], rec["hash"]))
        mine = self.entries
        if entries[: len(mine)] != mine:
            raise RegistryError("ledger on disk diverged from this process's view")
        self.entries = entries
        self.verify()

    @contextmanager
    def _lock(self, timeout: float = 30.0) -> Iterator[None]:
        """Exclusive cross-process lock (lock file created with O_EXCL)."""
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise RegistryError(
                        f"ledger lock {lock} held for more than {timeout:.0f}s"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(fd)
            lock.unlink(missing_ok=True)

    # ------------------------------------------------------------------ integrity
    @staticmethod
    def _hash(kind: str, body: Mapping[str, Any], prev: str) -> str:
        return content_hash({"kind": kind, "body": body, "prev": prev})

    def verify(self) -> None:
        prev = GENESIS
        for i, e in enumerate(self.entries):
            if e.prev != prev:
                raise RegistryError(f"chain broken at entry {i}: prev mismatch")
            if self._hash(e.kind, e.body, e.prev) != e.hash:
                raise RegistryError(f"entry {i} was modified after it was written")
            prev = e.hash

    @property
    def head(self) -> str:
        return self.entries[-1].hash if self.entries else GENESIS

    def _append(self, kind: str, body: dict[str, Any]) -> Entry:
        body = canonical(body)
        with self._lock():
            # Chain to the head that is on disk *now*: a long-running study must not fork the
            # chain because another command appended while it was computing.
            self._reload()
            prev = self.head
            e = Entry(kind, body, prev, self._hash(kind, body, prev))
            line = json.dumps(
                {"kind": kind, "body": body, "prev": prev, "hash": e.hash},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
            self.entries.append(e)
        return e

    # ------------------------------------------------------------------ API
    def preregister(
        self,
        *,
        family: str,
        name: str,
        hypothesis: str,
        config: Mapping[str, Any],
        gates: Mapping[str, Any],
        windows: Mapping[str, Any],
        now: dt.datetime,
        code: str | None = None,
    ) -> str:
        self._reload()
        cfg_hash = content_hash(config)
        for e in self.entries:
            if e.kind == "prereg" and e.body["family"] == family and e.body["name"] == name:
                if e.body["config_hash"] != cfg_hash:
                    raise RegistryError(
                        f"{family}/{name} is already registered with a different configuration; "
                        "register a new name (changing a trial after the fact is not allowed)"
                    )
                return e.hash
        e = self._append(
            "prereg",
            {
                "family": family,
                "name": name,
                "hypothesis": hypothesis,
                "config": dict(config),
                "config_hash": cfg_hash,
                "gates": dict(gates),
                "windows": dict(windows),
                "created_at": now.isoformat(),
                "code_digest": code or code_digest(),
                "code_files": code_manifest() if code is None else {},
            },
        )
        return e.hash

    def prereg(self, prereg_hash: str) -> Entry:
        for e in self.entries:
            if e.kind == "prereg" and e.hash == prereg_hash:
                return e
        raise RegistryError(f"no pre-registration {prereg_hash[:12]}")

    def record_result(
        self,
        prereg_hash: str,
        *,
        metrics: Mapping[str, Any],
        verdict: Mapping[str, Any],
        data_digest: str,
        now: dt.datetime,
        rerun_reason: str | None = None,
        code_files: Mapping[str, str] | None = None,
    ) -> str:
        """``code_files`` should be the manifest taken when the run *started*; the default
        (taken now) is only right if the source did not change while the run was in progress."""
        self._reload()
        pre = self.prereg(prereg_hash)
        if dt.datetime.fromisoformat(pre.body["created_at"]) > now:
            raise RegistryError("result timestamp precedes its pre-registration")
        files = dict(code_files) if code_files is not None else code_manifest()
        prior = [e for e in self.entries if e.kind == "result" and e.body["prereg"] == prereg_hash]
        if prior and not rerun_reason:
            raise RegistryError(
                f"{pre.body['name']} already has a result; pass rerun_reason to re-run"
            )
        e = self._append(
            "result",
            {
                "prereg": prereg_hash,
                "family": pre.body["family"],
                "name": pre.body["name"],
                "metrics": dict(metrics),
                "verdict": dict(verdict),
                "data_digest": data_digest,
                "created_at": now.isoformat(),
                "rerun_reason": rerun_reason,
                "code_digest": content_hash(files),
                "code_files": files,
            },
        )
        return e.hash

    def family_trials(self, family: str) -> int:
        return sum(
            1
            for e in self.entries
            if e.kind == "prereg"
            and e.body["family"] == family
            and not e.body["config"].get("benchmark", False)
        )

    def program_trials(self, prefix: str) -> int:
        """Non-benchmark trials across every family whose name starts with ``prefix``: the
        deflated Sharpe of a re-designed family must also pay for its predecessors' trials."""
        return sum(
            1
            for e in self.entries
            if e.kind == "prereg"
            and str(e.body["family"]).startswith(prefix)
            and not e.body["config"].get("benchmark", False)
        )

    def preregs(self, family: str) -> list[Entry]:
        return [e for e in self.entries if e.kind == "prereg" and e.body["family"] == family]

    def latest_results(self, family: str) -> dict[str, Entry]:
        out: dict[str, Entry] = {}
        for e in self.entries:
            if e.kind == "result" and e.body["family"] == family:
                out[e.body["name"]] = e
        return out
