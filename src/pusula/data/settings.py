"""Filesystem locations and credentials.

Everything pusula writes lives under ``PUSULA_HOME``, default ``~/.pusula``. Not a synced folder
(OneDrive locks files mid-sync and breaks atomic renames) and not ``AppData``: on Windows a
process started from an MSIX-packaged app (a Store terminal, IDE or desktop agent) reads and
writes a private redirected copy of ``AppData``, while the Task Scheduler sees the real one, so
a store under ``AppData`` built from one is invisible to the other. Credentials
come from the environment or from a dotenv file named by ``PUSULA_ENV_FILE``; their values are
never logged, hashed into results, or written anywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def home() -> Path:
    env = os.environ.get("PUSULA_HOME")
    p = Path(env) if env else Path.home() / ".pusula"
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: Path, text: str, retries: int = 20) -> None:
    """Write via a temp file + rename, retrying briefly if another process holds the target
    (antivirus or a sync client); the target is never left half-written."""
    import time

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for i in range(retries):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if i == retries - 1:
                raise
            time.sleep(0.05 * (i + 1))


def data_dir() -> Path:
    p = home() / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader (comments, blank lines and surrounding quotes handled)."""
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


@dataclass(frozen=True, slots=True)
class AlpacaCredentials:
    key_id: str
    secret: str

    def __repr__(self) -> str:  # never print secrets
        return "AlpacaCredentials(key_id=***, secret=***)"


def environment() -> dict[str, str]:
    """The process environment over the dotenv file named by ``PUSULA_ENV_FILE``."""
    env = dict(os.environ)
    env_file = env.get("PUSULA_ENV_FILE")
    if env_file and Path(env_file).is_file():
        env = {**read_dotenv(Path(env_file)), **{k: v for k, v in env.items() if v}}
    return env


def alpaca_credentials() -> AlpacaCredentials | None:
    env = environment()
    key = env.get("ALPACA_API_KEY") or env.get("APCA_API_KEY_ID")
    sec = env.get("ALPACA_SECRET_KEY") or env.get("APCA_API_SECRET_KEY")
    if not key or not sec:
        return None
    return AlpacaCredentials(key, sec)
