"""Telegram alerts and an external heartbeat. Both are opt-in through the environment.

``PUSULA_TELEGRAM_TOKEN`` + ``PUSULA_TELEGRAM_CHAT_ID`` turn alerts on. ``PUSULA_HEARTBEAT_URL``
is pinged when the system is healthy: only something outside this machine can notice that the
machine itself is off, so a dead-man service (healthchecks.io and the like) is pointed at it.

An alert is sent when it appears and again every ``REMIND_AFTER`` while it stays open, and its
resolution is sent once; a state file makes that hold across runs (NihaiAlgoritma's notifier
repeated itself every cycle until the channel was ignored). An alert is marked as sent only
after Telegram accepted it, so a failed send is retried by the next run.

A Telegram URL contains the token and httpx puts URLs in its exception messages, so transport
errors are re-raised without their text: no secret can reach a log.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from pusula.core.errors import PusulaError
from pusula.data.settings import atomic_write_text, environment

REMIND_AFTER = dt.timedelta(hours=24)
MAX_MESSAGE = 4000  # Telegram's limit is 4096 characters


class NotifyError(PusulaError):
    """A notification could not be delivered. The message never contains a secret."""


@dataclass(frozen=True, slots=True)
class Alert:
    key: str  # stable identity across runs
    severity: str  # critical | warning
    message: str


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    token: str
    chat_id: str

    def __repr__(self) -> str:  # never print secrets
        return "TelegramConfig(token=***, chat_id=***)"


def telegram_config() -> TelegramConfig | None:
    env = environment()
    token, chat = env.get("PUSULA_TELEGRAM_TOKEN"), env.get("PUSULA_TELEGRAM_CHAT_ID")
    return TelegramConfig(token, chat) if token and chat else None


def send_telegram(
    text: str, cfg: TelegramConfig, transport: httpx.BaseTransport | None = None
) -> None:
    try:
        with httpx.Client(transport=transport, timeout=20.0) as client:
            r = client.post(
                f"https://api.telegram.org/bot{cfg.token}/sendMessage",
                json={"chat_id": cfg.chat_id, "text": text[:MAX_MESSAGE]},
            )
    except httpx.HTTPError as exc:
        raise NotifyError(f"telegram: {type(exc).__name__}") from None
    if r.status_code != 200:
        raise NotifyError(f"telegram: HTTP {r.status_code}")


def ping_heartbeat(url: str, transport: httpx.BaseTransport | None = None) -> None:
    try:
        with httpx.Client(transport=transport, timeout=20.0) as client:
            r = client.get(url)
    except httpx.HTTPError as exc:
        raise NotifyError(f"heartbeat: {type(exc).__name__}") from None
    if r.status_code >= 400:
        raise NotifyError(f"heartbeat: HTTP {r.status_code}")


def _load(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    state: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return state


def compose(
    alerts: Sequence[Alert], state: dict[str, dict[str, Any]], now: dt.datetime
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """The message this run should send (None if nothing changed) and the state to keep if it
    is delivered."""
    fresh: list[Alert] = []
    reminders: list[Alert] = []
    out: dict[str, dict[str, Any]] = {}
    for a in alerts:
        seen = state.get(a.key)
        if seen is None:
            fresh.append(a)
            out[a.key] = {"severity": a.severity, "message": a.message, "sent": now.isoformat()}
        elif now - dt.datetime.fromisoformat(seen["sent"]) >= REMIND_AFTER:
            reminders.append(a)
            out[a.key] = {**seen, "message": a.message, "sent": now.isoformat()}
        else:
            out[a.key] = seen
    live = {a.key for a in alerts}
    resolved = [v["message"] for k, v in sorted(state.items()) if k not in live]
    if not (fresh or reminders or resolved):
        return None, out
    lines = ["pusula"]
    order = {"critical": 0, "warning": 1}
    for title, group in (("NEW", fresh), ("STILL OPEN", reminders)):
        if group:
            lines += ["", title]
            lines += [
                f"[{a.severity}] {a.message}"
                for a in sorted(group, key=lambda a: (order.get(a.severity, 2), a.key))
            ]
    if resolved:
        lines += ["", "RESOLVED", *resolved]
    return "\n".join(lines), out


def deliver_digest(
    text: str, session: str, state_path: Path, transport: httpx.BaseTransport | None = None
) -> str:
    """The night's summary, once per session (QuantBot sent one after every run). Silence from it
    means the night did not finish, which is why it is separate from the alerts."""
    cfg = telegram_config()
    if cfg is None:
        return "digest: telegram not configured"
    if state_path.exists():
        sent = json.loads(state_path.read_text(encoding="utf-8"))
        if sent.get("session") == session:
            return "digest: already sent"
    send_telegram(text, cfg, transport)
    atomic_write_text(state_path, json.dumps({"session": session}))
    return "digest: sent"


def deliver(
    alerts: Sequence[Alert],
    state_path: Path,
    now: dt.datetime,
    transport: httpx.BaseTransport | None = None,
) -> str:
    """Send what changed and ping the heartbeat when no alert is critical. Returns what happened,
    for the log. Raises :class:`NotifyError` after trying both, so one failing does not cost the
    other."""
    errors: list[str] = []
    notes: list[str] = []
    cfg = telegram_config()
    if cfg is None:
        notes.append("telegram not configured")
    else:
        state = _load(state_path)
        text, new_state = compose(alerts, state, now)
        if text is None:
            notes.append("nothing new to send")
        else:
            try:
                send_telegram(text, cfg, transport)
            except NotifyError as exc:
                errors.append(str(exc))
            else:
                atomic_write_text(state_path, json.dumps(new_state, indent=1))
                notes.append(f"telegram sent ({len(alerts)} open)")
    url = environment().get("PUSULA_HEARTBEAT_URL")
    if url and not any(a.severity == "critical" for a in alerts):
        try:
            ping_heartbeat(url, transport)
        except NotifyError as exc:
            errors.append(str(exc))
        else:
            notes.append("heartbeat pinged")
    if errors:
        raise NotifyError("; ".join(errors))
    return "; ".join(notes)
