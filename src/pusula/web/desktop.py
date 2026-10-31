"""The dashboard as a desktop program: its own window, no browser, no port to remember.

The same read-only application as ``pusula web``, served to the window alone: on a free loopback
port chosen at start, for as long as the window is open. WebView2 ships with Windows 11. Started
from the shortcut (``pythonw``) there is no console, so what fails goes to a log file and a message
box instead of nowhere.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn

from pusula.core.errors import PusulaError
from pusula.data.settings import home
from pusula.web.app import create_app

ICON = Path(__file__).with_name("pusula.ico")
TITLE = "Pusula"
BACKGROUND = "#0b0e13"
START_TIMEOUT = 20.0
STOP_TIMEOUT = 5.0


class DesktopError(PusulaError):
    """The window could not be opened. The message says what to do."""


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class AppServer:
    """uvicorn on a background thread. ``log_config=None``: its default formatter asks stdout
    whether it is a terminal, and under ``pythonw`` there is no stdout."""

    def __init__(self, ledger: Path, forward_root: Path | None) -> None:
        self.port = free_port()
        config = uvicorn.Config(
            create_app(ledger, forward_root),
            host="127.0.0.1",
            port=self.port,
            log_config=None,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="pusula-web")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + START_TIMEOUT
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise DesktopError("the local server did not start")
            time.sleep(0.02)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(STOP_TIMEOUT)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()


def _set_process_identity() -> None:
    """Group the window under its own taskbar entry rather than under Python's."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("pusula.desktop")


def _set_window_icon(window: Any) -> None:
    """pywebview gives the window the interpreter's icon on Windows; this replaces it."""
    if sys.platform == "win32" and ICON.exists():
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.LoadImageW.argtypes = (
            wintypes.HINSTANCE,
            wintypes.LPCWSTR,
            wintypes.UINT,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        )
        user32.SendMessageW.argtypes = (
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.HANDLE,
        )
        image_icon, load_from_file, wm_seticon = 1, 0x10, 0x80
        hwnd = int(window.native.Handle.ToInt64())
        for which, size in ((0, 16), (1, 32)):  # small, big
            icon = user32.LoadImageW(None, str(ICON), image_icon, size, size, load_from_file)
            if icon:
                user32.SendMessageW(hwnd, wm_seticon, which, icon)


def open_window(ledger: Path, forward_root: Path | None = None, webview: Any = None) -> None:
    """Open the window and return when it is closed, with the server stopped."""
    if webview is None:
        try:
            import webview  # type: ignore[import-not-found,no-redef,unused-ignore]
        except ImportError as exc:
            raise DesktopError(
                "pywebview is not installed: run `uv sync`. On Linux or a Mac use `pusula web`."
            ) from exc
    _set_process_identity()
    server = AppServer(ledger, forward_root)
    server.start()
    try:
        window = webview.create_window(
            TITLE,
            server.url,
            # maximised: a fixed size scaled by the display can run under the taskbar
            maximized=True,
            width=1280,
            height=800,
            min_size=(1000, 640),
            background_color=BACKGROUND,
            text_select=True,
        )
        window.events.shown += lambda: _set_window_icon(window)
        try:
            webview.start(gui="edgechromium" if sys.platform == "win32" else None)
        except webview.errors.WebViewException as exc:
            raise DesktopError(
                f"the window could not be opened ({exc}). It needs Microsoft Edge WebView2 "
                "(part of Windows 11); `pusula web` works without it."
            ) from exc
    finally:
        server.stop()


def report_failure(message: str) -> None:
    """Where a failure goes when no console is attached: a log file and a message box."""
    logs = home() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "app.err.log").open("a", encoding="utf-8") as f:
        f.write(message + "\n")
    if sys.stderr is not None:
        print(message, file=sys.stderr)
    elif sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, TITLE, 0x10)
