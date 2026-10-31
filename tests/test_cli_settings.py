from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from pusula import cli
from pusula.data import settings
from pusula.research.registry import Registry
from pusula.risk.limits import RiskLimits, gate_targets


def test_credentials_from_dotenv_never_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nALPACA_API_KEY='kid-123'\nALPACA_SECRET_KEY=\"sec-456\"\n\nOTHER=1\n",
        encoding="utf-8",
    )
    for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PUSULA_ENV_FILE", str(env))
    creds = settings.alpaca_credentials()
    assert creds is not None and creds.key_id == "kid-123" and creds.secret == "sec-456"
    assert "kid-123" not in repr(creds) and "sec-456" not in repr(creds)
    monkeypatch.setenv("PUSULA_ENV_FILE", str(tmp_path / "missing.env"))
    assert settings.alpaca_credentials() is None


def test_home_and_atomic_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUSULA_HOME", str(tmp_path / "h"))
    assert settings.home() == tmp_path / "h" and settings.data_dir().is_dir()
    target = tmp_path / "x.json"
    settings.atomic_write_text(target, "a")
    settings.atomic_write_text(target, "b")
    assert target.read_text() == "b" and not (tmp_path / "x.json.tmp").exists()
    # default: ~/.pusula, never under AppData (redirected for packaged apps, see settings)
    monkeypatch.delenv("PUSULA_HOME")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user"))
    assert settings.home() == tmp_path / "user" / ".pusula"


def test_program_and_paths() -> None:
    assert cli.program_of("us-equity-v3") == "us-equity-"
    assert cli.program_of("us-optsignal-v1") == "us-"  # registered override
    assert cli.results_path("fam").name == "RESULTS_fam.md"


def test_cli_report_and_reevaluate_use_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "ledger" / "experiments.jsonl"
    reg = Registry(ledger)
    h = reg.preregister(
        family="x-v1",
        name="t",
        hypothesis="",
        config={"a": 1},
        gates={},
        windows={"dev": ["a", "b"], "holdout": ["c", "d"]},
        now=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        code="c",
    )
    reg.record_result(
        h,
        metrics={
            "holdout": {"sr_daily": 0.05, "n": 500, "skew": 0.0, "kurt": 3.0},
            "dev": {},
            "factor": {},
            "summary": {},
            "yearly": {},
        },
        verdict={},
        data_digest="d",
        now=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
    )
    monkeypatch.setattr(cli, "LEDGER", ledger)
    monkeypatch.setattr(cli, "REPO", tmp_path)
    (tmp_path / "docs").mkdir()
    assert cli.main(["study", "report", "--family", "x-v1"]) == 0
    assert "# Results: `x-v1`" in (tmp_path / "docs" / "RESULTS_x-v1.md").read_text(
        encoding="utf-8"
    )
    assert cli.main(["study", "reevaluate", "--family", "x-v1"]) == 2  # a reason is mandatory
    assert cli.main(["study", "reevaluate", "--family", "x-v1", "--rerun-reason", "r"]) == 0
    assert "DSR" in capsys.readouterr().out
    assert Registry(ledger).latest_results("x-v1")["t"].body["rerun_reason"] == "r"


def test_cli_forward_rehearsal_needs_a_root(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["forward", "--rehearse-from", "2026-08-03", "--no-update"]) == 2
    assert "never writes the real books" in capsys.readouterr().err


def test_every_family_in_the_code_verifies_against_the_repository_ledger() -> None:
    """A config field added with a default must not change the hash of trials registered
    before it existed (2026-10-24: three new RiskModelSpec fields entered every registered
    backtest config; v1-v3, the option-implied and the filings family would have refused to run
    and the nightly forward run would have failed)."""
    from pusula.research import study as S
    from pusula.research.families import FAMILIES

    reg = Registry(cli.LEDGER)
    for family, make in FAMILIES.items():
        spec = make()
        assert set(S.verify_registration(spec, reg)) == {t.name for t in spec.trials}, family


@pytest.mark.parametrize(
    ("down", "reads", "message"),
    [
        ("sec", "needs_fundamentals", "fundamentals update failed"),
        ("finra", "needs_short_volume", "short-volume update failed"),
        ("dolthub", "needs_implied_vol", "implied-vol update failed"),
    ],
)
def test_a_source_outage_skips_only_the_families_that_read_it(
    down: str,
    reads: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The nightly run must not lose the other books because SEC, FINRA or DoltHub is down
    (2026-10-27: a DoltHub timeout stopped every book); the families that read the source sit
    the night out and the exit code reports it."""
    import types

    import pusula.data.fundamentals as F
    import pusula.data.iv as IV
    import pusula.data.pipeline as P
    import pusula.data.settings as St
    import pusula.data.shortvol as SV
    import pusula.live.forward as LF
    from pusula.core.errors import DataError

    (tmp_path / "fundamentals").mkdir()
    (tmp_path / "fundamentals" / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "shortvol").mkdir()
    (tmp_path / "shortvol" / "short_volume.parquet").write_bytes(b"")
    (tmp_path / "iv").mkdir()
    (tmp_path / "iv" / "iv_history.parquet").write_bytes(b"")
    monkeypatch.setattr(St, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(P, "update_store", lambda: None)
    monkeypatch.setattr(P, "last_completed_session", lambda: dt.date(2026, 10, 23))
    # a stand-in panel that has every field: the books are mocked
    monkeypatch.setattr(
        P,
        "research_panel",
        lambda: types.SimpleNamespace(panel={"ann_day0", "short_volume", "iv"}),
    )
    monkeypatch.setattr(P, "universe_data", lambda u: types.SimpleNamespace(panel=set()))
    for f in ("with_implied_vol", "with_fundamentals", "with_option_surface", "with_short_volume"):
        monkeypatch.setattr(P, f, lambda rd: rd)

    def fail(*a: object, **k: object) -> None:
        raise DataError(f"{down} GET ...: HTTP 503")

    def ok(*a: object, **k: object) -> None:
        return None

    monkeypatch.setattr(F, "update_fundamentals", fail if down == "sec" else ok)
    monkeypatch.setattr(SV, "update_shortvol", fail if down == "finra" else ok)
    monkeypatch.setattr(IV, "update_iv", fail if down == "dolthub" else ok)
    stepped: list[str] = []
    monkeypatch.setattr(
        LF, "step_books", lambda spec, when, **k: stepped.append(spec.family) or {"books": {}}
    )
    assert cli.main(["forward", "--family", "all"]) == 1
    from pusula.research.families import FAMILIES, FORWARD_FAMILIES

    skipped = {f for f in FORWARD_FAMILIES if getattr(cli, reads)(FAMILIES[f]())}
    assert skipped and set(stepped) == set(FORWARD_FAMILIES) - skipped
    assert message in capsys.readouterr().err


def test_gate_sector_vol_and_turnover_limits() -> None:
    syms = ("A", "B", "C", "D")
    cur = np.zeros(4)
    tgt = np.array([0.05, 0.05, -0.05, -0.05])  # Tech net 0.10; after the sector fix: turnover 0.13
    lim = RiskLimits(max_weight=0.05, max_sector_net=0.03, max_turnover=0.08, max_vol_annual=0.05)
    res = gate_targets(
        tgt,
        cur,
        syms,
        lim,
        price=np.full(4, 50.0),
        adv_dollars=np.full(4, 1e9),
        equity=1e6,
        sector_of={"A": "Tech", "B": "Tech", "C": "Energy"},
        vol_fn=lambda w: float(np.abs(w).sum()) * 0.8,
    )
    w = res.weights
    joined = " | ".join(res.breaches)
    assert "sector Tech" in joined and "turnover" in joined and "ex-ante vol" in joined
    assert np.abs(w).sum() * 0.8 <= 0.05 + 1e-9
    assert np.abs(w - cur).sum() <= 0.08 + 1e-9


def test_sector_repair_never_breaks_the_name_limit() -> None:
    """Regression: the sector repair shifted every name in the sector equally, which pushed a
    short already at the 5% limit to -6.67% (after the per-name clip had run)."""
    syms = ("A", "B", "C")
    tgt = np.array([0.05, 0.05, -0.05])
    lim = RiskLimits(max_weight=0.05, max_sector_net=0.0)
    res = gate_targets(
        tgt,
        np.zeros(3),
        syms,
        lim,
        price=np.full(3, 50.0),
        adv_dollars=np.full(3, 1e9),
        equity=1e6,
        sector_of={"A": "Tech", "B": "Tech", "C": "Tech"},
    )
    assert np.abs(res.weights).max() <= 0.05 + 1e-12
    assert abs(res.weights.sum()) <= 1e-12
    np.testing.assert_allclose(res.weights, [0.025, 0.025, -0.05])


def test_iv_families_detected_and_reports_dir_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pusula.research.families import FAMILIES

    assert cli.needs_implied_vol(FAMILIES["us-optsignal-v1"]())
    assert not cli.needs_implied_vol(FAMILIES["us-equity-v2"]())  # ML features are price-only
    monkeypatch.setattr(cli, "REPO", tmp_path)  # a container image has no docs/ folder
    assert cli.results_path("f").parent.is_dir()
