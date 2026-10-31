from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from pusula.alpha.combine import SignalSpec
from pusula.alpha.evaluate import daily_ic, forward_returns, summarize_ic
from pusula.alpha.lookahead import check_causal
from pusula.core.errors import RegistryError
from pusula.data.panel import Panel
from pusula.ml.walkforward import WalkForwardSpec, walk_forward_scores
from pusula.research.gates import GateSpec, evaluate
from pusula.research.registry import Registry

T0 = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


def test_registry_flow_and_rules(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "ledger.jsonl")
    h = reg.preregister(
        family="f",
        name="a",
        hypothesis="x",
        config={"p": 1},
        gates={},
        windows={},
        now=T0,
        code="c",
    )
    assert (
        reg.preregister(
            family="f",
            name="a",
            hypothesis="x",
            config={"p": 1},
            gates={},
            windows={},
            now=T0,
            code="c",
        )
        == h
    )  # idempotent
    with pytest.raises(RegistryError, match="different configuration"):
        reg.preregister(
            family="f", name="a", hypothesis="x", config={"p": 2}, gates={}, windows={}, now=T0
        )
    with pytest.raises(RegistryError, match="precedes"):
        reg.record_result(h, metrics={}, verdict={}, data_digest="d", now=T0 - dt.timedelta(days=1))
    reg.record_result(h, metrics={"s": 1.0}, verdict={"promoted": False}, data_digest="d", now=T0)
    with pytest.raises(RegistryError, match="rerun_reason"):
        reg.record_result(h, metrics={}, verdict={}, data_digest="d", now=T0)
    reg.record_result(
        h, metrics={}, verdict={}, data_digest="d", now=T0, rerun_reason="data refresh"
    )
    reg.preregister(
        family="f",
        name="b",
        hypothesis="y",
        config={"p": 3, "benchmark": True},
        gates={},
        windows={},
        now=T0,
        code="c",
    )
    assert reg.family_trials("f") == 1  # benchmarks are not trials
    assert Registry(tmp_path / "ledger.jsonl").head == reg.head  # reload verifies the chain


def test_registry_two_writers_keep_one_chain(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    a, b = Registry(path), Registry(path)  # e.g. a long study and a registration command
    h1 = a.preregister(
        family="f", name="x", hypothesis="", config={"p": 1}, gates={}, windows={}, now=T0, code="c"
    )
    b.preregister(
        family="g", name="y", hypothesis="", config={"p": 2}, gates={}, windows={}, now=T0, code="c"
    )
    a.record_result(h1, metrics={}, verdict={}, data_digest="d", now=T0)  # a's view was stale
    Registry(path).verify()
    assert len(Registry(path).entries) == 3


def test_registry_detects_tampering(tmp_path: Path) -> None:
    p = tmp_path / "ledger.jsonl"
    reg = Registry(p)
    reg.preregister(
        family="f",
        name="a",
        hypothesis="x",
        config={"p": 1},
        gates={},
        windows={},
        now=T0,
        code="c",
    )
    lines = p.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["body"]["hypothesis"] = "edited later"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="modified"):
        Registry(p)


def test_gate_missing_stress_fails() -> None:
    v = evaluate(
        GateSpec(),
        dsr=0.99,
        t_nw=3.0,
        alpha_t=3.0,
        sharpe=1.5,
        sharpe_lag2=float("nan"),
        t_nw_cost2=2.0,
        pbo=0.1,
        max_dd=0.1,
        positive_year_share=0.8,
    )
    assert not v.promoted
    assert v.failed == ["lag_stress"]
    ok = evaluate(
        GateSpec(),
        dsr=0.99,
        t_nw=3.0,
        alpha_t=3.0,
        sharpe=1.5,
        sharpe_lag2=1.0,
        t_nw_cost2=2.0,
        pbo=0.1,
        max_dd=0.1,
        positive_year_share=0.8,
    )
    assert ok.promoted


FEATS = (SignalSpec("planted"), SignalSpec("momentum"), SignalSpec("reversal"))


def test_ml_scores_are_causal(alpha_panel: Panel) -> None:
    spec = WalkForwardSpec(FEATS, model="ridge", refit_every=63, min_train=300)
    check_causal(
        lambda p: walk_forward_scores(spec, p).scores,
        alpha_panel,
        [520, 700],
        atol=1e-9,
        name="walk-forward ridge",
    )


def test_ml_learns_planted_signal(alpha_panel: Panel) -> None:
    spec = WalkForwardSpec(FEATS, model="ridge", refit_every=63, min_train=300)
    res = walk_forward_scores(spec, alpha_panel)
    assert res.refit_rows and res.coefs
    assert np.argmax(np.abs(res.coefs[-1])) == 0  # weight on the planted feature dominates
    ic = summarize_ic(
        daily_ic(res.scores, forward_returns(alpha_panel, 1, 1), alpha_panel["in_universe"])
    )
    assert ic.mean > 0.02 and ic.t_nw > 3


def test_gbm_runs_and_is_causal(alpha_panel: Panel) -> None:
    spec = WalkForwardSpec(FEATS, model="gbm", refit_every=126, min_train=400)
    check_causal(
        lambda p: walk_forward_scores(spec, p).scores, alpha_panel, [700], atol=1e-9, name="gbm"
    )


def test_programme_trial_count_spans_versions(tmp_path: Path) -> None:
    from pusula.cli import program_of

    assert program_of("us-equity-v2") == "us-equity-"
    assert program_of("us-options-v1") == "us-options-"
    assert program_of("adhoc") == "adhoc"
    reg = Registry(tmp_path / "l.jsonl")
    for fam in ("us-equity-v1", "us-equity-v2"):
        for name in ("a", "b"):
            reg.preregister(
                family=fam,
                name=name,
                hypothesis="",
                config={"f": fam, "n": name},
                gates={},
                windows={},
                now=T0,
                code="c",
            )
        reg.preregister(
            family=fam,
            name="bench",
            hypothesis="",
            config={"benchmark": True, "f": fam},
            gates={},
            windows={},
            now=T0,
            code="c",
        )
    assert reg.family_trials("us-equity-v2") == 2
    assert reg.program_trials("us-equity-") == 4


def test_reevaluate_uses_programme_dispersion(tmp_path: Path) -> None:
    from pusula.research.gates import GateSpec
    from pusula.research.study import reevaluate

    reg = Registry(tmp_path / "l.jsonl")
    gates = GateSpec().as_dict()

    def add(fam: str, name: str, sr: float) -> None:
        h = reg.preregister(
            family=fam,
            name=name,
            hypothesis="",
            config={"f": fam, "n": name},
            gates=gates,
            windows={},
            now=T0,
            code="c",
        )
        hold = {
            "sr_daily": sr,
            "n": 900,
            "skew": 0.0,
            "kurt": 3.0,
            "t_nw": 2.5,
            "sharpe": sr * 15.87,
            "max_drawdown": 0.1,
        }
        reg.record_result(
            h,
            metrics={
                "holdout": hold,
                "factor": {"alpha_t": 2.5},
                "pbo": 0.2,
                "lag2_holdout": {"sharpe": sr * 15},
                "cost2_holdout": {"t_nw": 2.0},
                "yearly": {"2024": {"net": 0.1}},
                "dsr": 0.99,
            },
            verdict={},
            data_digest="d",
            now=T0,
        )

    for i, sr in enumerate((0.060, 0.061, 0.062)):  # three similar trials: tiny own dispersion
        add("p-v2", f"a{i}", sr)
    for i, sr in enumerate((-0.07, 0.0, 0.03, -0.02, 0.05)):  # the programme's earlier trials
        add("p-v1", f"b{i}", sr)
    out = reevaluate(reg, "p-v2", "p-", T0, "test", code_files={})
    assert set(out) == {"a0", "a1", "a2"}
    assert all(r["dsr"] < 0.95 for r in out.values())  # wide programme spread -> higher hurdle
    latest = Registry(tmp_path / "l.jsonl").latest_results("p-v2")["a2"].body
    assert latest["rerun_reason"] == "test" and latest["metrics"]["dsr_inputs"]["n_trials"] == 8
