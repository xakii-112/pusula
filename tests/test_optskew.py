"""Option-surface features: the vectorised reducer against a per-group reference, the
streaming carry across chunks, and point-in-time alignment to the panel."""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pusula.core.calendar import sessions
from pusula.data import optskew as O
from pusula.data.panel import Panel


def _closest(df: pd.DataFrame, col: str, target: float, lo: float, hi: float) -> float:
    x = df[(df[col] >= lo) & (df[col] <= hi)]
    if x.empty:
        return float("nan")
    return float(x["vol"].iloc[int(np.argmin(np.abs(x[col].to_numpy() - target)))])


def reference(chain: pd.DataFrame) -> pd.DataFrame:
    """The first, per-group implementation (13 ms a group: kept as the specification)."""
    c = chain[
        (chain["bid"] > 0)
        & (chain["ask"] >= chain["bid"])
        & (chain["vol"] > 0.01)
        & (chain["vol"] < 5.0)
    ]
    c = c.assign(dte=(pd.to_datetime(c["expiration"]) - pd.to_datetime(c["date"])).dt.days)
    out = []
    for (date, sym), g in c.groupby(["date", "act_symbol"], sort=False):
        exps = g.drop_duplicates("expiration")[["expiration", "dte"]]
        near = exps[(exps["dte"] >= 10) & (exps["dte"] <= 60)]
        smirk = spread = slope = float("nan")
        if not near.empty:
            e30 = near["expiration"].iloc[int(np.argmin(np.abs(near["dte"].to_numpy() - 30)))]
            g30 = g[g["expiration"] == e30]
            calls, puts = g30[g30["call_put"] == "Call"], g30[g30["call_put"] == "Put"]
            atm = _closest(calls, "delta", 0.50, 0.35, 0.65)
            smirk = _closest(puts, "delta", -0.20, -0.35, -0.10) - atm
            pair = calls[(calls["delta"] >= 0.25) & (calls["delta"] <= 0.75)].merge(
                puts, on="strike", suffixes=("_c", "_p")
            )
            if not pair.empty:
                spread = float((pair["vol_c"] - pair["vol_p"]).mean())
            far = exps[(exps["dte"] >= 60) & (exps["dte"] <= 180)]
            if not far.empty and np.isfinite(atm):
                e90 = far["expiration"].iloc[int(np.argmin(np.abs(far["dte"].to_numpy() - 90)))]
                g90 = g[(g["expiration"] == e90) & (g["call_put"] == "Call")]
                slope = _closest(g90, "delta", 0.50, 0.35, 0.65) - atm
        out.append((date, sym, smirk, spread, slope))
    return pd.DataFrame(out, columns=["date", "symbol", *O.FEATURES])


def synthetic_chain(n_days: int = 6, n_syms: int = 25, seed: int = 0) -> pd.DataFrame:
    """Chains with several expirations, strike grids, missing sides, zero bids and ties."""
    rng = np.random.default_rng(seed)
    rows = []
    days = pd.date_range("2024-01-05", periods=n_days, freq="7D")
    for d in days:
        for s in range(n_syms):
            sym = f"S{s:02d}"
            n_exp = rng.integers(1, 5)
            for dte in sorted(
                rng.choice([7, 14, 28, 32, 45, 63, 91, 95, 150, 200], n_exp, replace=False)
            ):
                exp = (d + pd.Timedelta(days=int(dte))).strftime("%Y-%m-%d")
                for k in range(rng.integers(3, 12)):
                    strike = 80.0 + 2.5 * k
                    base = 0.25 + 0.05 * rng.random()
                    dc = float(np.round(np.clip(1.0 - 0.1 * k + 0.05 * rng.normal(), 0.0, 1.0), 2))
                    for side, delta in (("Call", dc), ("Put", dc - 1.0)):
                        if rng.random() < 0.1:
                            continue  # one side missing
                        bid = 0.0 if rng.random() < 0.05 else 1.0
                        vol = base + (0.02 if side == "Put" else 0.0) + 0.01 * rng.normal()
                        rows.append(
                            (
                                d.strftime("%Y-%m-%d"),
                                sym,
                                exp,
                                strike,
                                side,
                                bid,
                                bid + 0.1,
                                vol,
                                delta,
                            )
                        )
    return pd.DataFrame(rows, columns=O.COLUMNS)


def test_vectorised_reducer_matches_the_reference() -> None:
    chain = synthetic_chain()
    got = O.reduce_chain(chain).reset_index(drop=True)
    want = reference(chain).reset_index(drop=True)
    assert len(got) == len(want) and (got["symbol"] == want["symbol"]).all()
    for f in O.FEATURES:
        np.testing.assert_allclose(
            got[f].to_numpy(), want[f].to_numpy(), atol=1e-12, equal_nan=True
        )
    assert got["smirk"].notna().sum() > 50 and got["term_slope"].notna().sum() > 20


def test_streaming_carries_a_group_across_chunks() -> None:
    chain = synthetic_chain(n_days=3, n_syms=10, seed=1)
    data = chain.to_csv(index=False).encode()
    whole = O.reduce_chain(chain).reset_index(drop=True)
    for rows in (37, 101, 1000):  # chunk boundaries fall inside groups
        got = O.reduce_stream(io.BytesIO(data), chunk_rows=rows, log=False).reset_index(drop=True)  # type: ignore[arg-type]
        assert len(got) == len(whole)
        for f in O.FEATURES:
            np.testing.assert_allclose(
                got[f].to_numpy(), whole[f].to_numpy(), atol=1e-12, equal_nan=True
            )


class _Unseekable(io.RawIOBase):
    """A write-only, unseekable sink: zipfile then writes data descriptors (streaming ZIP),
    as DoltHub's export does."""

    def __init__(self) -> None:
        self.data = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, b: object) -> int:  # type: ignore[override]
        mv = memoryview(b)  # type: ignore[arg-type]
        self.data += mv
        return mv.nbytes


def test_zip_member_reads_a_streaming_archive_without_its_directory() -> None:
    import hashlib
    import zipfile

    sink = _Unseekable()
    chain = synthetic_chain(n_days=2, n_syms=4).to_csv(index=False).encode()
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("dolt_docs.csv", b"doc_name,doc_text\nREADME,x\n" * 50)
        z.writestr("option_chain.csv", chain)
        # large and incompressible, so that stopping at the chain's end is observable
        z.writestr("volatility_history.csv", np.random.default_rng(0).bytes(50_000).hex().encode())
    blob = bytes(sink.data)
    sha = hashlib.sha256()
    chunks = (blob[i : i + 997] for i in range(0, len(blob), 997))
    assert b"".join(O.zip_member(chunks, "option_chain.csv", sha)) == chain
    consumed = sha.hexdigest()
    assert consumed != hashlib.sha256(blob).hexdigest()  # stopped before the last member
    from pusula.core.errors import DataError

    with pytest.raises(DataError, match=r"not found|ended before"):
        list(O.zip_member(iter([blob]), "missing.csv", hashlib.sha256()))


def test_stream_reader_hashes_what_it_reads() -> None:
    import hashlib

    data = b"date,x\n" + b"2024-01-01,1\n" * 1000
    r = O._StreamReader(iter([data[:100], data[100:2500], data[2500:]]))
    assert io.BufferedReader(r).read() == data
    assert r.sha.hexdigest() == hashlib.sha256(data).hexdigest() and r.bytes == len(data)


def test_skew_fields_point_in_time(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-06", "2024-01-20"]),  # Saturday snapshots
            "symbol": ["AAA", "AAA"],
            "smirk": [0.05, 0.07],
            "vol_spread": [0.0, 0.01],
            "term_slope": [np.nan, 0.02],
        }
    ).to_parquet(tmp_path / "option_skew.parquet", index=False)
    dates = sessions(dt.date(2024, 1, 2), dt.date(2024, 1, 31))
    p = Panel(dates, ("AAA@2016-01-01", "BBB@2016-01-01"), {"close": np.ones((dates.size, 2))})
    f = O.skew_fields(p, tmp_path, max_stale_days=7)["smirk"]
    day = {str(d): i for i, d in enumerate(dates)}
    assert np.isnan(f[day["2024-01-05"], 0])  # Friday: the snapshot is dated Saturday
    assert f[day["2024-01-08"], 0] == 0.05 and f[day["2024-01-12"], 0] == 0.05
    assert np.isnan(f[day["2024-01-16"], 0])  # 10 days old: stale
    assert f[day["2024-01-22"], 0] == 0.07 and np.all(np.isnan(f[:, 1]))


def test_surface_signals_are_causal_and_signed(noise_panel: Panel) -> None:
    from pusula.alpha.lookahead import check_causal, default_cutoffs
    from pusula.alpha.signals import SURFACE_SIGNALS, build

    rng = np.random.default_rng(0)
    shape = noise_panel["close"].shape
    fields = {
        f: np.where(rng.random(shape) < 0.8, rng.normal(0.03, 0.02, shape), np.nan)
        for f in O.FEATURES
    }
    panel = noise_panel.with_fields(**fields)
    for name in SURFACE_SIGNALS:
        check_causal(build(name).compute, panel, default_cutoffs(panel.n_dates), name=name)
    # a steep smirk is bearish, a high call-minus-put spread and an upward term slope bullish
    np.testing.assert_array_equal(build("iv_smirk").compute(panel), -panel["smirk"])
    np.testing.assert_array_equal(build("iv_spread").compute(panel), panel["vol_spread"])
    np.testing.assert_array_equal(build("iv_term").compute(panel), panel["term_slope"])


def test_surface_family_is_wired_and_kept_out_of_the_nightly_run() -> None:
    from pusula import cli
    from pusula.research.families import FORWARD_FAMILIES, us_optsurface_v1

    spec = us_optsurface_v1()
    assert cli.needs_option_surface(spec) and not cli.needs_fundamentals(spec)
    assert cli.program_of(spec.family) == "us-" and spec.family not in FORWARD_FAMILIES
    assert "iv_term" not in {s.name for t in spec.trials for s in t.signals}  # failed coverage


def test_reduce_chain_handles_an_empty_chunk() -> None:
    empty = synthetic_chain().iloc[0:0]
    assert O.reduce_chain(empty).empty


@pytest.mark.parametrize("seed", [2, 3])
def test_reducer_matches_reference_on_more_seeds(seed: int) -> None:
    chain = synthetic_chain(n_days=4, n_syms=15, seed=seed)
    got, want = O.reduce_chain(chain), reference(chain)
    for f in O.FEATURES:
        np.testing.assert_allclose(
            got[f].to_numpy(), want[f].to_numpy(), atol=1e-12, equal_nan=True
        )
