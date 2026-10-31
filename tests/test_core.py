from __future__ import annotations

import dataclasses
import datetime as dt
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pusula.core import calendar as cal
from pusula.core.hashing import array_digest, canonical_json, content_hash
from pusula.core.money import apply_rate, format_usd, round_to_cents, to_micros


@dataclasses.dataclass(frozen=True)
class _Cfg:
    a: int
    b: tuple[float, ...]
    c: str = "x"


def test_canonical_is_order_independent_and_parameter_sensitive() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    h1 = content_hash(_Cfg(1, (0.5, 1.0)))
    assert h1 == content_hash(_Cfg(1, (0.5, 1.0)))
    assert h1 != content_hash(_Cfg(1, (0.5, 1.0000001)))
    assert h1 != content_hash(_Cfg(2, (0.5, 1.0)))


def test_canonical_rejects_nan() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        content_hash({"x": math.nan})


def test_array_digest_nan_equal() -> None:
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, np.nan, 3.0])
    assert array_digest(a) == array_digest(b)
    assert array_digest(a) != array_digest(np.array([1.0, np.nan, 3.0000001]))


def test_money_exact() -> None:
    assert to_micros(0.1) == 100_000
    assert to_micros("123.4567") == 123_456_700
    assert to_micros(0.0000005) == 0  # half-even: 0.5 micro -> 0
    assert to_micros(0.0000015) == 2
    assert apply_rate(1_000_000_000, 0.0000278) == 27_800
    assert round_to_cents(15_000) == 20_000  # 1.5 cents -> 2 (half-even, odd quotient)
    assert round_to_cents(25_000) == 20_000  # 2.5 cents -> 2
    assert format_usd(-1_234_567_890) == "-$1,234.56"


@given(st.integers(min_value=-(10**12), max_value=10**12))
def test_round_to_cents_is_within_half_cent(m: int) -> None:
    r = round_to_cents(m)
    assert r % 10_000 == 0
    assert abs(r - m) <= 5_000


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2016, 252),
        (2017, 251),
        (2018, 251),
        (2019, 252),
        (2020, 253),
        (2021, 252),
        (2022, 251),
        (2023, 250),
        (2024, 252),
        (2025, 250),
    ],
)
def test_session_counts(year: int, expected: int) -> None:
    assert cal.sessions(dt.date(year, 1, 1), dt.date(year, 12, 31)).size == expected


def test_specific_holidays() -> None:
    assert cal.easter(2024) == dt.date(2024, 3, 31)
    assert not cal.is_session(dt.date(2024, 3, 29))  # Good Friday
    assert not cal.is_session(dt.date(2022, 6, 20))  # Juneteenth observed Monday
    assert cal.is_session(dt.date(2021, 6, 18))  # Juneteenth not yet an NYSE holiday
    assert cal.is_session(dt.date(2021, 12, 31))  # Jan 1 2022 is a Saturday: no Friday closure
    assert not cal.is_session(dt.date(2021, 12, 24))  # Christmas observed Friday
    assert not cal.is_session(dt.date(2027, 6, 18))  # Juneteenth Saturday -> Friday
    assert not cal.is_session(dt.date(2025, 1, 9))  # Carter day of mourning
    assert cal.next_session(dt.date(2024, 3, 28)) == dt.date(2024, 4, 1)
    assert cal.previous_session(dt.date(2024, 4, 1)) == dt.date(2024, 3, 28)
    assert dt.date(2024, 11, 29) in cal.early_closes(2024)
