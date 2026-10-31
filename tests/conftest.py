"""Shared fixtures: small deterministic synthetic markets."""

from __future__ import annotations

import os

import pytest

from pusula.data.panel import Panel
from pusula.data.synthetic import SyntheticSpec, generate


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("PUSULA_NETWORK") == "1":
        return
    skip = pytest.mark.skip(reason="network test (set PUSULA_NETWORK=1)")
    for it in items:
        if "network" in it.keywords:
            it.add_marker(skip)


@pytest.fixture(scope="session")
def noise_panel() -> Panel:
    return generate(
        SyntheticSpec(n_symbols=80, start="2016-01-04", end="2019-12-31", ic=0.0, seed=1)
    )


@pytest.fixture(scope="session")
def alpha_panel() -> Panel:
    return generate(
        SyntheticSpec(n_symbols=120, start="2016-01-04", end="2019-12-31", ic=0.04, seed=2)
    )
