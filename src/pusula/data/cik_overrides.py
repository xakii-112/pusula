"""Hand-verified CIKs for membership spells that the automatic resolution cannot settle.

Key: security id (``TICKER@start``). Value: (CIK, or ``None`` when the security has no EDGAR
record; reason). Each entry is a research decision, versioned with the code, and is still
checked by ``fundamentals.resolve_master`` (the entity must have filed 10-K/10-Q reports near
both ends of the spell and carry the ticker or the index name, current or former); an entry
that fails the check is kept but labelled ``override_unverified`` in the security master.

Why automatic resolution failed (2026-10-24): EDGAR's entity search indexes current names, so
a company renamed after leaving the index is not found under the name the index used.
"""

from __future__ import annotations

_RENAMED = "renamed after leaving the index; EDGAR search finds only the current name"

CIK_OVERRIDES: dict[str, tuple[int | None, str]] = {
    "ADS@2016-01-01": (1101215, "Alliance Data Systems, now Bread Financial; " + _RENAMED),
    "ADT@2016-01-01": (1546640, "The ADT Corporation (taken private 2016), not ADT Inc. (2018)"),
    "ARNC@2016-11-01": (
        4281,
        "Arconic Inc. (formerly Alcoa Inc., now Howmet), not the 2020 spin-off",
    ),
    "BBBY@2016-01-01": (886158, "Bed Bath & Beyond Inc.; " + _RENAMED),
    "BCR@2016-01-01": (9892, "C. R. Bard, Inc."),
    "BMS@2019-06-07": (11199, "Bemis Company, Inc."),
    "CA@2016-01-01": (356028, "CA, Inc. (CA Technologies)"),
    "CCE@2016-01-01": (1491675, "Coca-Cola Enterprises, Inc. (2010 entity), not The Coca-Cola Co"),
    "CDAY@2021-09-20": (1725057, "Ceridian HCM Holding, now Dayforce; " + _RENAMED),
    "CHK@2016-01-01": (895126, "Chesapeake Energy, now Expand Energy; " + _RENAMED),
    "DD@2016-01-01": (
        30554,
        "E. I. du Pont de Nemours and Company, before the 2017 DowDuPont merger",
    ),
    "DISCA@2016-01-01": (1437107, "Discovery, Inc., now Warner Bros. Discovery; " + _RENAMED),
    "DISCK@2016-01-01": (
        1437107,
        "Discovery, Inc. (class C), now Warner Bros. Discovery; " + _RENAMED,
    ),
    "ESV@2016-01-01": (314808, "Ensco plc, later Valaris; " + _RENAMED),
    "ETFC@2016-01-01": (1015780, "E*TRADE Financial Corporation"),
    "FRC@2019-01-02": (
        None,
        "First Republic Bank filed with the FDIC, not the SEC: no EDGAR record",
    ),
    "GAS@2016-01-01": (1004155, "AGL Resources, now Southern Company Gas; " + _RENAMED),
    "GGP@2016-01-01": (1496048, "GGP Inc., later Brookfield Property REIT; " + _RENAMED),
    "MNK@2016-01-01": (1567892, "Mallinckrodt plc"),
    "PBCT@2016-01-01": (1378946, "People's United Financial, Inc."),
    "Q@2017-08-29": (1478242, "QuintilesIMS, now IQVIA Holdings; " + _RENAMED),
    "SBNY@2021-12-20": (None, "Signature Bank filed with the FDIC, not the SEC: no EDGAR record"),
    "TSS@2016-01-01": (721683, "Total System Services, Inc."),
    "TWC@2016-01-01": (1377013, "Time Warner Cable Inc."),
    "TWX@2016-01-01": (1105705, "Time Warner Inc."),
    "TYC@2016-01-01": (
        833444,
        "Tyco International plc, the entity that became Johnson Controls International plc in 2016",
    ),
    "WYN@2016-01-01": (1361658, "Wyndham Worldwide, now Travel + Leisure Co.; " + _RENAMED),
    "YHOO@2016-01-01": (1011006, "Yahoo! Inc., later Altaba Inc.; " + _RENAMED),
}

# Overrides whose name mapping was confirmed by hand because the index uses a brand or an
# abbreviation the EDGAR name does not contain. The automatic check still requires the
# entity's 10-K/10-Q reports to cover the spell; only the name test is skipped.
# (EDGAR names, checked 2026-10-24: BARD C R INC /NJ/; CA, INC.; IQVIA HOLDINGS INC. formerly
# Quintiles IMS Holdings, Inc.; TOTAL SYSTEM SERVICES INC.)
NAME_CONFIRMED: frozenset[str] = frozenset(
    {"BCR@2016-01-01", "CA@2016-01-01", "Q@2017-08-29", "TSS@2016-01-01"}
)
