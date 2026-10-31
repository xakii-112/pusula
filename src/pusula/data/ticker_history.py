"""Earlier tickers of index members, for sources that name a stock by the ticker it traded
under on the day (FINRA's daily short-sale files). A security id carries today's ticker
(``META@2016-01-01``); before a rename the company traded as FB, and today's ticker may then
have belonged to someone else (BNY was a municipal-bond fund while BNY Mellon traded as BK).

``EARLIER_TICKERS[sid]`` lists (earlier ticker, first session of the next ticker) in date
order. Each entry was found from the data, not from memory: the member's own FINRA volume did
not move with its bars (daily log-volume changes, correlation below 0.35 over a member year),
the earlier ticker's did, and the change date is the session after the earlier ticker's last
FINRA row; every entry is re-checked by ``tools/ticker_history_report.py``.
"""

from __future__ import annotations

EARLIER_TICKERS: dict[str, tuple[tuple[str, str], ...]] = {
    "BALL@2016-01-01": (("BLL", "2022-05-10"),),
    "BKR@2017-07-07": (("BHGE", "2019-10-18"),),
    # BNY was a municipal-bond fund's ticker until 2026-02-06
    "BNY@2016-01-01": (("BK", "2026-05-21"),),
    # COR was CoreSite Realty's ticker until 2021-12-27
    "COR@2016-01-01": (("ABC", "2023-08-30"),),
    "CPAY@2016-01-01": (("FLT", "2024-03-25"),),
    "CPRI@2016-01-01": (("KORS", "2019-01-02"),),
    # CTRA was Contura Energy's ticker until 2021-02-03; Cabot (COG) became Coterra
    "CTRA@2016-01-01": (("COG", "2021-10-04"),),
    # DOC was Physicians Realty Trust's ticker until Healthpeak took it over
    "DOC@2016-01-01": (("HCP", "2019-11-05"), ("PEAK", "2024-03-04")),
    # ECHO was Echo Global Logistics' ticker until 2021-11-22
    "ECHO@2016-01-01": (("SATS", "2026-06-24"),),
    "EG@2016-01-01": (("RE", "2023-07-10"),),
    "ELV@2016-01-01": (("ANTM", "2022-06-28"),),
    # FISV moved to FI and back; FI was Frank's International's ticker until 2021-10-01
    "FISV@2016-01-01": (("FISV", "2023-06-07"), ("FI", "2025-11-11")),
    # GEN was Genesis Healthcare's ticker until 2021-03-25
    "GEN@2016-01-01": (("SYMC", "2019-11-05"), ("NLOK", "2022-11-08")),
    "GL@2016-01-01": (("TMK", "2019-08-09"),),
    "J@2016-01-01": (("JEC", "2019-12-10"),),
    "LHX@2016-01-01": (("HRS", "2019-07-01"),),
    "LUMN@2016-01-01": (("CTL", "2020-09-18"),),
    # META was a metaverse ETF's ticker from 2021-06-30 to 2022-01-28
    "META@2016-01-01": (("FB", "2022-06-09"),),
    "MRSH@2016-01-01": (("MMC", "2026-01-14"),),
    "RTX@2016-01-01": (("UTX", "2020-04-03"),),
    "RVTY@2016-01-01": (("PKI", "2023-05-16"),),
    # priced only from 2025-08-07 (Paramount Skydance); earlier tickers have no bars to match
    "SKYD@2016-01-01": (("PSKY", "2026-10-06"),),
    "SW@2016-01-01": (("WRK", "2024-07-08"),),
    "TFC@2016-01-01": (("BBT", "2019-12-09"),),
    # IR passed to Gardner Denver (Ingersoll Rand) the day Trane became TT
    "TT@2016-01-01": (("IR", "2020-03-02"),),
    "VMRK@2016-01-01": (("EQR", "2026-08-18"),),
}
