#!/usr/bin/env python3
"""NHL league structure — conferences and divisions (current format).

Team-to-division mapping provided by the user (2026-10-19), matching the
2024-25 realignment. Used by Stage 9 (standings/seeding) and later stages
(playoff simulation). Codes follow the NHL API spelling (VGK, not VEG).
"""

from __future__ import annotations

EASTERN = ["BOS", "TBL", "OTT", "FLA", "BUF", "MTL", "DET", "TOR",
           "CAR", "NYR", "WSH", "NYI", "NJD", "PIT", "CBJ", "PHI"]
WESTERN = ["MIN", "UTA", "COL", "WPG", "DAL", "NSH", "STL", "CHI",
           "VGK", "EDM", "SEA", "SJS", "ANA", "VAN", "CGY", "LAK"]

ATLANTIC = ["BOS", "TBL", "OTT", "FLA", "BUF", "MTL", "DET", "TOR"]
METROPOLITAN = ["CAR", "NYR", "WSH", "NYI", "NJD", "PIT", "CBJ", "PHI"]
CENTRAL = ["MIN", "UTA", "COL", "WPG", "DAL", "NSH", "STL", "CHI"]
PACIFIC = ["VGK", "EDM", "SEA", "SJS", "ANA", "VAN", "CGY", "LAK"]

DIVISIONS = {
    "Atlantic": ATLANTIC,
    "Metropolitan": METROPOLITAN,
    "Central": CENTRAL,
    "Pacific": PACIFIC,
}
DIVISION_CONFERENCE = {
    "Atlantic": "Eastern",
    "Metropolitan": "Eastern",
    "Central": "Western",
    "Pacific": "Western",
}

TEAM_TO_DIVISION = {t: d for d, ts in DIVISIONS.items() for t in ts}
TEAM_TO_CONFERENCE = {t: DIVISION_CONFERENCE[d] for t, d in TEAM_TO_DIVISION.items()}


def _self_check() -> None:
    all_teams = sorted(TEAM_TO_DIVISION)
    assert len(all_teams) == 32, f"expected 32 teams, found {len(all_teams)}"
    assert all_teams == sorted(set(EASTERN) | set(WESTERN)), \
        "conference lists and division mapping disagree"
    assert sorted(EASTERN) == sorted(ATLANTIC + METROPOLITAN), "East != Atlantic+Metro"
    assert sorted(WESTERN) == sorted(CENTRAL + PACIFIC), "West != Central+Pacific"
    for d, ts in DIVISIONS.items():
        assert len(ts) == 8, f"division {d} has {len(ts)} teams, expected 8"


_self_check()
