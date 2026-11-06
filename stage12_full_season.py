#!/usr/bin/env python3
"""Stage 12 — Full-season simulation: regular season + entire postseason.

One call = one complete simulated NHL season:
  1. random true-talent draw (Stage 6: N(0.5, 0.0415), locked constants)
  2. full 1,312-game regular season (Stage 8 vectorized sim)
  3. standings + seeding (Stage 9: 2013-14+ format)
  4. full 4-round postseason (Stage 11)

Per-team output table, ALL 32 teams (user spec 2026-10-20):
  team, talent_gfpct  (randomly assigned true-talent GF%)
  PTS, GF, GA         (regular season)
  made_playoffs, division_winner, conference_best, presidents_trophy  (0/1)
  round1_wins, round2_wins, round3_wins, cup_winner  (0/1)

Flags are 0/1 ints; made_playoffs etc. come from season_outcomes(), the
three playoff-round flags and cup from Stage 11's rounds_won (>=1, >=2, >=3)
and cup_winner. Note: "conference winner" = conference BEST regular-season
team (top seed) — there is no NHL conference trophy; the name is kept
explicit in the column.

Running this file executes the Stage 12 verification checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stage9_standings import simulate_season_full
from stage11_postseason import simulate_postseason


def simulate_full_season(rng: np.random.Generator,
                         ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One end-to-end season; returns (results, standings, round_logs).

    results: 32 rows x the columns in the module docstring (user spec).
    standings: full ranked Stage 9 table (GF/GA/records/ranks — kept for
        richer capture; results is the canonical per-team summary).
    round_logs: Stage 11 per-round series results.
    """
    # 1-3: talent draw + regular season + ranked standings + RS outcome flags.
    standings, game_log = simulate_season_full(rng)

    # Talent table for the postseason sim (Stage 11 needs GF/GA rates).
    team_data = standings.set_index("team")[["GF_per_game", "GA_per_game"]]

    # 4: postseason.
    po, round_logs = simulate_postseason(standings, game_log, team_data, rng)

    results = standings[[
        "team", "GF_percent", "PTS", "GF", "GA",
        "made_playoffs", "division_winner", "conference_best",
        "presidents_trophy",
    ]].rename(columns={"GF_percent": "talent_gfpct"}).copy()

    rw = po.set_index("team")["rounds_won"]
    results["round1_wins"] = results["team"].map(rw).ge(1).astype(int)
    results["round2_wins"] = results["team"].map(rw).ge(2).astype(int)
    results["round3_wins"] = results["team"].map(rw).ge(3).astype(int)
    results["cup_winner"] = po["cup_winner"].astype(int).to_numpy()

    # 0/1 dtype everywhere.
    for c in ["made_playoffs", "division_winner", "conference_best",
              "presidents_trophy", "round1_wins", "round2_wins",
              "round3_wins", "cup_winner"]:
        results[c] = results[c].astype(int)

    # Structural checks (mirrors the stage-level asserts end to end).
    assert len(results) == 32
    assert results["made_playoffs"].sum() == 16
    assert results["division_winner"].sum() == 4
    assert results["conference_best"].sum() == 2
    assert results["presidents_trophy"].sum() == 1
    assert results["cup_winner"].sum() == 1
    assert results["round1_wins"].sum() == 8
    assert results["round2_wins"].sum() == 4
    assert results["round3_wins"].sum() == 2
    # Cumulative consistency: a team winning round 2 also won round 1, etc.
    assert (results["round2_wins"] <= results["round1_wins"]).all()
    assert (results["round3_wins"] <= results["round2_wins"]).all()
    assert (results["cup_winner"] <= results["round3_wins"]).all()

    return results, standings, round_logs


# --- Stage 12 verification checks ---------------------------------------------


def check_single_season(seed: int = 20261025) -> None:
    rng = np.random.default_rng(seed)
    results, standings, round_logs = simulate_full_season(rng)
    cup = results.loc[results["cup_winner"] == 1, "team"].iloc[0]
    print(f"[single] cup: {cup}")
    print(results.sort_values("PTS", ascending=False).to_string(index=False))
    print("  [PASS] single end-to-end season, all structural asserts inside")


def check_batch(seed: int = 20261026, n: int = 1_000) -> None:
    """1k full seasons: internal consistency + rates near Stage 11's own
    numbers; best-team (top talent GF%) outcomes vs JLikens ballparks
    (playoffs ~97.8%, cup ~22% — ours uses 32-team talent draws and a
    different playoff format, so treat as an order-of-magnitude anchor only;
    per plan rule: flag deviations, don't tune)."""
    cups = rounds0 = 0
    best_rows = []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        results, standings, _ = simulate_full_season(rng)
        assert int(results["cup_winner"].sum()) == 1
        cups += int(results["cup_winner"].sum())
        # 0 playoff rounds = 16 non-playoff teams + 8 round-1 losers.
        rounds0 += int((results["round1_wins"] == 0).sum())
        best = results.loc[results["talent_gfpct"].idxmax()]
        best_rows.append((int(best["made_playoffs"]),
                          int(best["presidents_trophy"]),
                          int(best["round3_wins"]),
                          int(best["cup_winner"])))

    best = pd.DataFrame(best_rows, columns=["po", "pt", "cf", "cup"])
    print(f"[batch] {n} full seasons: cups {cups} (= {n} expected), "
          f"teams winning 0 rounds {rounds0} (expect 24 per sim = {24*n})")
    assert cups == n and rounds0 == 24 * n
    print(f"  best-team rates: playoffs {best['po'].mean():.4f} "
          f"(JLikens ballpark ~0.978), Presidents' {best['pt'].mean():.4f} "
          f"(~0.32), conf-final {best['cf'].mean():.4f}, "
          f"cup {best['cup'].mean():.4f} (JLikens ~0.22)")
    print("  [PASS] 1k end-to-end seasons internally consistent")


def main() -> None:
    print("Stage 12 verification")
    check_single_season()
    check_batch()
    print("All Stage 12 checks PASS")


if __name__ == "__main__":
    main()
