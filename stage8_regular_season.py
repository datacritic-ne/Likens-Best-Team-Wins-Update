#!/usr/bin/env python3
"""Stage 8 — Regular-season simulation: full 82-game schedule + standings.

Extends the Stage 7 half-season machinery to the complete 2025-26 regular
season (1,312 games, Stage 5 schedule). Per simulated season:

    1. draw team talents from N(0.5, 0.0415)      (Stage 6 assign_team_talent)
    2. simulate all 1,312 games, vectorized        (Stage 6 simulate_games)
       home_lam = 2 * gf_h * ga_a / GPG + HIA
       away_lam = 2 * gf_a * ga_h / GPG            (multiplicative form,
                                                    revised 2026-10-18)
       ties -> 1 pt each + coin-flip extra point; loser-of-flip = OTL
    3. accumulate standings: GP, W, L, OTL, PTS, GF, GA per team

Clean goals only: the simulator produces no empty-net or shootout goals
(shootout = the coin flip for the extra point), so GF/GA here are the same
clean-goal quantity the talent draw is parameterized on. This is the design
decision confirmed by the user for Stage 8 (2026-10-19).

simulate_season() takes an explicit numpy Generator so later stages (12/13)
can drive many seasons with per-season seeds, exactly reproducibly.

Usage:
    from stage8_regular_season import simulate_season
    rng = np.random.default_rng(seed)
    standings, game_log = simulate_season(rng)

Running this file executes the Stage 8 verification checks and prints a demo
standings table for one fixed-seed season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from assign_team_talent import (
    HOME_ICE_ADVANTAGE,
    SCHEDULE_CSV,
    assign_team_talent,
)
from simulate_games import simulate_games

HIA = HOME_ICE_ADVANTAGE

_TEAMS_32 = 32
_GAMES_1312 = 1_312


def simulate_season(
    rng: np.random.Generator,
    schedule: pd.DataFrame | None = None,
    hia: float = HIA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate one complete 2025-26 regular season.

    Arguments:
        rng: numpy Generator; every draw (talents, goals, coin flips) comes
             from it, in a documented order: talents first, then the
             schedule-ordered game batch, then point classification flips.
        schedule: optional Stage 5 schedule DataFrame (defaults to the
             full 1,312-game 2025-26 schedule); used as-is.
        hia: home-ice advantage in goals/game (default: locked 0.2563).

    Returns:
        (standings, game_log)
        standings: DataFrame indexed by team code, sorted by PTS descending:
            team, GF_percent, GF_per_game, GA_per_game, GP, W, L, OTL,
            PTS, GF, GA, DIFF (GF-GA), home_gp, away_gp
        game_log: the raw per-game DataFrame from simulate_games().
    """
    if schedule is None:
        schedule = pd.read_csv(SCHEDULE_CSV)

    team_data = assign_team_talent(rng)          # draw order step 1
    game_log = simulate_games(schedule, team_data, rng, hia=hia)  # steps 2-3

    standings = _standings_from_log(game_log)
    standings = standings.join(
        team_data[["GF_percent", "GF_per_game", "GA_per_game"]], on="team"
    )
    standings = standings.sort_values(
        ["PTS", "DIFF", "team"], ascending=[False, False, True], ignore_index=True
    )
    cols = [
        "team", "GF_percent", "GF_per_game", "GA_per_game",
        "GP", "W", "L", "OTL", "PTS", "GF", "GA", "DIFF", "home_gp", "away_gp",
    ]
    return standings[cols], game_log


def _standings_from_log(game_log: pd.DataFrame) -> pd.DataFrame:
    """Standings table from a game log (pandas groupby path).

    Independent of running_team_totals(): recomputes per-team totals from
    scratch via groupby, so agreement between the two is a real check.
    """
    n = len(game_log)
    home = game_log["home"].to_numpy()
    away = game_log["away"].to_numpy()
    hg = game_log["home_goals"].to_numpy(dtype=int)
    ag = game_log["away_goals"].to_numpy(dtype=int)
    hp = game_log["home_pts"].to_numpy(dtype=int)
    ap = game_log["away_pts"].to_numpy(dtype=int)
    hw = (game_log["home_result"] == "W").to_numpy()
    aw = (game_log["away_result"] == "W").to_numpy()

    rows = []
    for team in pd.unique(np.concatenate([home, away])):
        is_home = home == team
        mine = is_home | (away == team)
        gf = int(np.where(is_home, hg, ag)[mine].sum())
        ga = int(np.where(is_home, ag, hg)[mine].sum())
        pts = int(np.where(is_home, hp, ap)[mine].sum())
        w = int((np.where(is_home, hw, aw) & mine).sum())
        gp = int(mine.sum())
        otl = int((np.where(is_home, hp, ap)[mine] == 1).sum())
        rows.append({
            "team": team,
            "GP": gp,
            "W": w,
            "L": gp - w - otl,
            "OTL": otl,
            "PTS": pts,
            "GF": gf,
            "GA": ga,
            "DIFF": gf - ga,
            "home_gp": int(is_home.sum()),
            "away_gp": int((away == team).sum()),
        })
    return pd.DataFrame(rows)


# --- Stage 8 verification checks ---------------------------------------------


def check_standings_consistency(standings: pd.DataFrame, game_log: pd.DataFrame) -> None:
    """Structural and conservation checks on one simulated season."""
    n_games = len(game_log)
    assert n_games == _GAMES_1312, f"expected {_GAMES_1312} games, got {n_games}"
    assert len(standings) == _TEAMS_32, f"expected 32 teams, got {len(standings)}"

    # Per-team structure
    assert (standings["GP"] == 82).all(), "not every team played 82 games"
    assert (standings["W"] + standings["L"] + standings["OTL"] == standings["GP"]).all(), \
        "W+L+OTL != GP for some team"
    assert (standings["home_gp"] == 41).all() and (standings["away_gp"] == 41).all(), \
        "home/away split is not 41/41 for some team"

    # Points: every game awards 2 points (regulation) or 3 (extra-point game)
    pts_awarded = (game_log["home_pts"] + game_log["away_pts"]).to_numpy(dtype=int)
    assert set(np.unique(pts_awarded)) <= {2, 3}, "game awarded an invalid point split"
    n_3pt = int((pts_awarded == 3).sum())
    assert standings["PTS"].sum() == 2 * n_games + n_3pt, "league points do not conserve"

    # Wins: one winning team per game
    n_wins = int(game_log["home_result"].eq("W").sum() + game_log["away_result"].eq("W").sum())
    assert n_wins == n_games, "total wins do not equal number of games"

    # GF == GA league-wide (goals only move between teams)
    assert standings["GF"].sum() == standings["GA"].sum(), "league GF != league GA"

    # Cross-check vs the Stage 6 running-totals accumulation (2nd path)
    from simulate_games import running_team_totals
    rt = running_team_totals(game_log)
    last = rt.groupby("team", as_index=False).last()[
        ["team", "games_played", "cum_GF", "cum_GA", "cum_PTS"]
    ].rename(columns={"games_played": "GP", "cum_GF": "GF", "cum_GA": "GA", "cum_PTS": "PTS"})
    merged = standings.merge(last, on="team", suffixes=("", "_rt"))
    for col in ["GP", "GF", "GA", "PTS"]:
        assert (merged[col] == merged[f"{col}_rt"]).all(), f"{col} disagrees between paths"


def check_points_conservation_over_batch(n_seasons: int = 1_000, seed: int = 20251019) -> dict:
    """Aggregate sanity over many seasons: PPG, 3-pt share, point range."""
    schedule = pd.read_csv(SCHEDULE_CSV)
    pts_per_gp, p3_share, max_pts, min_pts = [], [], 0.0, 300.0
    for i in range(n_seasons):
        standings, _ = simulate_season(np.random.default_rng(seed + i), schedule)
        pts_per_gp.append(standings["PTS"].sum() / len(schedule))
        p3_share.append((standings["PTS"].sum() - 2 * 1_312) / 1_312)
        max_pts = max(max_pts, standings["PTS"].max())
        min_pts = min(min_pts, standings["PTS"].min())
    return {
        "n_seasons": n_seasons,
        "league_points_per_game": float(np.mean(pts_per_gp)),
        "points_per_game_sd": float(np.std(pts_per_gp)),
        "mean_3pt_game_share": float(np.mean(p3_share)),
        "max_team_points": max_pts,
        "min_team_points": min_pts,
    }


def main() -> None:
    rng, seed = np.random.default_rng(20252026), 20252026
    standings, game_log = simulate_season(rng)

    print(f"Stage 8 verification (demo season, seed {seed})")
    check_standings_consistency(standings, game_log)
    print("  [PASS] structural + conservation checks "
          "(GP=82, W+L+OTL=82, 41/41 split, points/wins conserve, GF==GA, "
          "agrees with running_team_totals)")

    print("\nDemo standings table (one simulated season):")
    print(standings.to_string(
        index=False,
        formatters={"GF_percent": "{:.3f}".format, "GF_per_game": "{:.3f}".format,
                    "GA_per_game": "{:.3f}".format},
    ))

    print("\nBatch plausibility check (1,000 seasons):")
    stats = check_points_conservation_over_batch()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    assert 2.05 <= stats["league_points_per_game"] <= 2.30, "points-per-game out of plausible range"
    # Static-talent model: teams stay at their drawn quality all season, so
    # extreme draws (e.g. 3 SD) produce point totals beyond real historical
    # bounds. Deliberately loose guard (20-145); tail behaviour documented
    # for the writeup (JLikens's own static-ability caveat).
    assert stats["min_team_points"] >= 20 and stats["max_team_points"] <= 145, \
        "team point totals implausible even for the static-talent model"
    print("  [PASS] batch plausibility (PPG in 2.05-2.30; team points in 20-145, "
          "loose guard: static-talent model has wider tails than real seasons)")


if __name__ == "__main__":
    main()
