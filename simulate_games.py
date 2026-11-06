#!/usr/bin/env python3
"""Stage 6 (part 2) — Simulate NHL regular-season games (Poisson process).

For each game, expected goals (lambda) for each team come from the talent
draw (Stage 6 part 1, assign_team_talent.py) combined with the opponent's
rate and the home-ice advantage measured from Stage 2 data:

    home_gf_lambda = 2 * home_GF_per_game * away_GA_per_game / GPG + HIA
    away_gf_lambda = 2 * away_GF_per_game * home_GA_per_game / GPG

where GPG is the league-average clean goals/game (5.4933). This is the
multiplicative (full talent expression) form, revised 2026-10-18 from the
user's original averaging form after the Stage 7 diagnostic showed the
averaging form halves the talent signal and compresses the simulated
half-season GF% spread below the observed spread. Same structure as
Maher (1982) / Dixon & Coles (1997) with league-average normalization.

Goals are Poisson draws with these lambdas. Ties after 60 minutes follow the
settled convention (2026-09-21): each team gets 1 point, and the extra point
is a coin flip (OT rates assumed to hold; shootout = coin flip). The winner
of the extra point is recorded via the W/L/OTL classification, matching real
standings bookkeeping (winner 2 pts; regulation loser 0; OT/SO loser 1).

Adapted from the user's simulate_regular_season_game() in
"Confirming True Talent Distribution.ipynb": same lambda formulas and
tie-breaking rule. Reworked for performance because this is called
1,312 times per simulated season x >=100,000 seasons:
  - a vectorized batch simulator (simulate_games) draws every game of a
    schedule in one numpy call chain — this is the primary entry point;
  - a scalar simulate_game() kept for readability/spot-checking, implemented
    on top of the same math.

Home-ice advantage is additive in goals/game, per the user's design.

Usage:
    Running this file executes the Stage 6 verification checks on fixed-seed
    batches (10k+ single-game sims and full-schedule batches).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from assign_team_talent import (
    GOALS_PER_GAME_LEAGUEAVG,
    HOME_ICE_ADVANTAGE,
    SCHEDULE_CSV,
    assign_team_talent,
)

HIA = HOME_ICE_ADVANTAGE  # 0.2563 goals/game, pooled 2005-06 -> 2025-26
GPG = GOALS_PER_GAME_LEAGUEAVG  # 5.4933 clean goals/game, same pooled data

# Lambda combination form — REVISED 2026-10-18 from the averaging form to the
# multiplicative (full talent expression) form, per the Stage 7 diagnostic
# (stage7_diagnostic_arms.py): the averaging form halves the talent signal
# (coefficient 0.5) and compressed the simulated 41-game GF% spread to 0.038
# vs the observed ~0.052. The multiplicative form expresses talent fully
# (coefficient 1.0) and matches the literature standard (Maher 1982;
# Dixon & Coles 1997: lambda = attack * opponent_defence * home_factor,
# normalized here so a league-average matchup yields GPG/2 per side):
#
#     home_lam = 2 * GF_rate_home * GA_rate_away / GPG + HIA
#     away_lam = 2 * GF_rate_away * GA_rate_home / GPG
#
# HIA remains additive in goals/game (user's locked design). A league-average
# matchup reproduces GPG total expected goals exactly; identity:
# 2*gf*ga/GPG where gf+ga=GPG per team implies lam_gf/2 + ... symmetry.


def _lambdas(
    home_teams: np.ndarray,
    away_teams: np.ndarray,
    team_data: pd.DataFrame,
    hia: float = HIA,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected-goals (lambda) arrays for arrays of home/away team codes."""
    gf = team_data["GF_per_game"]
    ga = team_data["GA_per_game"]
    home_gf = gf.loc[home_teams].to_numpy(dtype=float)
    home_ga = ga.loc[home_teams].to_numpy(dtype=float)
    away_gf = gf.loc[away_teams].to_numpy(dtype=float)
    away_ga = ga.loc[away_teams].to_numpy(dtype=float)
    home_lam = 2.0 * home_gf * away_ga / GPG + hia
    away_lam = 2.0 * away_gf * home_ga / GPG
    return home_lam, away_lam


def _points(home_goals: np.ndarray, away_goals: np.ndarray, rng: np.random.Generator):
    """Points arrays from goal arrays; ties get 1 pt each + coin-flip extra point."""
    tie = home_goals == away_goals
    home_won_flip = rng.random(tie.shape) < 0.5  # only used where tie

    home_pts = np.where(
        home_goals > away_goals, 2,
        np.where(tie, np.where(home_won_flip, 2, 1), 0),
    )
    away_pts = np.where(
        away_goals > home_goals, 2,
        np.where(tie, np.where(home_won_flip, 1, 2), 0),
    )

    # W/L/OTL classification (standings bookkeeping; winner-of-flip opponent gets OTL)
    home_w = (home_goals > away_goals) | (tie & home_won_flip)
    away_w = (away_goals > home_goals) | (tie & ~home_won_flip)
    return home_pts, away_pts, home_w, away_w


def simulate_game(
    home_team: str,
    away_team: str,
    team_data: pd.DataFrame,
    rng: np.random.Generator,
    hia: float = HIA,
) -> dict:
    """Simulate one game; returns goals, points, and the lambdas used."""
    home_lam, away_lam = _lambdas(np.array([home_team]), np.array([away_team]), team_data, hia)
    home_goals = int(rng.poisson(home_lam[0]))
    away_goals = int(rng.poisson(away_lam[0]))
    home_pts, away_pts, home_w, away_w = _points(np.array([home_goals]), np.array([away_goals]), rng)
    return {
        "home": home_team,
        "away": away_team,
        "home_lambda": float(home_lam[0]),
        "away_lambda": float(away_lam[0]),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_pts": int(home_pts[0]),
        "away_pts": int(away_pts[0]),
        "home_result": "W" if home_w[0] else ("OTL" if home_pts[0] == 1 else "L"),
        "away_result": "W" if away_w[0] else ("OTL" if away_pts[0] == 1 else "L"),
    }


def simulate_games(
    schedule: pd.DataFrame,
    team_data: pd.DataFrame,
    rng: np.random.Generator,
    hia: float = HIA,
) -> pd.DataFrame:
    """Simulate every game in a schedule DataFrame (vectorized).

    Arguments:
        schedule: DataFrame with columns home_abbrev, away_abbrev (the Stage 5
                  2025-26 schedule schema; any order-preserving subset works).
        team_data: talent table from assign_team_talent (indexed by team code).
        rng: numpy Generator controlling all draws for this simulation.
        hia: home-ice advantage in goals/game added to the home team's lambda.

    Returns:
        DataFrame, one row per scheduled game: home, away, lambdas, goals,
        points, and W/L/OTL result per team. Preserves schedule order.
    """
    home_teams = schedule["home_abbrev"].to_numpy()
    away_teams = schedule["away_abbrev"].to_numpy()

    unknown = (set(home_teams) | set(away_teams)) - set(team_data.index)
    if unknown:
        raise KeyError(f"schedule contains team codes absent from talent table: {sorted(unknown)}")

    home_lam, away_lam = _lambdas(home_teams, away_teams, team_data, hia)
    home_goals = rng.poisson(home_lam)
    away_goals = rng.poisson(away_lam)
    home_pts, away_pts, home_w, away_w = _points(home_goals, away_goals, rng)

    return pd.DataFrame(
        {
            "home": home_teams,
            "away": away_teams,
            "home_lambda": home_lam,
            "away_lambda": away_lam,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_pts": home_pts,
            "away_pts": away_pts,
            "home_result": np.where(home_w, "W", np.where(home_pts == 1, "OTL", "L")),
            "away_result": np.where(away_w, "W", np.where(away_pts == 1, "OTL", "L")),
        }
    )


def running_team_totals(game_log: pd.DataFrame) -> pd.DataFrame:
    """Per-team running totals after each game of a (single-season) game log.

    Vectorized accumulation via numpy cumsum; consumes the DataFrame returned
    by simulate_games() (or a subset of it, in order).

    Arguments:
        game_log: game rows with columns home, away, home_goals, away_goals,
                  home_pts, away_pts, in the order the games were played.

    Returns:
        Long-format tidy table, one row per (team, game played), sorted by
        team then game order, with columns:
            team          — team code
            game_index    — 0-based position of the game in the log
            games_played  — 1..N cumulative count for that team
            cum_GF        — clean goals for, accumulated through this game
            cum_GA        — clean goals against, accumulated through this game
            cum_PTS       — standings points, accumulated through this game
        The row where games_played == 41 is the half-season cut for Stage 7;
        the last row per team is the full-season standings total.
    """
    n = len(game_log)
    home = game_log["home"].to_numpy()
    away = game_log["away"].to_numpy()
    home_gf = game_log["home_goals"].to_numpy(dtype=int)
    away_gf = game_log["away_goals"].to_numpy(dtype=int)
    home_pts = game_log["home_pts"].to_numpy(dtype=int)
    away_pts = game_log["away_pts"].to_numpy(dtype=int)

    teams = pd.unique(np.concatenate([home, away]))
    idx_all = np.arange(n)
    out = []
    for team in teams:
        is_home = home == team
        mine = is_home | (away == team)
        gf = np.where(is_home, home_gf, away_gf)[mine]
        ga = np.where(is_home, away_gf, home_gf)[mine]
        pts = np.where(is_home, home_pts, away_pts)[mine]
        out.append(pd.DataFrame({
            "team": team,
            "game_index": idx_all[mine],  # positions in the log of the team's own games
            "games_played": np.arange(1, mine.sum() + 1),
            "cum_GF": np.cumsum(gf),
            "cum_GA": np.cumsum(ga),
            "cum_PTS": np.cumsum(pts),
        }))
    return pd.concat(out, ignore_index=True)


# --- Stage 6 verification checks ---------------------------------------------

def _load_inputs(seed: int = 20252026):
    rng = np.random.default_rng(seed)
    team_data = assign_team_talent(rng)
    schedule = pd.read_csv(SCHEDULE_CSV)
    return rng, team_data, schedule


def check_scalar_consistency(n_games: int = 100_000) -> None:
    """A single fixed matchup: mean goals converge to the lambdas; point sums sane."""
    rng, team_data, _ = _load_inputs()
    fixed = {"home": "TBL", "away": "BOS"}  # biggest talent gap in the seed draw

    rows = [simulate_game(fixed["home"], fixed["away"], team_data, rng) for _ in range(n_games)]
    df = pd.DataFrame(rows)
    lam_h, lam_a = df["home_lambda"].iloc[0], df["away_lambda"].iloc[0]

    se_h = np.sqrt(lam_h / n_games)
    se_a = np.sqrt(lam_a / n_games)
    z_h = (df["home_goals"].mean() - lam_h) / se_h
    z_a = (df["away_goals"].mean() - lam_a) / se_a
    print(f"[scalar] {n_games} sims of {fixed['home']} vs {fixed['away']}")
    print(f"  lambda home {lam_h:.4f}: sim mean {df['home_goals'].mean():.4f} (z={z_h:+.2f})")
    print(f"  lambda away {lam_a:.4f}: sim mean {df['away_goals'].mean():.4f} (z={z_a:+.2f})")
    assert abs(z_h) < 4 and abs(z_a) < 4, "simulated goals do not match Poisson means"

    pts_per_game = df["home_pts"] + df["away_pts"]
    assert set(pts_per_game.unique()) <= {2, 3}, "points per game must be 2 (reg decision) or 3 (OT/SO)"
    share_2pts = (pts_per_game == 2).mean()
    from math import factorial
    p_tie = float(sum(  # P(tie) for two independent Poissons: sum_k P(home=k)P(away=k)
        np.exp(-(lam_h + lam_a)) * (lam_h ** k / factorial(k)) * (lam_a ** k / factorial(k))
        for k in range(0, 15)
    ))
    expected_reg = 1.0 - p_tie  # share of games decided in regulation (2 points total)
    print(f"  regulation decisions: {share_2pts:.4f} (analytic P(no tie)={expected_reg:.4f})")
    assert abs(share_2pts - expected_reg) < 0.01, "tie rate deviates from analytic Poisson tie rate"

    result_counts = df["home_result"].value_counts().to_dict()
    print(f"  home result mix: {result_counts}")
    assert df["home_result"].isin(["W", "L", "OTL"]).all()
    # TBL is much stronger; home team should win far more than lose in regulation
    assert df["home_pts"].mean() > df["away_pts"].mean(), "stronger team must earn more points on average"


def check_schedule_batch(n_seasons: int = 1_000) -> None:
    """Full-schedule batches: lambda identity, point conservation, sanity of aggregates."""
    rng, team_data, schedule = _load_inputs()

    import time
    t0 = time.perf_counter()
    logs = []
    for s in range(n_seasons):
        log = simulate_games(schedule, team_data, rng)
        log["season"] = s
        logs.append(log)
    dt = time.perf_counter() - t0
    all_games = pd.concat(logs, ignore_index=True)

    print(f"\n[batch] {n_seasons} full seasons ({len(all_games):,} games) in {dt:.2f}s "
          f"({dt / n_seasons * 1000:.1f} ms/season)")

    # (a) points conservation: every game awards 2 (reg) or 3 (OT/SO) points
    pts = all_games["home_pts"] + all_games["away_pts"]
    assert set(pts.unique()) <= {2, 3}
    print(f"  points/game: 2 pts in {(pts == 2).mean():.4f} of games, 3 pts otherwise")

    # (b) each game's lambdas: multiplicative-form identity
    #     lam_h - lam_a == 2*(home_gf*away_ga - away_gf*home_ga)/GPG + hia
    gf = team_data["GF_per_game"]
    ga = team_data["GA_per_game"]
    expected_diff = 2.0 * (
        gf.loc[all_games["home"]].to_numpy() * ga.loc[all_games["away"]].to_numpy()
        - gf.loc[all_games["away"]].to_numpy() * ga.loc[all_games["home"]].to_numpy()
    ) / GPG
    lam_diff = all_games["home_lambda"].to_numpy() - all_games["away_lambda"].to_numpy()
    assert np.allclose(lam_diff, expected_diff + HIA, atol=1e-12), "lambda difference identity failed"
    print("  lambda-difference identity (multiplicative form + HIA): PASS")

    # (c) W/L/OTL bookkeeping: W + L + OTL == games per team; every W has an opposing L or OTL
    for col, res_col in (("home", "home_result"), ("away", "away_result")):
        assert all_games[res_col].isin(["W", "L", "OTL"]).all()
    # every game has exactly one W
    n_winners = (all_games["home_result"] == "W").astype(int) + (all_games["away_result"] == "W").astype(int)
    assert (n_winners == 1).all(), "every game must have exactly one winner"
    # L (regulation loss) pairs exactly with opponent W
    reg = all_games["home_result"] == "L"
    assert (all_games.loc[reg, "away_result"] == "W").all()
    print("  W/L/OTL bookkeeping (exactly one W per game; L <=> opponent W): PASS")

    # (d) points mean per team-game: league-wide average points per team-game.
    # Real NHL (cap era): teams average ~91-95 pts / 82 gp = ~1.12-1.16.
    # Our model slightly underproduces 3-pt games (Poisson underestimates ties,
    # as JLikens noted), so the band is anchored below the real value.
    per_team_pts = np.concatenate([all_games["home_pts"].to_numpy(), all_games["away_pts"].to_numpy()])
    print(f"  mean points per team-game: {per_team_pts.mean():.4f} (real NHL ~1.12-1.16)")
    assert 1.00 < per_team_pts.mean() < 1.25


def check_running_totals(n_seasons: int = 200) -> None:
    """running_team_totals: final rows match from-scratch standings; 41-game cut sane."""
    rng, team_data, schedule = _load_inputs()

    final_gf_ga = {}
    final_pts = {}
    cut_rows = []  # every team's games_played==41 row across seasons
    for s in range(n_seasons):
        log = simulate_games(schedule, team_data, rng)
        totals = running_team_totals(log)

        # final row per team must match a from-scratch pandas groupby
        home = log.assign(team=log["home"], gf=log["home_goals"], ga=log["away_goals"], pts=log["home_pts"])
        away = log.assign(team=log["away"], gf=log["away_goals"], ga=log["home_goals"], pts=log["away_pts"])
        agg = pd.concat([home, away]).groupby("team")[["gf", "ga", "pts"]].sum()
        last = totals.sort_values("games_played").groupby("team").last()
        last = last.loc[agg.index]  # align row order before comparing
        assert (last["cum_GF"] == agg["gf"]).all(), "final cum_GF mismatch"
        assert (last["cum_GA"] == agg["ga"]).all(), "final cum_GA mismatch"
        assert (last["cum_PTS"] == agg["pts"]).all(), "final cum_PTS mismatch"

        cut = totals[totals["games_played"] == 41]
        assert len(cut) == 32, "half-season cut must have exactly one row per team"
        cut_rows.append(cut)
        # league-wide GF == GA over a COMPLETE season (each goal once for, once against);
        # this need NOT hold at the 41-game cut (opponents play different game numbers)
        assert last["cum_GF"].sum() == last["cum_GA"].sum(), "full-season GF != GA"
        final_gf_ga.update({t: (g, a) for t, g, a in zip(last.index, last["cum_GF"], last["cum_GA"])})
        final_pts.update(last["cum_PTS"].to_dict())

    cuts = pd.concat(cut_rows, ignore_index=True)
    gp_range = cuts.groupby("team")["games_played"].agg(["min", "max"])
    assert (gp_range["min"] == 41).all() and (gp_range["max"] == 41).all()
    print(f"[totals] {n_seasons} seasons: final rows match from-scratch aggregation (all teams, all seasons)")
    print(f"  full-season GF_total == GA_total check: PASS (each of {n_seasons} seasons)")
    print(f"  41-game cut: 32 teams x {n_seasons} seasons")
    print(f"  mean team GF at 41 games: {cuts['cum_GF'].mean():.1f}, GA: {cuts['cum_GA'].mean():.1f}, "
          f"PTS: {cuts['cum_PTS'].mean():.1f} (~half of a full season, sane)")


if __name__ == "__main__":
    check_scalar_consistency()
    check_schedule_batch()
    check_running_totals()
    print("\nAll Stage 6 verification checks PASSED")
