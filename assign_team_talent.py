#!/usr/bin/env python3
"""Stage 6 (part 1) — Assign true talent (GF%) to the 32 NHL teams.

Draws each team's true-talent GF% from the league talent distribution
estimated via the Stage 3/4 bootstrap decomposition, then converts each
draw into expected average goals for/against per game using the
league-wide scoring environment measured from the Stage 2 clean data.

Constants (locked 2026-10-13, from hockey-lab/analyses/likens data):
  - talent distribution: normal, mean 0.5, SD 0.0415
    (mean implied talent SD over the 18 seasons in stage4_fitted_table.csv)
  - scoring environment: 5.4933 total clean (non-EN, non-SO) goals/game
    (pooled mean over all 22,632 Stage 2 games, 2005-06 -> 2025-26)
  - home-ice advantage: 0.2563 goals/game (home clean GF - away clean GF,
    same pooled data) — used by the game simulator, recorded here for provenance

Adapted from the user's assign_team_talent() in
"Confirming True Talent Distribution.ipynb" (github.com/datacritic-ne/likensproject):
same formulas and team-DataFrame layout; refinements are (a) team codes read
from the actual 2025-26 schedule so they are guaranteed to match the simulator,
(b) numpy Generator-based RNG with explicit seed (reproducible, no global state),
(c) input validation and post-draw structural checks.

Usage:
    from assign_team_talent import assign_team_talent
    rng = np.random.default_rng(20252026)
    team_data = assign_team_talent(rng=rng)

Running this file directly executes a self-check on a fixed-seed draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Locked constants (see module docstring for provenance) -----------------
GFDIST_MEAN = 0.5
GFDIST_STD = 0.0415
GOALS_PER_GAME_LEAGUEAVG = 5.4933
HOME_ICE_ADVANTAGE = 0.2563  # for the game simulator (Stage 6, part 2)

# Schedule file this project's simulator will drive the season from.
SCHEDULE_CSV = "/workspace/projects/hockey-lab/analyses/likens/data/processed/schedule_20252026.csv"


def _schedule_team_codes(schedule_csv: str = SCHEDULE_CSV) -> list[str]:
    """The 32 team codes present in the 2025-26 schedule, sorted.

    Reading codes from the schedule (rather than hardcoding a list) guarantees
    the simulator never looks up a code missing from the talent table. Note the
    API spelling VGK (not hockey-reference's VEG).
    """
    sched = pd.read_csv(schedule_csv)
    codes = sorted(set(sched["home_abbrev"]) | set(sched["away_abbrev"]))
    if len(codes) != 32:
        raise ValueError(f"Expected 32 team codes in schedule, found {len(codes)}: {codes}")
    return codes


def assign_team_talent(
    rng: np.random.Generator,
    gfdist_mean: float = GFDIST_MEAN,
    gfdist_std: float = GFDIST_STD,
    goals_per_game_leagueavg: float = GOALS_PER_GAME_LEAGUEAVG,
    schedule_csv: str = SCHEDULE_CSV,
) -> pd.DataFrame:
    """Draw true-talent GF% for all 32 teams and compute per-game GF/GA rates.

    Arguments:
        rng: numpy Generator (e.g. np.random.default_rng(seed)); passed in so the
             draw is reproducible and call sites control the random stream.
        gfdist_mean: mean of the league talent (GF%) distribution.
        gfdist_std: SD of the league talent distribution (Stage 4: 0.0415).
        goals_per_game_leagueavg: league average total (GF+GA) clean goals/game.
        schedule_csv: schedule whose team codes define the 32 teams.

    Returns:
        DataFrame indexed by team code with columns:
        GF_percent, GF_percent_rank, GF_per_game, GA_per_game.
    """
    if gfdist_std <= 0:
        raise ValueError(f"gfdist_std must be positive, got {gfdist_std}")
    if not (0 < gfdist_mean < 1):
        raise ValueError(f"gfdist_mean must be in (0, 1), got {gfdist_mean}")
    if goals_per_game_leagueavg <= 0:
        raise ValueError(f"goals_per_game_leagueavg must be positive, got {goals_per_game_leagueavg}")

    teams = _schedule_team_codes(schedule_csv)

    # Draw GF% for each team; clip to [0, 1] (guards the vanishingly rare
    # extreme draw; with SD=0.0415 a clip requires |z| > 12).
    gfpct_raw = rng.normal(loc=gfdist_mean, scale=gfdist_std, size=len(teams))
    n_clipped = int(((gfpct_raw < 0.0) | (gfpct_raw > 1.0)).sum())
    gfpct = np.clip(gfpct_raw, 0.0, 1.0)

    gf_game = goals_per_game_leagueavg * gfpct
    ga_game = goals_per_game_leagueavg * (1.0 - gfpct)

    team_data = pd.DataFrame(
        {
            "GF_percent": gfpct,
            "GF_percent_rank": pd.Series(gfpct).rank(method="min", ascending=False).astype(int).to_numpy(),
            "GF_per_game": gf_game,
            "GA_per_game": ga_game,
        },
        index=pd.Index(teams, name="Team"),
    )

    # --- Structural checks (fail loudly rather than corrupt downstream sims) ---
    assert np.isclose(team_data["GF_per_game"] + team_data["GA_per_game"],
                      goals_per_game_leagueavg).all(), "GF_per_game + GA_per_game must equal league average"
    assert sorted(team_data["GF_percent_rank"]) == list(range(1, len(teams) + 1)), "ranks must be a 1..32 permutation"
    assert team_data["GF_percent"].between(0.0, 1.0).all(), "GF_percent must lie in [0, 1]"
    assert team_data.index.is_unique, "team codes must be unique"

    if n_clipped:
        print(f"NOTE: {n_clipped} talent draw(s) clipped to [0, 1]")

    return team_data


# --- Self-check on a fixed-seed draw ----------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(20252026)
    td = assign_team_talent(rng)

    print(f"Talent distribution: N(mean={GFDIST_MEAN}, sd={GFDIST_STD})")
    print(f"League scoring environment: {GOALS_PER_GAME_LEAGUEAVG} clean goals/game")
    print(f"Home-ice advantage (for game sim): {HOME_ICE_ADVANTAGE} goals/game\n")
    print(td.round(4).to_string())

    print("\n--- Checks ---")
    print(f"teams: {len(td)} (unique: {td.index.is_unique})")
    print(f"mean drawn GF%: {td['GF_percent'].mean():.4f} (target {GFDIST_MEAN})")
    print(f"sd   drawn GF%: {td['GF_percent'].std(ddof=1):.4f} (draw-to-draw around {GFDIST_STD}; "
          f"32 draws sample SE ~ {GFDIST_STD / np.sqrt(32):.4f})")
    best_team = td["GF_percent"].idxmax()
    worst_team = td["GF_percent"].idxmin()
    print(f"best team: {best_team} ({td.loc[best_team, 'GF_percent']:.4f}); "
          f"worst: {worst_team} ({td.loc[worst_team, 'GF_percent']:.4f})")

    # Reproducibility: an identical Generator state must reproduce the table.
    td2 = assign_team_talent(np.random.default_rng(20252026))
    assert td.equals(td2), "fixed-seed draw did not reproduce"
    print("reproducibility (same seed -> identical table): PASS")
