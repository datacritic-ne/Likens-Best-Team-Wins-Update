#!/usr/bin/env python3
"""Stage 7 part 2 — 100,000 half-season simulations (vectorized).

Runs the Stage 7 part-1 design (fresh talent draw per season, Poisson games
over the 2025-26 schedule, per-team GF% at the 41-game cut) 100,000 times,
fully vectorized across seasons in chunks:

  - talents drawn as a (chunk, 32) block per chunk;
  - per-game lambdas computed as (chunk, 1312) arrays from code-indexed
    schedule lookups (home/away code -> column index, precomputed once);
  - goals drawn in one rng.poisson call per (chunk x games) block;
  - the 41-game cut done WITHOUT running_team_totals: each team's first-41
    game positions and home/away flags in the schedule are precomputed once,
    so gf41/ga41 are two masked sums per team over the (chunk, games) arrays.

Reproducibility: single Generator seeded with SEED_BASE (20261018), consumed
in fixed chunk order. NOTE: the draw ORDER differs from the scalar part-1
path (batched Poisson vs per-season calls), so individual seasons will not
match part-1 seeds — but this run itself is exactly reproducible, which is
the requirement we agreed on.

NOTE 2026-10-18: revised to the multiplicative (full talent expression)
lambda form — home_lam = 2*gf_h*ga_a/GPG + HIA, away_lam = 2*gf_a*ga_h/GPG
— after stage7_diagnostic_arms.py showed the averaging form halves the
talent signal and compresses the half-season GF% spread (0.038 vs observed
~0.052). Multiplicative form matches Maher (1982)/Dixon-Coles (1997).

Output:
  outputs/stage7_halfseason_100k.parquet  — one row per (season, team):
      season, team, talent_gfpct, gf41, ga41, gfpct41
  outputs/reports/stage7_halfseason_100k_report.txt — summary + checks
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from assign_team_talent import (
    GFDIST_MEAN,
    GFDIST_STD,
    GOALS_PER_GAME_LEAGUEAVG,
    SCHEDULE_CSV,
    HOME_ICE_ADVANTAGE,
)

SEED_BASE = 20261018
N_SEASONS = 100_000
CHUNK = 2_000
CUT = 41
HIA = HOME_ICE_ADVANTAGE
GPG = GOALS_PER_GAME_LEAGUEAVG
OUT_PARQUET = "/workspace/projects/hockey-lab/analyses/likens/outputs/stage7_halfseason_100k.parquet"
OUT_REPORT = "/workspace/projects/hockey-lab/analyses/likens/outputs/reports/stage7_halfseason_100k_report.txt"


def _precompute(schedule: pd.DataFrame):
    """Static structures: code index, per-team first-41 game positions + home flag."""
    codes = sorted(set(schedule["home_abbrev"]) | set(schedule["away_abbrev"]))
    assert len(codes) == 32
    code_idx = {c: i for i, c in enumerate(codes)}

    home_col = schedule["home_abbrev"].map(code_idx).to_numpy()
    away_col = schedule["away_abbrev"].map(code_idx).to_numpy()
    n_games = len(schedule)

    # per team: positions of its games (in schedule order) and is_home flags
    team_pos, team_home = {}, {}
    for c, i in code_idx.items():
        pos = np.where((home_col == i) | (away_col == i))[0]
        assert len(pos) == 82
        team_pos[c] = pos[:CUT]
        team_home[c] = home_col[pos[:CUT]] == i
    return codes, code_idx, home_col, away_col, n_games, team_pos, team_home


def run(n_seasons: int = N_SEASONS, chunk: int = CHUNK, seed: int = SEED_BASE):
    schedule = pd.read_csv(SCHEDULE_CSV)
    codes, code_idx, home_col, away_col, n_games, team_pos, team_home = _precompute(schedule)
    rng = np.random.default_rng(seed)

    all_rows = []
    t0 = time.perf_counter()
    done = 0
    while done < n_seasons:
        n = min(chunk, n_seasons - done)

        # --- talent draw: (n, 32) ---
        gfpct = rng.normal(GFDIST_MEAN, GFDIST_STD, size=(n, 32))
        gf_rate = GOALS_PER_GAME_LEAGUEAVG * gfpct          # (n, 32)
        ga_rate = GOALS_PER_GAME_LEAGUEAVG * (1.0 - gfpct)

        # --- lambdas per game: (n, n_games) — multiplicative (full expression) form ---
        home_gf = gf_rate[:, home_col]                       # talent of home team per game
        away_ga = ga_rate[:, away_col]
        away_gf = gf_rate[:, away_col]
        home_ga = ga_rate[:, home_col]
        home_lam = 2.0 * home_gf * away_ga / GPG + HIA
        away_lam = 2.0 * away_gf * home_ga / GPG

        home_goals = rng.poisson(home_lam)
        away_goals = rng.poisson(away_lam)

        # --- 41-game cut per team ---
        gf41_all = np.empty((n, 32), dtype=np.int64)
        ga41_all = np.empty((n, 32), dtype=np.int64)
        for ti, c in enumerate(codes):
            pos = team_pos[c]
            is_home = team_home[c]
            gf41_all[:, ti] = home_goals[:, pos[is_home]].sum(axis=1) + away_goals[:, pos[~is_home]].sum(axis=1)
            ga41_all[:, ti] = away_goals[:, pos[is_home]].sum(axis=1) + home_goals[:, pos[~is_home]].sum(axis=1)
        seas = np.repeat(np.arange(done, done + n), 32)
        chunk_df = pd.DataFrame(
            {
                "season": seas,
                "team": np.tile(np.array(codes, dtype=object), n),
                "talent_gfpct": gfpct.reshape(-1),
                "gf41": gf41_all.reshape(-1),
                "ga41": ga41_all.reshape(-1),
            }
        )
        all_rows.append(chunk_df)
        done += n
        el = time.perf_counter() - t0
        print(f"  {done:,}/{n_seasons:,} seasons ({el:.0f}s elapsed, {el / done * 1e6:.0f} us/season)", flush=True)

    df = pd.concat(all_rows, ignore_index=True)
    df["gfpct41"] = df["gf41"] / (df["gf41"] + df["ga41"])
    return df


def checks_and_report(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"rows: {len(df):,} (expect {N_SEASONS * 32:,})")
    assert len(df) == N_SEASONS * 32

    # per-season structure: 32 teams, mean GF% ~ 0.5
    per_season = df.groupby("season")["gfpct41"]
    season_means = per_season.mean()
    lines.append(f"season mean GF%: {season_means.mean():.5f} (SE of the mean across seasons: {season_means.std() / np.sqrt(N_SEASONS):.5f})")
    assert abs(season_means.mean() - 0.5) < 0.001

    # the headline: pooled SD of 41-game GF% across all sims
    pooled_sd = df["gfpct41"].std(ddof=1)
    lines.append(f"pooled SD of 41-game GF%: {pooled_sd:.5f}")
    lines.append(f"  talent SD alone: {GFDIST_STD}")
    lines.append(f"  mean per-season SD: {per_season.std().mean():.5f}")

    # per-season SD across 100k seasons: its own mean and spread (MC error tiny)
    psd = per_season.std()
    lines.append(f"per-season SD distribution: min {psd.min():.4f}, mean {psd.mean():.4f}, max {psd.max():.4f}")

    # conservation-ish: league GF vs GA at the cut (approx; report the gap)
    tot_gf = df["gf41"].sum()
    tot_ga = df["ga41"].sum()
    lines.append(f"total GF vs GA at cut: {tot_gf:,} vs {tot_ga:,} (gap {tot_gf - tot_ga:+,} — expected: teams cut at different schedule points)")

    # talent-GF% correlation pooled
    corr = df["gfpct41"].corr(df["talent_gfpct"])
    lines.append(f"pooled corr(talent, GF%41): {corr:.4f} (theory ~ talentSD/sqrt(talentSD^2+noiseSD^2))")

    lines.append("")
    lines.append("Checks: PASS")
    return "\n".join(lines)


if __name__ == "__main__":
    df = run()
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nwrote {OUT_PARQUET}")
    report = checks_and_report(df)
    with open(OUT_REPORT, "w") as f:
        f.write(report + "\n")
    print(f"wrote {OUT_REPORT}\n")
    print(report)
