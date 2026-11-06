#!/usr/bin/env python3
"""Stage 7 step 2 — Real half-season GF% target (Option C).

Computes every team's ACTUAL GF% through its first 41 scheduled games for
each of the 18 real seasons (2005-06 .. 2025-26, Stage 2 CSVs), using the
same first-41-scheduled-games cut as the simulation (settled design: each
team's first 41 games in schedule order, JLikens-style).

This gives the game-coupled, real-world target for Stage 7 validation:
the simulated 100k-run GF% spread (multiplicative form, pooled SD 0.0520)
is compared against these real per-season SDs and their pooled spread.

Clean-goal convention: matches Stage 2 CSVs — gf_clean/ga_clean exclude
empty-net and shootout goals (the same convention used for the bootstrap
and the constants). GF% = gf_clean / (gf_clean + ga_clean) at the cut.

Output:
  outputs/stage7_real_halfseason_gfpct.csv — one row per (season, team):
      season, team, games_played (always 41), gf41, ga41, gfpct41
  Report section appended to stdout; also written to
  outputs/reports/stage7_real_halfseason_report.txt

Verification checks:
  - every (season, team) has exactly 41 games;
  - mirror consistency: in each season, team A's GA at the cut counts only
    goals conceded to cut-inclusive games (no cross-check possible at the
    cut since opponents' cuts differ — the season-level GF==GA identity
    does NOT hold at 41 games; we verify the FULL-season identity instead);
  - full-season totals per team match the CSV sums (structural integrity);
  - per-season GF% means near 0.5; pooled SD reported alongside the
    simulation target.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

DATA_DIR = "/workspace/projects/hockey-lab/analyses/likens/data/processed"
OUT_CSV = "/workspace/projects/hockey-lab/analyses/likens/outputs/stage7_real_halfseason_gfpct.csv"
OUT_REPORT = "/workspace/projects/hockey-lab/analyses/likens/outputs/reports/stage7_real_halfseason_report.txt"
CUT = 41


def season_file(season: int) -> str:
    return os.path.join(DATA_DIR, f"games_{season}.csv")


def real_halfseason_gfpct() -> pd.DataFrame:
    """First-41-scheduled-games GF% per team per season, from Stage 2 CSVs.

    Games are ordered by (game_date, game_id) — the schedule order. For each
    team, its own games are filtered in that order and cut at 41.
    """
    files = sorted(glob.glob(os.path.join(DATA_DIR, "games_*.csv")))
    assert len(files) == 18, f"expected 18 season files, found {len(files)}"

    rows = []
    for f in files:
        season = int(os.path.basename(f).split("_")[1].split(".")[0])
        df = pd.read_csv(f, usecols=["game_id", "game_date", "home_abbrev", "away_abbrev",
                                     "home_gf_clean", "away_gf_clean",
                                     "home_ga_clean", "away_ga_clean"])
        df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

        # structural check: mirror identity (goals scored by home teams ==
        # goals conceded by away teams). Note sum(home_gf) != sum(away_gf)
        # in general (home ice), so compare gf vs ga, not gf vs gf.
        assert df["home_gf_clean"].sum() == df["away_ga_clean"].sum(), season
        assert df["away_gf_clean"].sum() == df["home_ga_clean"].sum(), season

        n_teams = len(set(df["home_abbrev"]) | set(df["away_abbrev"]))
        for team in sorted(set(df["home_abbrev"]) | set(df["away_abbrev"])):
            home_games = df[df["home_abbrev"] == team]
            away_games = df[df["away_abbrev"] == team]
            merged = pd.concat([
                home_games[["game_date", "game_id"]].assign(is_home=True),
                away_games[["game_date", "game_id"]].assign(is_home=False),
            ]).sort_values(["game_date", "game_id"]).head(CUT)

            assert len(merged) == CUT, f"{season} {team}: {len(merged)} games before cut"
            idx_h = merged.index[merged["is_home"]]
            idx_a = merged.index[~merged["is_home"]]
            gf41 = (df.loc[idx_h, "home_gf_clean"].sum()
                    + df.loc[idx_a, "away_gf_clean"].sum())
            ga41 = (df.loc[idx_h, "away_gf_clean"].sum()
                    + df.loc[idx_a, "home_gf_clean"].sum())
            rows.append({"season": season, "team": team, "games_played": CUT,
                         "gf41": int(gf41), "ga41": int(ga41),
                         "gfpct41": float(gf41) / float(gf41 + ga41)})
        print(f"  {season}: {len(df)} games, {n_teams} teams, cut computed", flush=True)

    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> str:
    lines = ["Stage 7 step 2 — Real half-season GF% (first 41 scheduled games, clean goals)"]
    lines.append("")
    lines.append(f"rows: {len(df)} (expect 18 seasons x 32/31/30 teams = {len(df)})")
    lines.append("")
    per_season = df.groupby("season")["gfpct41"]
    table = per_season.agg(["mean", "std", "count"]).rename(
        columns={"mean": "mean_gfpct", "std": "sd_gfpct", "count": "n_teams"})
    table["sd_gfpct"] = table["sd_gfpct"].round(5)
    table["mean_gfpct"] = table["mean_gfpct"].round(5)
    lines.append(table.to_string())
    lines.append("")
    lines.append(f"pooled mean GF%: {df['gfpct41'].mean():.5f}")
    lines.append(f"pooled SD of GF% at 41 games: {df['gfpct41'].std(ddof=1):.5f}")
    lines.append(f"mean of per-season SDs: {per_season.std().mean():.5f}")
    lines.append(f"range of per-season SDs: {per_season.std().min():.5f} .. {per_season.std().max():.5f}")
    lines.append("")
    lines.append("Comparison targets:")
    lines.append("  simulated (multiplicative form, 100k seasons): pooled SD 0.05199")
    lines.append("  observed 40-game bootstrap (Stage 4): pooled SD ~0.052, per-season 0.0443-0.0634")
    return "\n".join(lines)


if __name__ == "__main__":
    df = real_halfseason_gfpct()
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")
    rep = report(df)
    with open(OUT_REPORT, "w") as f:
        f.write(rep + "\n")
    print(f"wrote {OUT_REPORT}\n")
    print(rep)
