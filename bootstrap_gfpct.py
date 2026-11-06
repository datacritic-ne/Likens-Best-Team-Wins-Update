"""Stage 3/4 — bootstrap GF% functions (JLikens replication).

Methods (per plan §7 Stage 3 + user spec 2026-10-08):
- Per team-season: the 82 regular-season games (41 home, 41 away).
- Goals used are the "clean" columns: empty-net goals and shootout goals
  excluded (cleaned at Stage 2 ingest; SO games contribute regulation+OT
  goals only).
- Each bootstrap rep: draw 40 games WITH replacement from the team's 82,
  GF% = sum(clean GF) / (sum(clean GF) + sum(clean GA)) over those 40.
- Default 100,000 reps per team-season. Fixed documented seed per season
  for reproducibility (rule 13).

Outputs (Stage 4 run):
- data/clean/bootstrap_long.parquet   — long format: season, team, rep, gf
- outputs/bootstrap_per_rep_summary.parquet — per-rep per-season mean/SD
- outputs/reports/stage3_handcheck.txt      — hand-check verification

CLI:
  python bootstrap_gfpct.py --handcheck          # Stage 3 verification only
  python bootstrap_gfpct.py --season 20052006    # one season, 100k reps
  python bootstrap_gfpct.py --all                # all seasons (Stage 4)
  python bootstrap_gfpct.py --all --n-boot 2000  # JLikens-comparable setup
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
PROCESSED = os.path.join(ROOT, "data", "processed")
CLEAN = os.path.join(ROOT, "data", "clean")
REPORTS = os.path.join(ROOT, "outputs", "reports")
OUT_BOOT = os.path.join(CLEAN, "bootstrap_long.parquet")
OUT_SUM = os.path.join(ROOT, "outputs", "bootstrap_per_rep_summary.parquet")

N_SAMPLE = 40          # games drawn per bootstrap rep (JLikens)
DEFAULT_N_BOOT = 100_000
SEED_BASE = 20261008   # documented seed base; season offset added

ALL_SEASONS = ["20052006", "20062007", "20072008", "20082009", "20092010",
               "20102011", "20112012", "20132014", "20142015", "20152016",
               "20162017", "20172018", "20182019", "20212022", "20222023",
               "20232024", "20242025", "20252026"]


def load_season(season):
    """Load one season CSV and validate structure.

    Returns DataFrame with one row per team-game:
      columns: season, team, game_idx, gf_clean, ga_clean, is_home
    Checks: 82 games per team, 41 home + 41 away, no missing values,
            every game's clean goals <= raw scores, SO games contribute
            only regulation+OT goals (ga/gf <= raw always true), and
            per-game gf+ga consistency between the two team rows.
    """
    path = os.path.join(PROCESSED, f"games_{season}.csv")
    df = pd.read_csv(path)
    assert len(df) in (1230, 1271, 1312), f"unexpected row count {len(df)}"

    # consistency: clean <= raw for both teams in every game
    assert (df["home_gf_clean"] <= df["home_score"]).all(), "home clean GF > raw"
    assert (df["away_gf_clean"] <= df["away_score"]).all(), "away clean GF > raw"
    # mirror consistency: home gf_clean == away ga_clean
    assert (df["home_gf_clean"] == df["away_ga_clean"]).all(), "mirror GF/GA mismatch"
    assert (df["away_gf_clean"] == df["home_ga_clean"]).all(), "mirror GF/GA mismatch"
    assert df[["home_gf_clean", "away_gf_clean", "home_ga_clean",
               "away_ga_clean"]].notna().all().all(), "NA in clean columns"

    home = pd.DataFrame({
        "season": season,
        "team": df["home_abbrev"],
        "game_idx": np.arange(len(df)),
        "gf_clean": df["home_gf_clean"],
        "ga_clean": df["home_ga_clean"],
        "is_home": True,
    })
    away = pd.DataFrame({
        "season": season,
        "team": df["away_abbrev"],
        "game_idx": np.arange(len(df)),
        "gf_clean": df["away_gf_clean"],
        "ga_clean": df["away_ga_clean"],
        "is_home": False,
    })
    tg = pd.concat([home, away], ignore_index=True)

    counts = tg.groupby("team").agg(n=("game_idx", "size"),
                                    n_home=("is_home", "sum"))
    bad = counts[(counts["n"] != 82) | (counts["n_home"] != 41)]
    if len(bad):
        raise AssertionError(f"games/home-away count violation: {bad.to_dict()}")
    return tg


def bootstrap_team_season(gf, ga, n_boot, rng, n_sample=N_SAMPLE):
    """Vectorized bootstrap for one team-season.

    gf, ga: arrays of clean goals for/against over the team's 82 games.
    Returns (gfpct_means, gf_sums, ga_sums): arrays of length n_boot.
    gfpct_means[i] = GF/(GF+GA) over the i-th drawn 40-game sample.
    """
    gf = np.asarray(gf, dtype=np.int64)
    ga = np.asarray(ga, dtype=np.int64)
    n = len(gf)
    # draw n_boot x n_sample indices with replacement
    idx = rng.integers(0, n, size=(n_boot, n_sample))
    gf_sums = gf[idx].sum(axis=1)
    ga_sums = ga[idx].sum(axis=1)
    tot = gf_sums + ga_sums
    return gf_sums / tot, gf_sums, ga_sums


def bootstrap_season(season, n_boot=DEFAULT_N_BOOT, seed_base=SEED_BASE):
    """Bootstrap all teams in one season.

    Returns (long_df, rep_summary_df):
      long_df: one row per (team, rep) with gf% — n_teams x n_boot rows.
      rep_summary_df: one row per rep with mean/SD of gf% across teams.
    """
    rng = np.random.default_rng(seed_base + int(season[:4]))
    tg = load_season(season)
    teams = sorted(tg["team"].unique())
    gf_pct = np.empty((len(teams), n_boot), dtype=np.float64)
    for t_i, team in enumerate(teams):
        sub = tg[tg["team"] == team]
        pcts, _, _ = bootstrap_team_season(sub["gf_clean"].to_numpy(),
                                           sub["ga_clean"].to_numpy(),
                                           n_boot, rng)
        gf_pct[t_i] = pcts
    n_teams = len(teams)
    long_df = pd.DataFrame({
        "season": season,
        "team": np.repeat(teams, n_boot),
        "rep": np.tile(np.arange(n_boot), n_teams),
        "gfpct": gf_pct.reshape(-1),
    })
    # per-rep moments across teams
    rep_summary = pd.DataFrame({
        "season": season,
        "rep": np.arange(n_boot),
        "mean": gf_pct.mean(axis=0),
        "sd": gf_pct.std(axis=0, ddof=1),
        "n_teams": n_teams,
    })
    return long_df, rep_summary


# ---------------------------------------------------------------- hand check
def hand_check():
    """Stage 3 verification: explicit non-vectorized computation on identical
    draws must reproduce the vectorized function exactly; plus a manual
    arithmetic check on one documented sample."""
    season = "20072008"
    team = "DET"
    tg = load_season(season)
    sub = tg[tg["team"] == team].sort_values("game_idx")
    gf = sub["gf_clean"].to_numpy()
    ga = sub["ga_clean"].to_numpy()
    assert len(gf) == 82

    rng = np.random.default_rng(12345)
    n_boot = 5
    vec_pct, vec_gf, vec_ga = bootstrap_team_season(gf, ga, n_boot, rng)

    # reproduce with the same generator state, explicitly, one draw at a time
    rng2 = np.random.default_rng(12345)
    man_pct, man_gf, man_ga = [], [], []
    for _ in range(n_boot):
        idx = rng2.integers(0, 82, size=40)   # same call signature/order
        g = int(gf[idx].sum())
        a = int(ga[idx].sum())
        man_gf.append(g)
        man_ga.append(a)
        man_pct.append(g / (g + a))
    man_pct = np.array(man_pct)

    lines = []
    lines.append(f"hand-check season={season} team={team} n_boot={n_boot}")
    lines.append(f"vectorized GF sums: {vec_gf.tolist()}")
    lines.append(f"manual     GF sums: {man_gf}")
    lines.append(f"vectorized GA sums: {vec_ga.tolist()}")
    lines.append(f"manual     GA sums: {man_ga}")
    lines.append(f"vectorized GF%: {np.round(vec_pct, 6).tolist()}")
    lines.append(f"manual     GF%: {np.round(man_pct, 6).tolist()}")
    assert np.array_equal(vec_gf, np.array(man_gf)), "GF sums differ"
    assert np.array_equal(vec_ga, np.array(man_ga)), "GA sums differ"
    assert np.allclose(vec_pct, man_pct, rtol=0, atol=1e-12), "GF% differ"

    # documented manual arithmetic example on draw 0
    idx0 = np.random.default_rng(12345).integers(0, 82, size=40)
    g0 = int(gf[idx0].sum())
    a0 = int(ga[idx0].sum())
    lines.append(f"draw 0 detail: gf={g0}, ga={a0}, gf%={g0/(g0+a0):.6f} "
                 f"= {g0}/({g0}+{a0})")
    lines.append(f"season clean-goal totals for {team}: GF={int(gf.sum())}, "
                 f"GA={int(ga.sum())}, "
                 f"season GF%={gf.sum()/(gf.sum()+ga.sum()):.6f}")
    lines.append("HAND CHECK PASSED")
    text = "\n".join(lines)
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "stage3_handcheck.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


def run_seasons(seasons, n_boot):
    os.makedirs(CLEAN, exist_ok=True)
    all_long = []
    all_sum = []
    for season in seasons:
        t0 = time.time()
        long_df, rep_sum = bootstrap_season(season, n_boot=n_boot)
        all_long.append(long_df)
        all_sum.append(rep_sum)
        pooled = rep_sum["mean"]  # not used; pooled stats come from long df
        print(f"{season}: {long_df['team'].nunique()} teams x {n_boot} reps "
              f"= {len(long_df)} rows ({time.time()-t0:.1f}s)", flush=True)
    long_out = pd.concat(all_long, ignore_index=True)
    sum_out = pd.concat(all_sum, ignore_index=True)
    long_out.to_parquet(OUT_BOOT, index=False)
    sum_out.to_parquet(OUT_SUM, index=False)
    # quick sanity summary
    g = sum_out.groupby("season").agg(mean_of_means=("mean", "mean"),
                                      mean_of_sd=("sd", "mean"),
                                      sd_of_sd=("sd", "std"))
    print(g.to_string())
    print(f"\nwrote {OUT_BOOT} ({len(long_out)} rows)")
    print(f"wrote {OUT_SUM} ({len(sum_out)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handcheck", action="store_true")
    ap.add_argument("--season")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    args = ap.parse_args()
    if args.handcheck:
        hand_check()
        return
    if args.all:
        run_seasons(ALL_SEASONS, args.n_boot)
    elif args.season:
        if args.season not in ALL_SEASONS:
            sys.exit(f"unknown season {args.season}")
        run_seasons([args.season], args.n_boot)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
