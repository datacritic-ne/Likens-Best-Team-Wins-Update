"""Stage 4 — bootstrap analyses (JLikens replication & refinement).

Processes one season at a time (memory-bounded; the 55.2M-row long table
is read per-season via pyarrow filters).

Inputs:  data/clean/bootstrap_long.parquet (Stage 3: season, team, rep, gfpct)
Outputs: outputs/stage4_fitted_table.csv        — per-season fitted moments
         outputs/stage4_jlikens_replication.csv — 2,000-rep JLikens-setup view
         outputs/reports/stage4_report.txt      — full report

Decomposition (documented in report):
  The pooled bootstrap GF% distribution across teams mixes
    (a) between-team true-talent variance  sigma_t^2, and
    (b) 40-game sampling noise  sigma_n^2.
  For each team-season, the variance of its gf% ACROSS bootstrap reps
  estimates its own sampling noise (talent fixed within team-season;
  rep-to-rep variation is purely which 40 games were drawn). Averaging
  over teams gives sigma_n^2; then sigma_t = sqrt(pooled_var - sigma_n^2).
  JLikens's published SD = 0.03 is smaller than sigma_n alone, so it can
  only be a talent-like parameter, not the full bootstrap spread.
"""
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
LONG = os.path.join(ROOT, "data", "clean", "bootstrap_long.parquet")
OUT_TABLE = os.path.join(ROOT, "outputs", "stage4_fitted_table.csv")
OUT_REPL = os.path.join(ROOT, "outputs", "stage4_jlikens_replication.csv")
REPORT = os.path.join(ROOT, "outputs", "reports", "stage4_report.txt")

JLIKENS_SEASONS = ["20052006", "20062007", "20072008", "20082009",
                   "20092010", "20102011"]


def season_table(season, n_reps_repl=2000):
    df = pq.read_table(LONG, filters=[("season", "=", season)]).to_pandas()
    g = df["gfpct"]
    pooled_mean = g.mean()
    pooled_var = g.var(ddof=1)
    # within-team across-rep variance = sampling noise
    noise_var = df.groupby("team")["gfpct"].var(ddof=1).mean()
    talent_var = pooled_var - noise_var
    # per-rep pooled SD (distribution of the SD estimate)
    rep_sd = df.groupby("rep")["gfpct"].std(ddof=1)
    rep_mean = df.groupby("rep")["gfpct"].mean()
    # JLikens-setup view: first 2,000 reps (same seeded stream)
    sub = df[df["rep"] < n_reps_repl]
    sub_rep_sd = sub.groupby("rep")["gfpct"].std(ddof=1)
    sub_rep_mean = sub.groupby("rep")["gfpct"].mean()
    del df, sub
    return {
        "season": season,
        "n_teams": df["team"].nunique() if False else int(g.size / (rep_sd.size)),
        "pooled_mean": pooled_mean,
        "pooled_sd": np.sqrt(pooled_var),
        "noise_sd": np.sqrt(noise_var),
        "talent_sd": np.sqrt(talent_var),
        "mean_of_rep_sd": rep_sd.mean(),
        "sd_of_rep_sd": rep_sd.std(ddof=1),
        "repl_mean_of_means": sub_rep_mean.mean(),
        "repl_sd_of_means": sub_rep_mean.std(ddof=1),
        "repl_mean_of_sd": sub_rep_sd.mean(),
        "repl_sd_of_sd": sub_rep_sd.std(ddof=1),
    }


def main():
    pf = pq.ParquetFile(LONG)
    seasons = sorted(pd.read_parquet(LONG, columns=["season"])["season"].unique())
    rows = []
    for s in seasons:
        r = season_table(s)
        rows.append(r)
        print(f"{s}: pooled_sd={r['pooled_sd']:.5f} noise_sd={r['noise_sd']:.5f} "
              f"talent_sd={r['talent_sd']:.5f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_TABLE, index=False)

    lines = []
    lines.append("Stage 4 — fitted bootstrap GF% distribution per season")
    lines.append("(100,000 reps; GF% from clean goals, 40-game samples w/ replacement)")
    lines.append("")
    show = table[["season", "n_teams", "pooled_mean", "pooled_sd",
                  "noise_sd", "talent_sd", "sd_of_rep_sd"]]
    lines.append(show.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    lines.append("")
    lines.append("Definitions:")
    lines.append("  pooled_mean/pooled_sd : mean/SD of all bootstrapped GF% (teams+reps)")
    lines.append("  noise_sd              : mean within-team SD across reps (40-game sampling noise)")
    lines.append("  talent_sd             : sqrt(pooled_sd^2 - noise_sd^2) — implied true-talent spread")
    lines.append("  sd_of_rep_sd          : SD across reps of the per-rep pooled SD (MC error of SD)")
    lines.append("")
    lines.append(f"Grand averages: pooled_sd={table['pooled_sd'].mean():.5f}  "
                 f"noise_sd={table['noise_sd'].mean():.5f}  "
                 f"talent_sd={table['talent_sd'].mean():.5f}")
    lines.append("")
    six = table[table["season"].isin(JLIKENS_SEASONS)]
    lines.append("JLikens-setup replication (first 2,000 reps, his 6 seasons):")
    lines.append("")
    lines.append(six[["season", "repl_mean_of_means", "repl_sd_of_means",
                      "repl_mean_of_sd", "repl_sd_of_sd"]]
                 .to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    lines.append("")
    lines.append(f"JLikens 6-season average implied talent_sd: "
                 f"{six['talent_sd'].mean():.5f}")
    lines.append("JLikens published: mean 0.5, best-fit normal SD 0.03.")
    lines.append("Our full bootstrap spread (~0.05) exceeds 0.03; 40-game noise")
    lines.append("alone is ~0.034. Under the decomposition, 0.03 is consistent")
    lines.append("only with a TALENT-spread parameter; his half-season validation")
    lines.append("would have calibrated sim-from-talent (0.03) + sample noise to")
    lines.append("the observed bootstrapped spread.")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    repl_cols = ["season", "repl_mean_of_means", "repl_sd_of_means",
                 "repl_mean_of_sd", "repl_sd_of_sd"]
    table[table["season"].isin(JLIKENS_SEASONS)][repl_cols].to_csv(OUT_REPL, index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
