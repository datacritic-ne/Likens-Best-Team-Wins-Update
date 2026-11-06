#!/usr/bin/env python3
"""Stage 15 — Sample-size sensitivity of the bootstrap (user spec 2026-10-21).

Question: can bootstrap samples of 20 or 30 games per team (100,000 reps)
reproduce the talent-distribution estimates we get from the primary 40-game
bootstrap (Stage 3/4)? If yes, shortened seasons (2012-13, 2019-20, 2020-21)
become usable — a later, user-gated stage decides whether to ingest them.

Method:
- For each sample size n in {20, 30} and each of the 18 Stage 2 seasons:
  draw n games WITH replacement per team per rep (Stage 3 function,
  n_sample parameter), GF% from clean goals, 100,000 reps.
- Decompose each season's pooled GF% variance exactly as Stage 4:
  noise_var  = mean over teams of the within-team across-rep variance
               (sampling noise; talent fixed within a team-season)
  talent_var = pooled_var - noise_var
- Compare pooled SD / noise SD / implied talent SD against the 40-game
  Stage 4 table (outputs/stage4_fitted_table.csv).

Seeds: SEED_BASE15 + season-start-year, distinct from the Stage 3 base
(20261008) so the n=20/30 runs use fresh streams, documented and exact.

Verification (user-approved criterion):
- |Δ pooled implied talent SD| vs 40-game value < 0.003 for n=20 and n=30
  pooled over all 18 seasons;
- per-season Δ's same sign/rough magnitude (correlation and mean |Δ| reported
  — raw numbers printed regardless so the user can judge).
- Hand-check: one documented team-season re-verified with an explicit loop at
  n_sample=20 against the vectorized function (same draws, exact equality).
"""
from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bootstrap_gfpct import load_season, bootstrap_team_season

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT_TABLE = os.path.join(ROOT, "outputs", "stage15_samplesize_table.csv")
OUT_FIG = os.path.join(ROOT, "outputs", "stage15_fig_samplesize.png")
REPORT = os.path.join(ROOT, "outputs", "reports", "stage15_samplesize_report.txt")
STAGE4_TABLE = os.path.join(ROOT, "outputs", "stage4_fitted_table.csv")

SAMPLE_SIZES = [20, 30]
N_BOOT = 100_000
SEED_BASE15 = 20261021  # documented seed base for Stage 15 runs
SEASONS = ["20052006", "20062007", "20072008", "20082009", "20092010",
           "20102011", "20112012", "20132014", "20142015", "20152016",
           "20162017", "20172018", "20182019", "20212022", "20222023",
           "20232024", "20242025", "20252026"]


def hand_check_n20() -> list[str]:
    """Stage 3-style explicit-loop check at n_sample=20 (exact equality)."""
    season, team = "20072008", "DET"
    tg = load_season(season)
    sub = tg[tg["team"] == team].sort_values("game_idx")
    gf = sub["gf_clean"].to_numpy()
    ga = sub["ga_clean"].to_numpy()
    rng = np.random.default_rng(98765)
    n_boot = 5
    vec_pct, vec_gf, vec_ga = bootstrap_team_season(gf, ga, n_boot, rng, n_sample=20)
    rng2 = np.random.default_rng(98765)
    man_pct, man_gf, man_ga = [], [], []
    for _ in range(n_boot):
        idx = rng2.integers(0, 82, size=20)  # same call signature/order
        g = int(gf[idx].sum())
        a = int(ga[idx].sum())
        man_gf.append(g)
        man_ga.append(a)
        man_pct.append(g / (g + a))
    assert np.array_equal(vec_gf, np.array(man_gf)), "GF sums differ (n=20)"
    assert np.array_equal(vec_ga, np.array(man_ga)), "GA sums differ (n=20)"
    assert np.allclose(vec_pct, np.array(man_pct), rtol=0, atol=1e-12), "GF% differ"
    lines = [f"hand-check (n_sample=20) season={season} team={team} n_boot={n_boot}",
             f"vectorized GF sums: {vec_gf.tolist()}",
             f"manual     GF sums: {man_gf}",
             "HAND CHECK (n=20) PASSED"]
    return lines


def run_sample_size(n_sample: int) -> pd.DataFrame:
    """Bootstrap all 18 seasons at this sample size; per-season decomposition."""
    rows = []
    for season in SEASONS:
        t0 = time.time()
        rng = np.random.default_rng(SEED_BASE15 + int(season[:4]))
        tg = load_season(season)
        teams = sorted(tg["team"].unique())
        n_teams = len(teams)
        gf_pct = np.empty((n_teams, N_BOOT), dtype=np.float64)
        for t_i, team in enumerate(teams):
            s = tg[tg["team"] == team]
            pcts, _, _ = bootstrap_team_season(s["gf_clean"].to_numpy(),
                                               s["ga_clean"].to_numpy(),
                                               N_BOOT, rng, n_sample=n_sample)
            gf_pct[t_i] = pcts
        pooled_mean = float(gf_pct.mean())
        pooled_var = float(gf_pct.var(ddof=1))
        noise_var = float(gf_pct.var(axis=1, ddof=1).mean())
        talent_var = pooled_var - noise_var
        rows.append({
            "season": season, "n_sample": n_sample, "n_teams": n_teams,
            "pooled_mean": pooled_mean, "pooled_sd": np.sqrt(pooled_var),
            "noise_sd": np.sqrt(noise_var), "talent_sd": np.sqrt(talent_var),
        })
        print(f"  n={n_sample} {season}: pooled_sd={np.sqrt(pooled_var):.5f} "
              f"noise_sd={np.sqrt(noise_var):.5f} "
              f"talent_sd={np.sqrt(talent_var):.5f} ({time.time()-t0:.1f}s)",
              flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    lines = ["Stage 15 — sample-size sensitivity of the bootstrap",
             f"(reps per team-season: {N_BOOT:,}; sample sizes {SAMPLE_SIZES} "
             f"vs the primary 40-game Stage 3/4 run)",
             ""]

    hc = hand_check_n20()
    lines += hc + [""]

    frames = []
    for n in SAMPLE_SIZES:
        print(f"n_sample={n}")
        frames.append(run_sample_size(n))
    res = pd.concat(frames, ignore_index=True)

    base = pd.read_csv(STAGE4_TABLE, dtype={"season": str})  # 40-game primary results
    res["season"] = res["season"].astype(str)
    base = base.rename(columns={"pooled_sd": "pooled_sd_40", "noise_sd": "noise_sd_40",
                                "talent_sd": "talent_sd_40"})
    res = res.merge(base[["season", "talent_sd_40", "noise_sd_40", "pooled_sd_40"]],
                    on="season", how="left")
    res["talent_sd_diff"] = res["talent_sd"] - res["talent_sd_40"]
    res["noise_sd_diff"] = res["noise_sd"] - res["noise_sd_40"]
    res["pooled_sd_diff"] = res["pooled_sd"] - res["pooled_sd_40"]
    res.to_csv(OUT_TABLE, index=False)

    # --- comparison summary per sample size -------------------------------
    lines.append("Per-sample-size comparison vs the 40-game primary")
    lines.append("(mean over 18 seasons; diffs = n-game value minus 40-game value)")
    lines.append("")
    verdict = {}
    for n in SAMPLE_SIZES:
        s = res[res["n_sample"] == n]
        d_talent_pooled = float(s["talent_sd"].mean() - s["talent_sd_40"].mean())
        mean_abs = float(s["talent_sd_diff"].abs().mean())
        max_abs = float(s["talent_sd_diff"].abs().max())
        corr = float(np.corrcoef(s["talent_sd"], s["talent_sd_40"])[0, 1])
        lines.append(f"n={n}: pooled talent_sd {s['talent_sd'].mean():.5f} "
                     f"vs 40-game {s['talent_sd_40'].mean():.5f} "
                     f"(Δ {d_talent_pooled:+.5f}) | mean |Δ| per season {mean_abs:.5f} "
                     f"max |Δ| {max_abs:.5f} | corr(per-season) {corr:.3f}")
        lines.append(f"      noise_sd  {s['noise_sd'].mean():.5f} vs "
                     f"{s['noise_sd_40'].mean():.5f} "
                     f"(Δ {float(s['noise_sd'].mean() - s['noise_sd_40'].mean()):+.5f})  "
                     f"[theory: noise SD ∝ 1/sqrt(n): 40-game {s['noise_sd_40'].mean():.4f} "
                     f"-> expected n={n} ~{s['noise_sd_40'].mean() * np.sqrt(40 / n):.4f}]")
        lines.append(f"      pooled_sd {s['pooled_sd'].mean():.5f} vs "
                     f"{s['pooled_sd_40'].mean():.5f} "
                     f"(Δ {float(s['pooled_sd'].mean() - s['pooled_sd_40'].mean()):+.5f})")
        verdict[n] = abs(d_talent_pooled) < 0.003
        lines.append("")

    lines.append("USER-APPROVED COMPARABILITY CRITERION: |Δ pooled implied talent SD| < 0.003")
    for n in SAMPLE_SIZES:
        lines.append(f"  n={n}: {'COMPARABLE' if verdict[n] else 'NOT COMPARABLE'}")
    lines.append("")
    lines.append("Per-season detail (talent_sd, n-game vs 40-game):")
    lines.append("")
    show = res[["season", "n_sample", "talent_sd", "talent_sd_40",
                "talent_sd_diff"]].copy()
    lines.append(show.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    lines.append("")
    lines.append("Decision gate (user's): whether these results justify ingesting")
    lines.append("the shortened seasons (2012-13 and/or 2019-20; 2020-21 flagged")
    lines.append("problematic by the user — realigned single-division schedule) and")
    lines.append("revisiting the primary analysis. Recorded in the planning doc; not")
    lines.append("decided here.")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # --- figure: implied talent SD per season, 20/30/40 ---------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    label = {20: "20-game", 30: "30-game"}
    for n in SAMPLE_SIZES:
        s = res[res["n_sample"] == n]
        ax.plot(s["season"], s["talent_sd"], "o--", label=label[n], alpha=0.8)
    ax.plot(res["season"], res["talent_sd_40"], "s-", label="40-game (primary)",
            color="black", lw=2)
    ax.set_ylabel("implied talent SD (GF%)")
    ax.set_title(f"Stage 15: implied talent SD vs bootstrap sample size "
                 f"({N_BOOT:,} reps/season)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"\nwrote {OUT_TABLE}")
    print(f"wrote {REPORT}")
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
