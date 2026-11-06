#!/usr/bin/env python3
"""Stage 14 figure (writeup support) — bootstrapped GF% distribution vs
theoretical talent distributions with sampling noise added back.

Three curves on one axes:
  1. Observed: pooled 40-game bootstrapped GF% distribution (18 seasons,
     100k reps each) — drawn as a KDE line, not a histogram.
  2. Ours: normal(0.5, sqrt(talent 0.0415^2 + noise^2)).
  3. JLikens: normal(0.5, sqrt(talent 0.03^2 + noise^2)).

Noise SD = mean per-season bootstrap sampling-noise SD from the Stage 4
decomposition, so all three curves describe the same observable quantity
(a 40-game GF% spread) and are visually comparable.

Inputs:  data/clean/bootstrap_long.parquet (gfpct column)
         outputs/stage4_fitted_table.csv (noise_sd per season)
Output:  outputs/stage14_fig_gfpct_distribution.png

Verification: pooled bootstrapped mean/SD printed and must match the Stage 4
pooled values (mean ~0.5, SD ~0.052); the two normals integrate to 1 by
construction (scipy.pdf normalized); curve SDs reported in the caption box.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
BOOT = os.path.join(ROOT, "data", "clean", "bootstrap_long.parquet")
STAGE4 = os.path.join(ROOT, "outputs", "stage4_fitted_table.csv")
OUT = os.path.join(ROOT, "outputs", "stage14_fig_gfpct_distribution.png")

TALENT_OURS = 0.0415
TALENT_JL = 0.03

sns.set_theme(style="white", context="talk")


def main() -> None:
    boot = pd.read_parquet(BOOT, columns=["season", "gfpct"])
    # Subsample for the KDE: the full 55.2M rows exceed memory; take a fixed
    # 20,000 rows per season (seeded, so the figure is exactly reproducible).
    # Subsample moments match the full pooled moments to ~1e-3.
    rng = np.random.default_rng(20261027)
    samp = (boot.groupby("season", group_keys=False)
                .apply(lambda g: g.sample(n=min(20_000, len(g)),
                                          random_state=20261027)))
    vals = samp["gfpct"].to_numpy()
    del boot
    full = pd.read_parquet(BOOT, columns=["gfpct"])
    pooled_mean_full, pooled_sd_full = float(np.mean(full["gfpct"])), float(np.std(full["gfpct"], ddof=0))
    del full
    pooled_mean, pooled_sd = float(np.mean(vals)), float(np.std(vals, ddof=0))
    print(f"full pooled:       mean {pooled_mean_full:.5f}, SD {pooled_sd_full:.5f}")
    print(f"subsample (200k):  mean {pooled_mean:.5f}, SD {pooled_sd:.5f}")

    s4 = pd.read_csv(STAGE4, dtype={"season": str})
    noise = float(s4["noise_sd"].mean())

    # --- verification against Stage 4 pooled values ------------------------
    assert abs(pooled_mean_full - 0.5) < 0.005, pooled_mean_full
    assert abs(pooled_sd_full - s4["pooled_sd"].mean()) < 0.005, pooled_sd_full
    assert abs(pooled_sd - pooled_sd_full) < 0.002, (pooled_sd, pooled_sd_full)
    print(f"pooled bootstrap: mean {pooled_mean:.4f}, SD {pooled_sd:.4f} "
          f"(n={len(vals):,})")
    print(f"noise SD (mean of Stage 4 per-season noise): {noise:.4f}")

    sd_ours = float(np.sqrt(TALENT_OURS**2 + noise**2))
    sd_jl = float(np.sqrt(TALENT_JL**2 + noise**2))
    print(f"comparable SDs: ours {sd_ours:.4f}  JLikens {sd_jl:.4f}")

    x = np.linspace(0.30, 0.70, 600)
    norm_ours = np.exp(-0.5 * ((x - 0.5) / sd_ours) ** 2) / (sd_ours * np.sqrt(2 * np.pi))
    norm_jl = np.exp(-0.5 * ((x - 0.5) / sd_jl) ** 2) / (sd_jl * np.sqrt(2 * np.pi))

    fig, ax = plt.subplots(figsize=(11, 6.5))

    sns.kdeplot(x=vals, ax=ax, color="#333333", lw=2.5, label="Bootstrapped GF% (18 seasons)")

    ax.plot(x, norm_ours, color="#1f77b4", lw=2.5,
            label="Updated: talent SD 0.0415")
    ax.plot(x, norm_jl, color="#d62728", lw=2.5,
            label="JLikens: talent SD 0.03")

    ax.set_xlabel("GF% (40-game samples)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Bootstrapped GF% and Theoretical Team Ability Distributions\n"
                 "(theoretical curves include 40-game sampling noise, SD ≈ "
                 f"{noise:.3f}, for comparability)",
                 pad=52)
    ax.set_xlim(0.30, 0.70)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncols=3, frameon=False, fontsize=13,
              columnspacing=1.4, handlelength=1.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
