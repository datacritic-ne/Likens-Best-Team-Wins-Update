#!/usr/bin/env python3
"""Stage 16b — Parity over time in TRUE-TALENT terms (user request 2026-10-27).

Companion to stage16_parity_trend.py. Stage 16 smoothed the per-season
bootstrapped GF% SD, which mixes true-talent variation with 40-game sampling
noise. Here we instead smooth the per-season IMPLIED TALENT SD from the
Stage 4 decomposition (talent ⊕ noise; variance adds), which is the parity
measure aligned with the simulation's talent-SD parameter.

No new compute: reads outputs/stage4_fitted_table.csv only.

Inputs:  outputs/stage4_fitted_table.csv (per-season talent_sd)
Outputs: outputs/stage16b_parity_trend_talent.csv (season, talent_sd, ma3, ma5)
         outputs/stage16b_fig_parity_talent.png
         outputs/reports/stage16b_parity_talent_report.txt

Verification: two MA windows recomputed by hand must match exactly; trailing
windows leave NaN until filled; sanity check that sqrt(pooled_sd^2 - noise^2)
reproduces the talent_sd column from the decomposition identity.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
STAGE4 = os.path.join(ROOT, "outputs", "stage4_fitted_table.csv")
OUT_CSV = os.path.join(ROOT, "outputs", "stage16b_parity_trend_talent.csv")
OUT_FIG = os.path.join(ROOT, "outputs", "stage16b_fig_parity_talent.png")
REPORT = os.path.join(ROOT, "outputs", "reports", "stage16b_parity_talent_report.txt")


def trailing_ma(values: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=window).mean().to_numpy()


def main() -> None:
    t = pd.read_csv(STAGE4, dtype={"season": str})
    t = t.sort_values("season").reset_index(drop=True)
    assert len(t) == 18, f"expected 18 seasons, got {len(t)}"

    # Sanity: the talent column must reproduce the decomposition identity
    # sqrt(pooled_sd^2 - noise_sd^2) to within rounding of the CSV values.
    ident = np.sqrt(t["pooled_sd"].to_numpy() ** 2 - t["noise_sd"].to_numpy() ** 2)
    ident_err = float(np.max(np.abs(ident - t["talent_sd"].to_numpy())))
    assert ident_err < 1e-3, f"decomposition identity mismatch: {ident_err}"

    t["ma3"] = trailing_ma(t["talent_sd"].to_numpy(), 3)
    t["ma5"] = trailing_ma(t["talent_sd"].to_numpy(), 5)

    # --- verification: hand-recompute two windows -------------------------
    sd = t["talent_sd"].to_numpy()
    hand3 = float(np.mean(sd[2:5]))          # window ending at index 4
    hand5 = float(np.mean(sd[4:9]))          # window ending at index 8
    assert abs(hand3 - t["ma3"].iloc[4]) < 1e-12, "MA3 hand-check failed"
    assert abs(hand5 - t["ma5"].iloc[8]) < 1e-12, "MA5 hand-check failed"
    assert t["ma3"].isna().sum() == 2 and t["ma5"].isna().sum() == 4, \
        "trailing-NaN handling wrong"
    print(f"hand-checks PASSED (MA3 window rows 3-5, MA5 window rows 5-9; "
          f"trailing NaNs: ma3=2, ma5=4)")
    print(f"sanity: max |sqrt(pooled_sd^2-noise_sd^2) - talent_sd| = {ident_err:.2e}")

    t.to_csv(OUT_CSV, index=False)

    mean_talent = float(np.mean(sd))
    lines = ["Stage 16b — parity over time in TRUE-TALENT terms",
             "(implied talent SD per Stage 4 decomposition; same bootstrap:",
             " 40-game samples, 100,000 reps per season)",
             ""]
    show = t[["season", "pooled_sd", "noise_sd", "talent_sd", "ma3", "ma5"]]
    lines.append(show.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    lines.append("")
    lines.append(f"Mean implied talent SD across 18 seasons: {mean_talent:.4f} "
                 f"(locked constant 0.0415)")
    lo3, hi3 = t["ma3"].min(), t["ma3"].max()
    lo5, hi5 = t["ma5"].min(), t["ma5"].max()
    lines.append(f"Talent MA3 range: {lo3:.5f} .. {hi3:.5f}   "
                 f"Talent MA5 range: {lo5:.5f} .. {hi5:.5f}")
    first_valid3, first_valid5 = t["ma3"].dropna().iloc[0], t["ma5"].dropna().iloc[0]
    last3, last5 = t["ma3"].iloc[-1], t["ma5"].iloc[-1]
    lines.append(f"Talent MA3 first-valid {first_valid3:.5f} -> last {last3:.5f} "
                 f"(Δ {last3 - first_valid3:+.5f})")
    lines.append(f"Talent MA5 first-valid {first_valid5:.5f} -> last {last5:.5f} "
                 f"(Δ {last5 - first_valid5:+.5f})")
    lines.append("")
    lines.append("Reading: unlike Stage 16's total-SD series, this measures")
    lines.append("true-talent spread directly (sampling noise removed via the")
    lines.append("talent ⊕ noise decomposition). Interpret with care — 18")
    lines.append("points, overlapping windows, descriptive only.")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # --- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(t))
    ax.plot(x, sd, "o:", color="grey", alpha=0.6,
            label="per-season implied talent SD (Stage 4)")
    ax.plot(x, t["ma3"], "-", lw=2, label="3-year trailing MA")
    ax.plot(x, t["ma5"], "-", lw=2, label="5-year trailing MA")
    ax.axhline(0.0415, color="red", ls="--", lw=1,
               label="locked talent SD 0.0415 (sim input)")
    ax.set_xticks(x)
    ax.set_xticklabels(t["season"], rotation=45)
    ax.set_ylabel("implied talent SD (GF%)")
    ax.set_title("Stage 16b: parity over time — true-talent SD, 2005-06 → 2025-26")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {REPORT}")
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
