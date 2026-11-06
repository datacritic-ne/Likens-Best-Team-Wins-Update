#!/usr/bin/env python3
"""Stage 16 — Parity over time (user spec 2026-10-21).

From the EXISTING 40-game, 100k-rep bootstrap results (no new compute):
3-year and 5-year TRAILING moving averages of the per-season bootstrap SD
across the 18 Stage 2 seasons (2005-06 -> 2025-26). The per-season mean GF%
is computed too as the 0.5 sanity check (reported, not featured).

Inputs:  outputs/stage4_fitted_table.csv (per-season pooled_mean/pooled_sd)
Outputs: outputs/stage16_parity_trend.csv  (season, pooled_sd, ma3, ma5)
         outputs/stage16_fig_parity.png
         outputs/reports/stage16_parity_report.txt

Verification: one MA window recomputed by hand (explicit sum) must match the
rolling result exactly; trailing windows leave NaN until the window fills.
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
OUT_CSV = os.path.join(ROOT, "outputs", "stage16_parity_trend.csv")
OUT_FIG = os.path.join(ROOT, "outputs", "stage16_fig_parity.png")
REPORT = os.path.join(ROOT, "outputs", "reports", "stage16_parity_report.txt")


def trailing_ma(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average; NaN until `window` observations exist."""
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=window).mean().to_numpy()


def main() -> None:
    t = pd.read_csv(STAGE4, dtype={"season": str})
    t = t.sort_values("season").reset_index(drop=True)
    assert len(t) == 18, f"expected 18 seasons, got {len(t)}"

    t["ma3"] = trailing_ma(t["pooled_sd"].to_numpy(), 3)
    t["ma5"] = trailing_ma(t["pooled_sd"].to_numpy(), 5)

    # --- verification: hand-recompute two windows -------------------------
    sd = t["pooled_sd"].to_numpy()
    hand3 = float(np.mean(sd[2:5]))          # seasons index 2,3,4 -> 5th row
    hand5 = float(np.mean(sd[4:9]))          # seasons index 4..8 -> 9th row
    assert abs(hand3 - t["ma3"].iloc[4]) < 1e-12, "MA3 hand-check failed"
    assert abs(hand5 - t["ma5"].iloc[8]) < 1e-12, "MA5 hand-check failed"
    assert t["ma3"].isna().sum() == 2 and t["ma5"].isna().sum() == 4, \
        "trailing-NaN handling wrong"
    mean_check = (t["pooled_mean"] - 0.5).abs().max()
    print(f"hand-checks PASSED (MA3 window rows 3-5, MA5 window rows 5-9; "
          f"trailing NaNs: ma3={2}, ma5={4})")
    print(f"sanity: max |pooled_mean - 0.5| across seasons = {mean_check:.4f}")

    t.to_csv(OUT_CSV, index=False)

    lines = ["Stage 16 — parity over time (trailing moving averages of the "
             "bootstrapped GF% SD)",
             "(40-game samples, 100,000 reps per season, Stage 3/4 bootstrap)",
             ""]
    show = t[["season", "pooled_mean", "pooled_sd", "ma3", "ma5"]]
    lines.append(show.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    lines.append("")
    lines.append(f"Sanity: pooled means range "
                 f"{t['pooled_mean'].min():.4f}..{t['pooled_mean'].max():.4f} "
                 f"(≈0.5, as required — one team's GF is another's GA).")
    lo3, hi3 = t["ma3"].min(), t["ma3"].max()
    lo5, hi5 = t["ma5"].min(), t["ma5"].max()
    lines.append(f"MA3 range: {lo3:.5f} .. {hi3:.5f}   "
                 f"MA5 range: {lo5:.5f} .. {hi5:.5f}")
    first_valid3, first_valid5 = t["ma3"].dropna().iloc[0], t["ma5"].dropna().iloc[0]
    last3, last5 = t["ma3"].iloc[-1], t["ma5"].iloc[-1]
    lines.append(f"MA3 first-valid {first_valid3:.5f} -> last {last3:.5f} "
                 f"(Δ {last3 - first_valid3:+.5f})")
    lines.append(f"MA5 first-valid {first_valid5:.5f} -> last {last5:.5f} "
                 f"(Δ {last5 - first_valid5:+.5f})")
    lines.append("")
    lines.append("Reading (for the writeup; interpret with care — 18 points,")
    lines.append("overlapping windows, no significance machinery): a falling MA")
    lines.append("means the league's bootstrapped GF% spread narrowed (MORE")
    lines.append("parity); a rising MA means it widened (LESS parity).")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # --- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(t))
    ax.plot(x, sd, "o:", color="grey", alpha=0.6, label="per-season SD (40-game bootstrap)")
    ax.plot(x, t["ma3"], "-", lw=2, label="3-year trailing MA")
    ax.plot(x, t["ma5"], "-", lw=2, label="5-year trailing MA")
    ax.set_xticks(x)
    ax.set_xticklabels(t["season"], rotation=45)
    ax.set_ylabel("bootstrapped GF% SD")
    ax.set_title("Stage 16: parity over time — bootstrapped GF% spread, 2005-06 → 2025-26")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {REPORT}")
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
