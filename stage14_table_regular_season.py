#!/usr/bin/env python3
"""Stage 14 figure (writeup support) — regular-season outcomes by talent rank.

One row per true-talent rank 1..32 (rank 1 = best drawn GF% in each simulated
season), aggregated over the 100,000-season Stage 13 run:

  rank | mean drawn GF% | GF/season | GA/season | PTS/season |
  P(playoffs) | P(division) | P(conference best) | P(Presidents' Trophy)

Rendered as a clean PNG table graphic for the Substack post.

Inputs:  outputs/stage13_full_100k.parquet
Output:  outputs/stage14_table_regular_season.png
Checks:  each season contributes exactly one team per rank (1..32);
         P(playoffs) ranks 1..16 sum appropriately (16 teams/season across
         32 ranks); flag columns are 0/1.
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
PARQUET = os.path.join(ROOT, "outputs", "stage13_full_100k.parquet")
OUT = os.path.join(ROOT, "outputs", "stage14_table_regular_season.png")

sns_ok = True
try:
    import seaborn as sns
    sns.set_theme(style="white", context="talk")
except ImportError:  # pragma: no cover
    sns_ok = False


def main() -> None:
    p = pd.read_parquet(PARQUET)

    # --- checks -------------------------------------------------------------
    n_seasons = p["season"].nunique()
    assert n_seasons == 100_000, n_seasons
    assert set(p["made_playoffs"].unique()) <= {0, 1}
    # every season ranks every team 1..32 exactly once
    rank = (p.groupby("season")["talent_gfpct"]
             .rank(ascending=False, method="first").astype(int))
    assert rank.min() == 1 and rank.max() == 32
    assert (rank.value_counts() == n_seasons).all()

    p["talent_rank"] = rank
    g = p.groupby("talent_rank")
    tab = pd.DataFrame({
        "gfpct": g["talent_gfpct"].mean(),
        "gf": g["GF"].mean(),
        "ga": g["GA"].mean(),
        "pts": g["PTS"].mean(),
        "playoffs": g["made_playoffs"].mean(),
        "division": g["division_winner"].mean(),
        "conference": g["conference_best"].mean(),
        "trophy": g["presidents_trophy"].mean(),
    })
    # structural sanity: 16 playoff teams per season spread over 32 ranks
    assert abs(tab["playoffs"].sum() - 16.0) < 1e-6
    assert abs(tab["trophy"].sum() - 1.0) < 1e-6
    assert tab["gfpct"].is_monotonic_decreasing
    print(tab.head(3).to_string(float_format=lambda x: f"{x:.3f}"))
    print(tab.tail(3).to_string(float_format=lambda x: f"{x:.3f}"))

    # --- format -------------------------------------------------------------
    disp = pd.DataFrame(index=tab.index)
    disp["GF%"] = tab["gfpct"].map("{:.3f}".format)
    disp["GF"] = tab["gf"].round(0).astype(int).astype(str)
    disp["GA"] = tab["ga"].round(0).astype(int).astype(str)
    disp["PTS"] = tab["pts"].round(0).astype(int).astype(str)
    disp["Playoffs"] = tab["playoffs"].map("{:.1%}".format)
    disp["Division"] = tab["division"].map("{:.1%}".format)
    disp["Conference"] = tab["conference"].map("{:.1%}".format)
    disp["Presidents' Trophy"] = tab["trophy"].map("{:.1%}".format)

    n_rows = len(disp)
    row_h = 0.30
    fig_h = 1.9 + row_h * (n_rows + 1)
    fig, ax = plt.subplots(figsize=(12.3, fig_h))
    ax.axis("off")

    col_labels = ["GF%", "GF", "GA", "PTS", "Playoffs", "Division",
                  "Conference", "Presidents' Trophy"]
    tbl = ax.table(
        cellText=disp.values,
        rowLabels=[str(i) for i in disp.index],
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.35)

    header_color = "#2c3e50"
    band = "#f2f4f7"
    edge = "#d0d5dd"
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(edge)
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_facecolor(header_color)
            cell.get_text().set_fontsize(9)
            cell.set_height(cell.get_height() * 1.25)
        elif r % 2 == 0:
            cell.set_facecolor(band)

    ax.set_title("Simulated Regular-Season Outcomes by True-Talent Rank\n"
                 f"({n_seasons:,} simulated seasons; rank 1 = best drawn GF%)",
                 fontsize=15, fontweight="bold", pad=24)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")

    # CSV twin for traceability (rule 18)
    csv_out = OUT.replace(".png", ".csv")
    tab.round(5).to_csv(csv_out)
    print(f"wrote {csv_out}")


if __name__ == "__main__":
    main()
