#!/usr/bin/env python3
"""Stage 14 figure (writeup support) — playoff outcomes by talent rank.

One row per true-talent rank 1..32 (rank 1 = best drawn GF% in each simulated
season), aggregated over the 100,000-season Stage 13 run:

  rank | P(playoffs) | P(win round 1) | P(round 2) | P(round 3) | P(Cup)

"Win round k" = reached the conference final / Cup final / won the Cup
(cumulative, matching the round{k}_wins >= k flags in the Stage 13 output).

Inputs:  outputs/stage13_full_100k.parquet
Output:  outputs/stage14_table_playoffs.png + .csv
Checks:  one team per rank per season; per-season sums of the round flags
         equal {8, 4, 2, 1}; P(Cup) sums to 1.0 across ranks.
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
OUT = os.path.join(ROOT, "outputs", "stage14_table_playoffs.png")


def main() -> None:
    p = pd.read_parquet(PARQUET)

    # --- checks -------------------------------------------------------------
    n_seasons = p["season"].nunique()
    assert n_seasons == 100_000, n_seasons
    rank = (p.groupby("season")["talent_gfpct"]
             .rank(ascending=False, method="first").astype(int))
    assert rank.min() == 1 and rank.max() == 32
    assert (rank.value_counts() == n_seasons).all()

    p["talent_rank"] = rank
    # per-season structural gates: 8 R1 winners, 4 R2, 2 R3 (conf-final wins
    # = Cup finalists), 1 Cup. Cup final is the separate cup_winner flag
    # (cup_winner is NOT a subset of round3_wins by construction).
    per_season = p.groupby("season")[["round1_wins", "round2_wins",
                                      "round3_wins", "cup_winner"]].sum()
    assert (per_season == [8, 4, 2, 1]).all().all()
    # cumulative consistency: round2 <= round1, round3 <= round2 per team-season;
    # cup_winner must be a playoff team with rounds_won >= 1
    assert (p["round2_wins"] <= p["round1_wins"]).all()
    assert (p["round3_wins"] <= p["round2_wins"]).all()
    assert ((p["cup_winner"] == 0) |
            ((p["cup_winner"] == 1) & (p["made_playoffs"] == 1))).all()

    g = p.groupby("talent_rank")
    tab = pd.DataFrame({
        "playoffs": g["made_playoffs"].mean(),
    })
    # direct cumulative probabilities: share of team-seasons with rounds_won >= k
    # round flags are binary (1 = won that round), so cumulative probability
    # of reaching round k+1 = share with flag == 1
    tab["r1"] = g["round1_wins"].mean()
    tab["r2"] = g["round2_wins"].mean()
    tab["r3"] = g["round3_wins"].mean()
    tab["cup"] = g["cup_winner"].mean()

    assert abs(tab["cup"].sum() - 1.0) < 1e-6
    assert abs(tab["playoffs"].sum() - 16.0) < 1e-6
    assert (tab["r2"] <= tab["r1"]).all() and (tab["r3"] <= tab["r2"]).all()
    assert (tab["cup"] <= tab["playoffs"]).all()
    print(tab.head(3).to_string(float_format=lambda x: f"{x:.3f}"))
    print(tab.tail(3).to_string(float_format=lambda x: f"{x:.3f}"))

    # --- format -------------------------------------------------------------
    disp = pd.DataFrame(index=tab.index)
    disp["Playoffs"] = tab["playoffs"].map("{:.1%}".format)
    disp["Win Round 1"] = tab["r1"].map("{:.1%}".format)
    disp["Win Round 2"] = tab["r2"].map("{:.1%}".format)
    disp["Win Round 3"] = tab["r3"].map("{:.1%}".format)
    disp["Win Cup"] = tab["cup"].map("{:.1%}".format)

    row_h = 0.30
    fig_h = 1.9 + row_h * (len(disp) + 1)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=disp.values,
        rowLabels=[str(i) for i in disp.index],
        colLabels=list(disp.columns),
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
            cell.set_height(cell.get_height() * 1.25)
        elif r % 2 == 0:
            cell.set_facecolor(band)

    ax.set_title("Simulated Playoff Outcomes by True-Talent Rank\n"
                 f"({n_seasons:,} simulated seasons; rank 1 = best drawn GF%)",
                 fontsize=15, fontweight="bold", pad=24)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")

    csv_out = OUT.replace(".png", ".csv")
    tab.round(5).to_csv(csv_out)
    print(f"wrote {csv_out}")


if __name__ == "__main__":
    main()
