#!/usr/bin/env python3
"""Stage 17 — Parameter sensitivity of the headline result (user spec 2026-10-21).

Deterministic univariate sensitivity of P(best team wins the Cup) to the
three locked simulation inputs:
  - true-talent SD     (default 0.0415)
  - total clean GPG    (default 5.4933)
  - home-ice advantage (default 0.2563 goals/game)
The talent-distribution MEAN is never varied (must be 0.5: one team's GF is
another's GA). For each grid value, rerun the full 100,000-season simulation
with that parameter overridden and the other two at defaults; report the
best-team (highest drawn talent) Cup/playoffs/PT rates per value.

Grid (user-approved):
  talent SD {0.030, 0.035, 0.040, 0.0415*, 0.045, 0.050, 0.055}
  GPG       {5.0, 5.4933*, 6.0, 6.5}
  HIA       {0.0, 0.10, 0.2563*, 0.40}
* = default (reused from the primary Stage 13 run where the seed matches —
the default run here uses seed 20261028, the Stage 13 full-run seed, so its
results ARE the primary results and double as the reproduction check).

Verification:
  - default-value run reproduces the Stage 14 headline P(Cup|best)=0.2753
    (it is the same seeded computation; assert equality);
  - structure checks pass per run;
  - each non-default run compared at 100k seasons; MC SE ~0.0014 reported.

Outputs: outputs/stage17_param_sensitivity.csv
         outputs/reports/stage17_param_sensitivity_report.txt
         outputs/stage17_fig_sensitivity.png
"""
from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stage13_batch_engine import run_batch, _check_structure

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT_CSV = os.path.join(ROOT, "outputs", "stage17_param_sensitivity.csv")
OUT_FIG = os.path.join(ROOT, "outputs", "stage17_fig_sensitivity.png")
REPORT = os.path.join(ROOT, "outputs", "reports", "stage17_param_sensitivity_report.txt")

N_SEASONS = 100_000
SEED = 20261028          # the Stage 13 full-run seed -> default run IS primary
TALENT_SD_DEFAULT = 0.0415
GPG_DEFAULT = 5.4933
HIA_DEFAULT = 0.2563

GRID = {
    "talent_sd": [0.030, 0.035, 0.040, TALENT_SD_DEFAULT, 0.045, 0.050, 0.055],
    "gpg": [5.0, GPG_DEFAULT, 6.0, 6.5],
    "hia": [0.0, 0.10, HIA_DEFAULT, 0.40],
}


def best_team_rates(res: dict[str, np.ndarray]) -> dict[str, float]:
    """Rates for the best team = highest drawn talent GF% each season."""
    n = res["talent_gfpct"].shape[0]
    # known-safe best-team pick: first index of the max per row
    best = (res["talent_gfpct"] == res["talent_gfpct"].max(axis=1, keepdims=True))
    has_tie = best.sum(axis=1) > 1
    n_tie = int(has_tie.sum())
    # pick the first True per row; ties are measure-zero on float32 draws but counted
    idx = best.argmax(axis=1)                       # [n]
    rows = np.arange(n)
    return {
        "p_playoffs": float(res["made_playoffs"][rows, idx].mean()),
        "p_pt": float(res["presidents_trophy"][rows, idx].mean()),
        "p_cup": float(res["cup_winner"][rows, idx].mean()),
        "n_seasons": n, "n_talent_ties": n_tie,
    }


def main() -> None:
    rows = []
    t0 = time.time()
    default_res = None
    for param, values in GRID.items():
        for v in values:
            kw = {}
            if param == "talent_sd" and v != TALENT_SD_DEFAULT:
                kw["talent_sd"] = v
            elif param == "gpg" and v != GPG_DEFAULT:
                kw["gpg"] = v
            elif param == "hia" and v != HIA_DEFAULT:
                kw["hia"] = v
            if kw:
                t1 = time.time()
                res = run_batch(N_SEASONS, SEED, **kw)
                _check_structure(res, N_SEASONS)
                tag = f"override {param}={v}"
            else:
                if default_res is None:
                    t1 = time.time()
                    default_res = run_batch(N_SEASONS, SEED)
                    _check_structure(default_res, N_SEASONS)
                    # primary reproduction check: this seeded run IS the
                    # Stage 13 full run -> must give the Stage 14 headline
                    bt = best_team_rates(default_res)
                    assert abs(bt["p_cup"] - 0.2753) < 0.0006, \
                        f"default P(cup) {bt['p_cup']} != primary 0.2753"
                    print(f"  default run reproduces Stage 13/14 headline "
                          f"P(Cup|best)={bt['p_cup']:.4f} (primary 0.2753) PASS "
                          f"({time.time()-t1:.0f}s)")
                res = default_res
                tag = "DEFAULT (primary)"
            bt = best_team_rates(res)
            rows.append({"parameter": param, "value": v, "is_default": not kw,
                         **bt})
            print(f"  {param}={v} ({tag}): P(cup)={bt['p_cup']:.4f} "
                  f"P(po)={bt['p_playoffs']:.4f} P(pt)={bt['p_pt']:.4f} "
                  f"({time.time()-t1:.0f}s)", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)
    print(f"total time {time.time()-t0:.0f}s")

    # --- report ------------------------------------------------------------
    lines = ["Stage 17 — parameter sensitivity of P(best team wins the Cup)",
             f"({N_SEASONS:,} seasons per grid value; seed {SEED} — the Stage 13",
             "full-run seed, so the default row IS the primary result)",
             ""]
    for param, label in [("talent_sd", "true-talent SD (default 0.0415)"),
                         ("gpg", "total clean goals/game (default 5.4933)"),
                         ("hia", "home-ice advantage, goals/game (default 0.2563)")]:
        s = table[table["parameter"] == param].sort_values("value")
        base = s.loc[s["is_default"], "p_cup"].iloc[0]
        lines.append(f"{label}")
        for _, r in s.iterrows():
            mark = "  <- default" if r["is_default"] else ""
            lines.append(f"  {r['value']:8.4f}: P(cup)={r['p_cup']:.4f} "
                         f"(Δ {r['p_cup']-base:+.4f}, MC SE ~0.0014) "
                         f"P(po)={r['p_playoffs']:.4f} P(pt)={r['p_pt']:.4f}{mark}")
        lines.append("")
    lines.append("Notes:")
    lines.append("- Best team = highest drawn talent GF% per season (first-index")
    lines.append(f"  pick; talent ties across 32 float32 draws per season: "
                 f"{int(table['n_talent_ties'].max())} seasons max).")
    lines.append("- Each sweep varies ONE parameter; the other two stay at defaults.")
    lines.append("- The default row equals the Stage 13/14 primary result")
    lines.append("  (asserted: P(cup) 0.2753).")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # --- figure: 3 panels ---------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    titles = {"talent_sd": "true-talent SD", "gpg": "goals/game (clean)",
              "hia": "home-ice advantage (goals/game)"}
    for ax, (param, title) in zip(axes, titles.items()):
        s = table[table["parameter"] == param].sort_values("value")
        ax.plot(s["value"], s["p_cup"], "o-", lw=2)
        d = s[s["is_default"]]
        ax.axhline(0.03125, color="grey", ls=":", lw=1)
        ax.text(s["value"].min(), 0.03125, " 1/32 baseline", va="bottom",
                fontsize=8, color="grey")
        if len(d):
            ax.plot(d["value"], d["p_cup"], "s", ms=10, mfc="none", mec="red",
                    mew=2, label="default")
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("P(best team wins Cup)")
    fig.suptitle(f"Stage 17: univariate sensitivity of the headline result "
                 f"({N_SEASONS:,} seasons/point)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {REPORT}")
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
