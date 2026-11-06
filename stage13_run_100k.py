#!/usr/bin/env python3
"""Stage 13 (part 2) — Mass simulation driver: 100,000 full seasons.

Runs the Stage 13 batch engine (stage13_batch_engine.run_batch) for 100k
seasons and writes the per-team outcome stack to parquet, long format
(season, team, 12 outcome columns), plus a summary report.

Deterministic: seed 20261028, chunk 2,000 (disjoint Generator streams
seed + chunk_start). Re-running reproduces the file exactly.

Usage:
    .venv/bin/python stage13_run_100k.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import stage13_batch_engine as eng

N_SEASONS = 100_000
SEED = 20261028
OUT_PARQUET = ("/workspace/projects/hockey-lab/analyses/likens/outputs/"
               "stage13_full_100k.parquet")
OUT_REPORT = ("/workspace/projects/hockey-lab/analyses/likens/outputs/"
              "reports/stage13_100k_report.txt")
TEAMS = eng._TEAMS


def main() -> None:
    t0 = time.perf_counter()
    res = eng.run_batch(N_SEASONS, seed=SEED)
    dt = time.perf_counter() - t0
    print(f"100k seasons in {dt:.1f}s ({dt / N_SEASONS * 1e3:.2f} ms/season)")

    # Structure gates before writing anything.
    eng._check_structure(res, N_SEASONS)

    K = np.arange(N_SEASONS)[:, None]
    frames = []
    for j, team in enumerate(TEAMS):
        f = pd.DataFrame({
            "season": np.arange(N_SEASONS, dtype=np.int32),
            "team": team,
            "team_idx": j,
        })
        for c in eng.COLS:
            f[c] = res[c][:, j]
        frames.append(f)
    long = pd.concat(frames, ignore_index=True)
    long.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET} ({len(long):,} rows)")

    # Summary report.
    tr = (-res["talent_gfpct"]).argsort(1).argsort(1)  # 0 = best talent
    best = (tr == 0).argmax(1)  # position of the best-talent team per season
    ki = np.arange(N_SEASONS)
    lines = [
        "Stage 13 — 100,000 full-season simulation (batch engine)",
        f"seed {SEED}, chunk {eng.CHUNK}, runtime {dt:.1f}s "
        f"({dt / N_SEASONS * 1e3:.2f} ms/season)",
        "",
        "League-level:",
        f"  league PPG {res['PTS'].sum() / (N_SEASONS * 32 * 82):.4f}",
        f"  mean team points {res['PTS'].mean():.2f}",
        f"  mean playoff-teams/season {res['made_playoffs'].sum(1).mean():.2f}",
        "",
        "Best team (highest drawn true-talent GF%) outcomes:",
        f"  P(make playoffs)      {res['made_playoffs'][ki, best].mean():.4f}"
        "   (JLikens ~0.978)",
        f"  P(Presidents' Trophy) {res['presidents_trophy'][ki, best].mean():.4f}"
        "   (JLikens ~0.32)",
        f"  P(reach conf final)   {res['round3_wins'][ki, best].mean():.4f}",
        f"  P(win Cup)            {res['cup_winner'][ki, best].mean():.4f}"
        "   (JLikens ~0.22)",
        "",
        "Cup probability by talent rank (1 = best of 32):",
    ]
    for r in [1, 2, 3, 4, 5, 8, 12, 16, 24, 32]:
        pos = (tr == (r - 1)).argmax(1)
        p = res["cup_winner"][ki, pos].mean()
        lines.append(f"  rank {r:2d}: {p:.4f}")
    report = "\n".join(lines)
    with open(OUT_REPORT, "w") as fh:
        fh.write(report + "\n")
    print(report)
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
