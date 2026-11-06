#!/usr/bin/env python3
"""Stage 7 diagnostic — game-coupled vs bootstrap-style independent noise.

Arms (each 100k half-seasons, chunked, seed 20261018 continuing the run
sequence; talents drawn fresh per season from N(0.5, 0.0415)):

  Arm 1 (existing, from stage7_halfseason_100k.parquet):
      real 2025-26 schedule, user's averaging lambda form
          lam_gf = (gf_t + ga_o)/2 + HIA   (home team)
      Game-coupled noise: goals are shared events between teams.

  Arm 2a (bootstrap-style, neutral environment):
      per team, 41 games generated INDEPENDENTLY of all other teams.
      Home/away by coin flip per slot. Opponent = league-average rates
      (gf_o = ga_o = GOALS_PER_GAME/2) every game — the "neutral
      environment" reading of JLikens's validation. Same averaging lambda
      form, HIA applied on the team's home slots (and against it away).

  Arm 2b (bootstrap-style, random opponent):
      as 2a but a fresh opponent talent o ~ N(0.5, 0.0415) drawn per game
      slot, so opponent-strength variance is added while games stay
      independent across teams.

Analytic expectation (delta method, averaging form, league-avg opponent):
  Per game, lam_gf + lam_ga = GOALS_PER_GAME (5.4933) regardless of talent,
  and E[GF%] = (t + 0.5)/2 — NOTE the averaging form halves the talent
  signal (a .600 team scores at .550 expected GF% vs an average opponent).
  So predicted per-team GF% noise SD ~ sqrt(1/(4*41*5.4933)) = 0.0333 and
  predicted pooled SD = sqrt((0.5*0.0415)^2 + 0.0333^2) = 0.0392.
  If instead talent expressed fully (coefficient 1.0, e.g. a multiplicative
  lambda form), pooled SD would be sqrt(0.0415^2 + 0.0333^2) = 0.0531.

The measured arms localize WHERE the 0.0384-vs-0.052 gap comes from:
  - Arm 2a ≈ 0.039  -> the gap is the lambda FORM (talent halving), not
    game coupling (independent-noise arm reproduces Arm 1!).
  - Arm 2a ≈ 0.053  -> the gap is game coupling after all.
  - Arm 2b vs 2a    -> how much opponent-strength variance alone adds.

Output: outputs/reports/stage7_diagnostic_arms.txt
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from assign_team_talent import GFDIST_MEAN, GFDIST_STD, GOALS_PER_GAME_LEAGUEAVG, HOME_ICE_ADVANTAGE

SEED = 20261018
N_SEASONS = 100_000
CHUNK = 2_000
CUT = 41
HIA = HOME_ICE_ADVANTAGE
GPG = GOALS_PER_GAME_LEAGUEAVG
AVG_RATE = GPG / 2.0  # league-average per-team GF and GA rate

OUT_REPORT = "/workspace/projects/hockey-lab/analyses/likens/outputs/reports/stage7_diagnostic_arms.txt"


def run_arm(opponent: str, seed: int = SEED + 7_000_000) -> np.ndarray:
    """Simulate 100k independent-noise half-seasons; return (N_SEASONS*32,) GF% array.

    opponent: 'average' (Arm 2a) or 'random' (Arm 2b).
    """
    rng = np.random.default_rng(seed)
    out = np.empty(N_SEASONS * 32, dtype=np.float64)
    done = 0
    t0 = time.perf_counter()
    while done < N_SEASONS:
        n = min(CHUNK, N_SEASONS - done)
        # talents (n, 32) and per-slot home flips (n, 32, CUT)
        t = rng.normal(GFDIST_MEAN, GFDIST_STD, size=(n, 32))
        gf_rate = GPG * t
        ga_rate = GPG * (1.0 - t)
        home = rng.random((n, 32, CUT)) < 0.5

        if opponent == "average":
            gf_o = np.full((n, 32, CUT), AVG_RATE)
            ga_o = np.full((n, 32, CUT), AVG_RATE)
        elif opponent == "random":
            o = rng.normal(GFDIST_MEAN, GFDIST_STD, size=(n, 32, CUT))
            gf_o = GPG * o
            ga_o = GPG * (1.0 - o)
        else:
            raise ValueError(opponent)

        lam_gf = (gf_rate[:, :, None] + ga_o) / 2.0 + HIA * home
        lam_ga = (ga_rate[:, :, None] + gf_o) / 2.0 + HIA * (~home)
        GF = rng.poisson(lam_gf).sum(axis=2).astype(np.float64)
        GA = rng.poisson(lam_ga).sum(axis=2).astype(np.float64)
        out[done * 32:(done + n) * 32] = (GF / (GF + GA)).ravel()
        done += n
    print(f"  arm 2{'a' if opponent == 'average' else 'b'}: {N_SEASONS:,} half-seasons in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    return out


def summarize(name: str, vals: np.ndarray, lines: list[str]) -> None:
    per_season = vals.reshape(N_SEASONS, 32)
    pooled_sd = vals.std(ddof=1)
    lines.append(f"{name}")
    lines.append(f"  pooled SD: {pooled_sd:.5f}")
    lines.append(f"  mean per-season SD: {per_season.std(axis=1, ddof=1).mean():.5f} "
                 f"(min {per_season.std(axis=1, ddof=1).min():.4f}, max {per_season.std(axis=1, ddof=1).max():.4f})")
    lines.append(f"  pooled mean GF%: {vals.mean():.5f}")


def main() -> None:
    lines = ["Stage 7 diagnostic — noise-structure arms (100k half-seasons each)"]

    # Arm 1 from the existing run.
    # GUARD (fixed 2026-10-27): the parquet may be the AVERAGING-form run
    # (pre-revision) or the MULTIPlicative-form run (current); the label must
    # say which. The averaging-form run had pooled corr(talent, GF%41)=0.516
    # vs the multiplicative-form 0.763 — distinguish via that correlation.
    arm1 = pd.read_parquet("/workspace/projects/hockey-lab/analyses/likens/outputs/stage7_halfseason_100k.parquet")
    corr = np.corrcoef(arm1["talent_gfpct"], arm1["gfpct41"])[0, 1]
    form = "multiplicative" if corr > 0.6 else "averaging (STALE — pre-revision run!)"
    summarize(f"Arm 1 — game-coupled schedule, {form} lambdas (existing 100k run; corr {corr:.3f})",
              arm1["gfpct41"].to_numpy(), lines)

    # analytic predictions (delta method)
    noise_var = 1.0 / (4.0 * CUT * GPG)
    noise_sd = np.sqrt(noise_var)
    lines.append("")
    lines.append("Analytic (delta method, averaging lambda form):")
    lines.append(f"  per-team independent noise SD: {noise_sd:.5f}")
    lines.append(f"  predicted pooled SD, talent coefficient 0.5 (averaging form): "
                 f"{np.sqrt((0.5 * GFDIST_STD) ** 2 + noise_var):.5f}")
    lines.append(f"  predicted pooled SD, talent coefficient 1.0 (full expression): "
                 f"{np.sqrt(GFDIST_STD ** 2 + noise_var):.5f}")
    lines.append("")

    for opp, label in (("average", "Arm 2a — independent games, league-average opponent"),
                       ("random", "Arm 2b — independent games, fresh random opponent per game")):
        vals = run_arm(opp)
        summarize(label, vals, lines)
        lines.append("")

    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
