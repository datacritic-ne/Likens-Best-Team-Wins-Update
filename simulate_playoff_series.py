#!/usr/bin/env python3
"""Stage 10 — Best-of-7 playoff series simulation.

Two layers, both reusing the Stage 6 multiplicative Poisson machinery
(assign_team_talent rates, locked constants GPG 5.4933 / HIA 0.2563):

  1. simulate_playoff_game(): one playoff game — same lambda form as the
     regular-season simulator (home_lam = 2*gf_h*ga_a/GPG + HIA; away_lam =
     2*gf_a*ga_h/GPG), Poisson goal draws, but NO standings points: only the
     game winner (and goals) is returned.

     Playoff OT (user decision 2026-10-19, option 1): regulation ties are
     resolved by a memoryless Poisson race with the regulation rates —
     P(home wins in OT) = lam_home / (lam_home + lam_away), one draw per
     tied game. This is "rates hold in OT" (our settled 2026-09-21
     convention) formalized as unlimited OT periods: the next-goal time in a
     superposed Poisson process is Exp(lam_home + lam_away) and belongs to
     home with probability lam_home/(lam_home + lam_away). Under equal
     lambdas it reduces to a coin flip, consistent with the regular-season
     treatment. Ties cannot survive: every game yields a winner.

  2. simulate_series(): best-of-7 in the 2-2-1-1-1 home-ice pattern — the
     team with home ice hosts games 1, 2, 5, and 7 (its opponent hosts 3, 4,
     6; confirmed from the 2013-14+ format, records.nhl.com). Wins are
     tallied and the series terminates when one team reaches 4.

Adapted from the user's notebook approach: single-game Poisson draw, then a
best-of-7 loop tracking the win tally. Reworked on the Stage 6 vectorized
machinery so a future vectorized 100k-series run (Stage 13) can reuse the
same math; the scalar functions here are the readable reference
implementation and the verification entry point.

Usage:
    from simulate_playoff_series import simulate_playoff_game, simulate_series
    winner, log = simulate_series("TBL", "BOS", team_data, rng)

Running this file executes the Stage 10 verification checks (goal-convergence
z-checks, the closed-form series-length distribution under pure 50/50 games,
direction checks, and a 2-2-1-1-1 home-pattern hand-check).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from assign_team_talent import assign_team_talent
from simulate_games import GPG, HIA, _lambdas

# 2-2-1-1-1: home-ice team hosts games 1, 2, 5, 7 (0-indexed 0, 1, 4, 6).
SERIES_GAMES = 7
WINS_NEEDED = 4
HOME_PATTERN_HIGHER = np.array([True, True, False, False, True, False, True])


def simulate_playoff_game(
    home_team: str,
    away_team: str,
    team_data: pd.DataFrame,
    rng: np.random.Generator,
    hia: float = HIA,
) -> dict:
    """Simulate one playoff game; returns goals and the winner (no points)."""
    home_lam, away_lam = _lambdas(np.array([home_team]), np.array([away_team]), team_data, hia)
    home_goals = int(rng.poisson(home_lam[0]))
    away_goals = int(rng.poisson(away_lam[0]))

    if home_goals == away_goals:
        # Memoryless Poisson race in OT (rates hold; unlimited OT periods).
        p_home_ot = home_lam[0] / (home_lam[0] + away_lam[0])
        home_won = bool(rng.random() < p_home_ot)
    else:
        home_won = home_goals > away_goals

    return {
        "home": home_team,
        "away": away_team,
        "home_lambda": float(home_lam[0]),
        "away_lambda": float(away_lam[0]),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "went_to_ot": home_goals == away_goals,
        "winner": home_team if home_won else away_team,
    }


def simulate_series(
    higher: str,
    lower: str,
    team_data: pd.DataFrame,
    rng: np.random.Generator,
    hia: float = HIA,
) -> tuple[str, pd.DataFrame]:
    """Simulate one best-of-7 series (2-2-1-1-1).

    Arguments:
        higher: team with home ice (hosts games 1, 2, 5, 7).
        lower: its opponent (hosts games 3, 4, 6).
        team_data: talent table (assign_team_talent output).
        rng: numpy Generator controlling all draws.
        hia: home-ice advantage in goals/game.

    Returns:
        (winner, game_log)
        winner: team code of the series winner (exactly 4 wins).
        game_log: DataFrame, one row per game played: game_no (1-based),
            home, away, goals, winner, went_to_ot.
    """
    wins = {higher: 0, lower: 0}
    rows = []
    for game_no in range(1, SERIES_GAMES + 1):
        home, away = (higher, lower) if HOME_PATTERN_HIGHER[game_no - 1] else (lower, higher)
        g = simulate_playoff_game(home, away, team_data, rng, hia=hia)
        rows.append({
            "game_no": game_no, "home": home, "away": away,
            "home_goals": g["home_goals"], "away_goals": g["away_goals"],
            "winner": g["winner"], "went_to_ot": g["went_to_ot"],
        })
        wins[g["winner"]] += 1
        if wins[g["winner"]] == WINS_NEEDED:
            break

    assert wins[higher] == WINS_NEEDED or wins[lower] == WINS_NEEDED, \
        "series ended without a 4-win team"
    assert len(rows) <= SERIES_GAMES, "series exceeded 7 games"
    assert max(wins.values()) == WINS_NEEDED
    winner = higher if wins[higher] == WINS_NEEDED else lower
    return winner, pd.DataFrame(rows)


def series_winner_vectorized(
    higher: str,
    lower: str,
    team_data: pd.DataFrame,
    rng: np.random.Generator,
    n_series: int = 1,
    hia: float = HIA,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized batch: winners and series lengths for n_series sims.

    Same math and draw order as the scalar path (per-game Poisson draws in
    schedule order, OT race draws on ties). For Stage 13 mass runs; the
    scalar functions above remain the readable reference.
    """
    gf = team_data["GF_per_game"]
    ga = team_data["GA_per_game"]
    hg, ag = gf.loc[higher], ga.loc[higher]
    lg, la = gf.loc[lower], ga.loc[lower]

    lam_home_h = np.full(n_series, 2.0 * hg * la / GPG + hia)
    lam_away_l = np.full(n_series, 2.0 * lg * ag / GPG)
    lam_home_l = np.full(n_series, 2.0 * lg * ag / GPG + hia)
    lam_away_h = np.full(n_series, 2.0 * hg * la / GPG)

    # Per potential game, goals for the home-ice team and away team.
    home_lams = [lam_home_h, lam_home_h, lam_home_l, lam_home_l,
                 lam_home_h, lam_home_l, lam_home_h]
    away_lams = [lam_away_l, lam_away_l, lam_away_h, lam_away_h,
                 lam_away_l, lam_away_h, lam_away_l]

    higher_wins = np.zeros(n_series, dtype=int)
    lower_wins = np.zeros(n_series, dtype=int)
    lengths = np.full(n_series, SERIES_GAMES, dtype=int)
    decided = np.zeros(n_series, dtype=bool)

    for g in range(SERIES_GAMES):
        pending = ~decided
        if not pending.any():
            break
        hg_draw = rng.poisson(home_lams[g][pending])
        ag_draw = rng.poisson(away_lams[g][pending])
        tie = hg_draw == ag_draw
        p_home = np.where(
            home_lams[g][pending] + away_lams[g][pending] > 0,
            home_lams[g][pending] / (home_lams[g][pending] + away_lams[g][pending]),
            0.5,
        )
        home_won = (hg_draw > ag_draw) | (tie & (rng.random(tie.shape) < p_home))
        higher_is_home = HOME_PATTERN_HIGHER[g]
        win_higher = home_won if higher_is_home else ~home_won
        higher_wins[pending] += win_higher
        lower_wins[pending] += ~win_higher
        newly = pending & ((higher_wins == WINS_NEEDED) | (lower_wins == WINS_NEEDED))
        lengths[newly] = g + 1
        decided |= newly

    assert ((higher_wins == WINS_NEEDED) ^ (lower_wins == WINS_NEEDED)).all(), \
        "vectorized series ended without exactly one 4-win team"
    winners = np.where(higher_wins == WINS_NEEDED, higher, lower)
    return winners, lengths


# --- Stage 10 verification checks ---------------------------------------------


# Check-count policy (2026-10-20): the original counts (100k scalar games /
# 200k scalar series) made the scalar loops time out twice (>400 s). Large-N
# checks now run through series_winner_vectorized (the Stage 13 path); scalar
# loops are kept only at small n for the reference-implementation checks and
# a scalar-vs-vectorized agreement check.

def check_game_convergence(n_games: int = 20_000, seed: int = 20251019) -> None:
    rng = np.random.default_rng(seed)
    team_data = assign_team_talent(rng)
    fixed_home, fixed_away = "TBL", "CGY"  # big talent gap in the seed draw

    lam_h = float(2.0 * team_data.loc[fixed_home, "GF_per_game"]
                  * team_data.loc[fixed_away, "GA_per_game"] / GPG + HIA)
    lam_a = float(2.0 * team_data.loc[fixed_away, "GF_per_game"]
                  * team_data.loc[fixed_home, "GA_per_game"] / GPG)

    home_wins = 0
    hg_sum = ag_sum = 0
    for _ in range(n_games):
        g = simulate_playoff_game(fixed_home, fixed_away, team_data, rng)
        hg_sum += g["home_goals"]
        ag_sum += g["away_goals"]
        home_wins += g["winner"] == fixed_home

    se_h, se_a = np.sqrt(lam_h / n_games), np.sqrt(lam_a / n_games)
    z_h = (hg_sum / n_games - lam_h) / se_h
    z_a = (ag_sum / n_games - lam_a) / se_a
    print(f"[game] {n_games} sims of {fixed_home} (home) vs {fixed_away}")
    print(f"  lambda home {lam_h:.4f}: sim mean {hg_sum/n_games:.4f} (z={z_h:+.2f})")
    print(f"  lambda away {lam_a:.4f}: sim mean {ag_sum/n_games:.4f} (z={z_a:+.2f})")
    assert abs(z_h) < 4 and abs(z_a) < 4, "simulated goals do not match Poisson means"
    print(f"  home win share {home_wins/n_games:.4f} (OT included; no ties survive)")
    print("  [PASS] goal convergence + every game yields a winner")


def check_series_closed_form(n_series: int = 200_000, seed: int = 20251020) -> None:
    """Pure 50/50 games (equal talents, hia=0): closed-form length dist.

    Run through series_winner_vectorized (200k scalar series timed out);
    vectorized=scalar math agreement is separately checked in main().
    """
    rng = np.random.default_rng(seed)
    team_data = assign_team_talent(rng)
    t = team_data.index[0]
    team_data = team_data.copy()
    # Set ALL teams to identical (league-average) rates => 50/50 games, but
    # keep distinct labels so "higher" vs "lower" winners are distinguishable.
    avg = team_data.mean(numeric_only=True)
    for c in team_data.columns:
        team_data[c] = float(avg[c])

    winners, lengths = series_winner_vectorized(t, team_data.index[1],
                                                team_data, rng,
                                                n_series=n_series, hia=0.0)

    # Closed form for p=0.5 best-of-7: standard result {4: .125, 5: .25,
    # 6: .3125, 7: .3125}
    expected = {4: 0.125, 5: 0.25, 6: 0.3125, 7: 0.3125}
    obs = pd.Series(lengths).value_counts(normalize=True).sort_index()
    for n, p in expected.items():
        se = np.sqrt(p * (1 - p) / n_series)
        z = (obs.get(n, 0) - p) / se
        print(f"  length {n}: obs {obs.get(n, 0):.4f} vs {p:.4f} (z={z:+.2f})")
        assert abs(z) < 4, f"series-length distribution mismatch at n={n}"
    share = float(np.mean(winners == t))
    print(f"  P(designated team wins) = {share:.4f} (expect 0.5)")
    assert abs(share - 0.5) < 4 * np.sqrt(0.25 / n_series)
    print("  [PASS] closed-form series-length distribution + 50/50 winner (vectorized)")


def check_direction(seed: int = 20251021, n_series: int = 20_000) -> None:
    rng = np.random.default_rng(seed)
    team_data = assign_team_talent(rng)
    # Strongest vs weakest by DRAWN talent (seed-independent gap).
    strong = team_data["GF_per_game"].idxmax()
    weak = team_data["GF_per_game"].idxmin()

    w_strong_v, len_strong = series_winner_vectorized(
        strong, weak, team_data, rng, n_series=n_series)
    w_strong = float(np.mean(w_strong_v == strong))
    # Equal talents + HIA: higher seed should win > 50%.
    equal = team_data.astype(float).copy()
    avg = equal.mean(numeric_only=True)
    for c in equal.columns:
        equal[c] = float(avg[c])
    w_hia_v, _ = series_winner_vectorized("TBL", "CGY", equal, rng,
                                          n_series=n_series)
    w_hia = float(np.mean(w_hia_v == "TBL"))
    print(f"[direction] P(strong beats weak, home) = {w_strong:.4f}; "
          f"P(higher wins | equal talents + HIA) = {w_hia:.4f}")
    assert w_strong > 0.8, "big talent gap does not produce a strong favorite"
    assert 0.5 < w_hia < 0.75, "HIA effect implausible"
    print("  [PASS] direction checks (talent favors; HIA matters) (vectorized)")


def check_home_pattern(seed: int = 20251022) -> None:
    rng = np.random.default_rng(seed)
    team_data = assign_team_talent(rng)
    w, log = simulate_series("TBL", "CGY", team_data, rng)
    expected_home = ["TBL", "TBL", "CGY", "CGY", "TBL", "CGY", "TBL"][:len(log)]
    assert list(log["home"]) == expected_home, "2-2-1-1-1 home pattern violated"
    tally = log["winner"].value_counts()
    assert tally[w] == 4 and len(log) <= 7
    print(f"[hand-check] sample series: {w} in {len(log)} games; "
          f"home pattern {list(log['home'])}; tally {tally.to_dict()}")
    print("  [PASS] 2-2-1-1-1 pattern + win tally + 4-win termination")


def main() -> None:
    print("Stage 10 verification")
    check_game_convergence()
    check_series_closed_form()
    check_direction()
    check_home_pattern()

    # Vectorized vs scalar consistency. NOTE: numpy consumes Generator state
    # differently for scalar vs array Poisson draws, so the two paths CANNOT
    # agree series-by-series on a shared seed even with identical math. The
    # check is therefore distributional: independent large samples from each
    # path must agree on P(higher wins) and mean series length within MC error.
    rng_s = np.random.default_rng(20251023)
    team_data = assign_team_talent(rng_s)
    n_pair = 2_000
    scal_w, scal_l = [], []
    for _ in range(n_pair):
        w, log = simulate_series("TBL", "CGY", team_data, rng_s)
        scal_w.append(w)
        scal_l.append(len(log))
    rng_v = np.random.default_rng(20251024)
    vec_w, vec_l = series_winner_vectorized("TBL", "CGY", team_data, rng_v,
                                            n_series=10 * n_pair)
    p_s = float(np.mean(np.array(scal_w) == "TBL"))
    p_v = float(np.mean(vec_w == "TBL"))
    se = np.sqrt(p_v * (1 - p_v) / n_pair)  # scalar sample is the noisier one
    z = (p_s - p_v) / se
    print(f"[scalar=vectorized] P(TBL wins): scalar {p_s:.4f} (n={n_pair}) vs "
          f"vectorized {p_v:.4f} (n={10*n_pair}), z={z:+.2f}")
    assert abs(z) < 4, "scalar and vectorized winner rates disagree"
    ml_s, ml_v = float(np.mean(scal_l)), float(vec_l.mean())
    print(f"  mean length: scalar {ml_s:.3f} vs vectorized {ml_v:.3f}")
    assert abs(ml_s - ml_v) < 4 * float(np.std(scal_l)) / np.sqrt(n_pair), \
        "scalar and vectorized mean series lengths disagree"
    print("  [PASS] vectorized path statistically matches scalar reference")

    share_v = p_v
    print(f"[vectorized] {10 * n_pair} series: P(TBL wins) = {share_v:.4f}, "
          f"mean length {vec_l.mean():.3f}")
    # Sanity bounds only (seed-independent; the direction check covers the
    # talent-favors-winner behaviour with a seed-independent talent gap).
    assert 0.02 < share_v < 0.98, "vectorized winner share implausible"
    print("  [PASS] vectorized batch path sane")


if __name__ == "__main__":
    main()
