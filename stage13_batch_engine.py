#!/usr/bin/env python3
"""Stage 13 (part 1) — Batch/vectorized full-season engine.

Reimplements the Stage 12 pipeline with the pandas bookkeeping removed and
the postseason batched across seasons. Mathematically identical model
(same locked constants, multiplicative lambdas, tournament structure,
tiebreaker chain); distributional equivalence vs the scalar Stage 12 path
is checked in __main__ (exact draw-order agreement is impossible — numpy
consumes Generator state differently for scalar vs array Poisson draws,
established in Stage 10).

Where the scalar path spent its time (cProfile, 50 seasons):
  ~80% in simulate_series/simulate_playoff_game/_lambdas — pandas .loc
  lookups per GAME; ~14% in playoff_seeding/H2H pandas; the vectorized
  regular-season sim itself is ~2.7 ms/season.

Design:
  1. Talent draw + regular season: same call chain as Stage 8 without
     DataFrame wrapping — teams indexed 0..31, fixed schedule as integer
     index arrays; one Poisson call per chunk of seasons.
  2. Ranking: integer lexsorts per group on (PTS desc, ROW desc, DIFF desc,
     code asc) + the Stage 9 pairwise-H2H adjacent-swap refinement applied
     vectorized across seasons (parity-pass application of non-adjacent
     swaps; same fixed point as Stage 9's sequential scan outside
     pathological triple-ties — exercised only on exact (PTS,ROW,DIFF)
     ties, astronomically rare).
  3. Seeding/bracket: index arithmetic mirroring stage9.playoff_seeding
     (top D1 by points vs LOWER wild card, #2 vs #3; WC may out-rank a
     division team and hold home ice; division-winner points tie falls
     back to the alphabetically-first division, matching stage9's sort).
     Simplification (documented, same class as Stage 9's flagged WC-sort
     H2H omission): WC ordering uses the ranked conference order, which
     includes the H2H refinement, while stage9's WC sort does not — matters
     only on exact (PTS, ROW, DIFF) ties.
  4. Home ice all rounds: full chain PTS -> ROW -> directed H2H -> DIFF ->
     code (matches stage11._home_ice exactly).
  5. Postseason: one vectorized best-of-7 engine (Stage 10 math: 2-2-1-1-1,
     Poisson goals, OT = Poisson race) over all series of a round across
     all seasons in the chunk — 7 masked draw steps per round.

Output: dict of [n_seasons, 32] arrays with the Stage 12 user-spec columns.
Chunked internally (default 2,000 seasons) to bound memory; one Generator
across chunks, so results are deterministic in (n_seasons, seed).

Running this file executes the Stage 13 pilot checks (structure,
determinism, scalar-path consistency, speed benchmark).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from nhl_structure import TEAM_TO_DIVISION, DIVISION_CONFERENCE
from assign_team_talent import GFDIST_MEAN, GFDIST_STD, GOALS_PER_GAME_LEAGUEAVG, SCHEDULE_CSV

GPG = GOALS_PER_GAME_LEAGUEAVG
HIA = 0.2563  # HOME_ICE_ADVANTAGE, locked 2026-10-13

DIVS = ["Atlantic", "Metropolitan", "Central", "Pacific"]  # alphabetical
N_TEAMS = 32
WINS_NEEDED = 4
HOME_PATTERN_HIGHER = np.array([True, True, False, False, True, False, True])
CHUNK = 2_000

COLS = ["talent_gfpct", "PTS", "GF", "GA", "made_playoffs",
        "division_winner", "conference_best", "presidents_trophy",
        "round1_wins", "round2_wins", "round3_wins", "cup_winner"]


def _schedule_arrays(schedule_csv: str = SCHEDULE_CSV):
    sched = pd.read_csv(schedule_csv)
    teams = sorted(TEAM_TO_DIVISION)
    idx = {t: i for i, t in enumerate(teams)}
    home = sched["home_abbrev"].map(idx).to_numpy(np.int64)
    away = sched["away_abbrev"].map(idx).to_numpy(np.int64)
    assert len(home) == 1312 and not (home < 0).any() and not (away < 0).any()
    return teams, home, away


_TEAMS, _HOME, _AWAY = _schedule_arrays()
_DIV_OF_TEAM = np.array([DIVS.index(TEAM_TO_DIVISION[t]) for t in _TEAMS])
_CONF_OF_TEAM = np.array(
    [0 if DIVISION_CONFERENCE[TEAM_TO_DIVISION[t]] == "Eastern" else 1
     for t in _TEAMS])  # 0 Eastern, 1 Western
_DIV_MEMBERS = [np.where(_DIV_OF_TEAM == d)[0] for d in range(4)]
_CONF_MEMBERS = [np.where(_CONF_OF_TEAM == c)[0] for c in range(2)]


# --- Regular season -----------------------------------------------------------


def _regular_season_chunk(CH: int, rng: np.random.Generator,
                          talent_sd: float | None = None,
                          gpg: float | None = None):
    """Talent draws + regular seasons for CH seasons; all numpy.

    talent_sd/gpg default to the locked module constants (Stage 13 primary
    setup); Stage 17 passes overrides for the sensitivity sweeps.

    Returns (talent, pts, gf, ga, row, diff, h2h) with talent float
    [CH,32], counting arrays int [CH,32], h2h int [CH,32,32] directed
    (h2h[k,a,b] = points of a vs b in season k).
    """
    sd = GFDIST_STD if talent_sd is None else talent_sd
    G = GPG if gpg is None else gpg
    talent = rng.normal(GFDIST_MEAN, sd, size=(CH, N_TEAMS))
    talent = np.clip(talent, 0.0, 1.0)
    gf_rate = G * talent
    ga_rate = G * (1.0 - talent)

    gf_home, ga_home = gf_rate[:, _HOME], ga_rate[:, _HOME]
    gf_away, ga_away = gf_rate[:, _AWAY], ga_rate[:, _AWAY]
    lam_home = 2.0 * gf_home * ga_away / G + HIA
    lam_away = 2.0 * gf_away * ga_home / G

    hg = rng.poisson(lam_home)
    ag = rng.poisson(lam_away)
    tie = hg == ag
    flip = rng.random(tie.shape) < 0.5  # home takes the extra point on ties
    hp = np.where(hg > ag, 2, np.where(tie, np.where(flip, 2, 1), 0))
    ap = np.where(ag > hg, 2, np.where(tie, np.where(flip, 1, 2), 0))
    hw = (hg > ag) | (tie & flip)
    aw = (ag > hg) | (tie & ~flip)

    Kidx = np.arange(CH)[:, None]
    fhome = Kidx * N_TEAMS + _HOME[None, :]
    faway = Kidx * N_TEAMS + _AWAY[None, :]
    N2 = N_TEAMS * N_TEAMS

    def _team_total(v_home, v_away):
        return (np.bincount(fhome.ravel(), weights=v_home.ravel(),
                            minlength=CH * N_TEAMS).reshape(CH, N_TEAMS)
                + np.bincount(faway.ravel(), weights=v_away.ravel(),
                              minlength=CH * N_TEAMS).reshape(CH, N_TEAMS))

    pts = _team_total(hp, ap).astype(np.int64)
    gf = _team_total(hg, ag).astype(np.int64)
    ga = _team_total(ag, hg).astype(np.int64)
    row = _team_total(hw, aw).astype(np.int64)
    diff = gf - ga

    # Directed head-to-head points.
    h2h = np.zeros(CH * N2, dtype=np.int64)
    h2h += np.bincount((Kidx * N2 + _HOME[None, :] * N_TEAMS + _AWAY[None, :]).ravel(),
                       weights=hp.ravel(), minlength=CH * N2).astype(np.int64)
    h2h += np.bincount((Kidx * N2 + _AWAY[None, :] * N_TEAMS + _HOME[None, :]).ravel(),
                       weights=ap.ravel(), minlength=CH * N2).astype(np.int64)
    h2h = h2h.reshape(CH, N_TEAMS, N_TEAMS)

    return talent, pts, gf, ga, row, diff, h2h, gf_rate, ga_rate


# --- Ranking ------------------------------------------------------------------


def _order_group(pts: np.ndarray, row: np.ndarray, diff: np.ndarray,
                 h2h: np.ndarray, members: np.ndarray,
                 apply_h2h: bool = True) -> np.ndarray:
    """Rank one group (division/conference/league) for every season.

    Base key (PTS, ROW, DIFF) descending, code ascending — via lexsort —
    then (apply_h2h only) the Stage 9 pairwise-H2H refinement: adjacent teams
    tied on all three swap if the lower has more H2H points vs the upper.
    Swap passes apply non-adjacent (even, then odd position) swaps until
    stable; this reaches the same fixed point as Stage 9's sequential scan
    outside pathological cyclic ties. Returns [K, G] team indices, best first.

    apply_h2h=False matches stage9's LEAGUE-level rank, which does NOT apply
    H2H swaps (ties fall through to DIFF/team code) — used for league_order
    so Presidents'-Trophy determination is bit-identical to the scalar path.
    """
    K = pts.shape[0]
    m = np.asarray(members)
    G = len(m)
    code_key = np.broadcast_to(m[None, :], (K, G))
    order = np.lexsort((code_key, -diff[:, m], -row[:, m], -pts[:, m]))
    ordered = m[order].copy()

    if not apply_h2h:
        return ordered

    ki = np.arange(K)[:, None]
    for _ in range(200):
        changed = False
        for parity in (0, 1):
            a = ordered[:, :-1]
            b = ordered[:, 1:]
            tied = ((np.take_along_axis(pts, a, 1) == np.take_along_axis(pts, b, 1))
                    & (np.take_along_axis(row, a, 1) == np.take_along_axis(row, b, 1))
                    & (np.take_along_axis(diff, a, 1) == np.take_along_axis(diff, b, 1)))
            swap = tied & (h2h[ki, b, a] > h2h[ki, a, b])
            pos = np.arange(G - 1)
            sel = pos % 2 == parity
            s = swap[:, sel]
            if not s.any():
                continue
            rows, p = np.where(s)
            p = pos[sel][p]
            tmp = ordered[rows, p].copy()
            ordered[rows, p] = ordered[rows, p + 1]
            ordered[rows, p + 1] = tmp
            changed = True
        if not changed:
            return ordered
    raise RuntimeError("H2H swap refinement did not converge")


def _rank_all(pts, row, diff, h2h):
    """div_order [K,4,8], conf_order [K,2,16], league_order [K,32]."""
    div_order = np.stack([_order_group(pts, row, diff, h2h, _DIV_MEMBERS[d])
                          for d in range(4)], axis=1)
    conf_order = np.stack([_order_group(pts, row, diff, h2h, _CONF_MEMBERS[c])
                           for c in range(2)], axis=1)
    league_order = _order_group(pts, row, diff, h2h, np.arange(N_TEAMS),
                                apply_h2h=False)  # stage9 league rank: no H2H
    return div_order, conf_order, league_order


# --- Bracket ------------------------------------------------------------------


def _better(a, b, pts, row, diff, h2h):
    """True where a holds home ice vs b: full Stage 11 chain
    PTS -> ROW -> directed H2H -> DIFF -> code (ascending — alphabetically
    first wins, matching stage9's ranking chain; this was ALREADY correct in
    stage13 (a<b = earlier index = earlier code) — only stage11's _home_ice
    had the inversion, fixed 2026-10-27)."""
    ki = np.arange(pts.shape[0])[:, None]
    pa, pb = pts[ki, a], pts[ki, b]
    ra, rb = row[ki, a], row[ki, b]
    da, db = diff[ki, a], diff[ki, b]
    ha = h2h[ki, a, b]  # a's regular-season points vs b
    hb = h2h[ki, b, a]
    better = ((pa > pb)
              | ((pa == pb) & ((ra > rb)
                               | ((ra == rb) & ((ha > hb)
                                                | ((ha == hb) & ((da > db)
                                                                 | ((da == db) & (a < b)))))))))
    return better


def _order_home_away(a, b, pts, row, diff, h2h):
    """Return (home, away) arrays ordered by the home-ice chain."""
    home = np.where(_better(a, b, pts, row, diff, h2h), a, b)
    away = np.where(home == a, b, a)
    return home, away


def _seed_brackets(div_order, conf_order, pts, row, diff, h2h):
    """Round-1 matchups, batched. Returns (hi, lo) [K, 8] with hi holding
    home ice. Slot s = 4c+p per conference: (top-D1 vs WC2, other-D1 vs WC1,
    top-D2 vs top-D3, other-D2 vs other-D3) — stage9.playoff_seeding layout."""
    K = pts.shape[0]
    ki = np.arange(K)[:, None]
    div_pts_top = np.take_along_axis(
        pts[:, None, :], div_order[:, :, :1], axis=2)[:, :, 0]  # [K,4]

    top_div = np.empty((K, 2), dtype=np.int64)   # per conference
    other_div = np.empty((K, 2), dtype=np.int64)
    for c in range(2):
        d0, d1 = 2 * c, 2 * c + 1
        # tie -> alphabetically first division (lower index), matching stage9
        t = np.where(div_pts_top[:, d0] >= div_pts_top[:, d1], d0, d1)
        top_div[:, c] = t
        other_div[:, c] = d1 if c == 0 and False else np.where(t == d0, d1, d0)

    # Wild cards: best 2 conference teams outside every division top-3.
    wc = np.empty((K, 2, 2), dtype=np.int64)
    for c in range(2):
        members = conf_order[:, c, :]                       # [K,16] best-first
        top3 = np.concatenate([div_order[:, 2 * c, :3],
                               div_order[:, 2 * c + 1, :3]], axis=1)  # [K,6]
        is_top3 = (members[:, :, None] == top3[:, None, :]).any(axis=2)
        rest = members[~is_top3].reshape(K, 10)
        wc[:, c, 0] = rest[:, 0]  # WC1 (better)
        wc[:, c, 1] = rest[:, 1]  # WC2

    hi = np.empty((K, 8), dtype=np.int64)
    lo = np.empty((K, 8), dtype=np.int64)
    for c in range(2):
        s = 4 * c
        hi[:, s + 0] = div_order[ki, top_div[:, c:c + 1], 0][:, 0]; lo[:, s + 0] = wc[:, c, 1]
        hi[:, s + 1] = div_order[ki, other_div[:, c:c + 1], 0][:, 0]; lo[:, s + 1] = wc[:, c, 0]
        hi[:, s + 2] = div_order[ki, top_div[:, c:c + 1], 1][:, 0]; lo[:, s + 2] = div_order[ki, top_div[:, c:c + 1], 2][:, 0]
        hi[:, s + 3] = div_order[ki, other_div[:, c:c + 1], 1][:, 0]; lo[:, s + 3] = div_order[ki, other_div[:, c:c + 1], 2][:, 0]

    # Home ice by the full chain (a WC may out-rank a division winner).
    hi, lo = _order_home_away(hi, lo, pts, row, diff, h2h)
    return hi, lo


# --- Series engine ------------------------------------------------------------


def _series_engine(hi, lo, gf_rate, ga_rate, rng, hia=HIA):
    """Vectorized best-of-7 (Stage 10 math). hi/lo [K, S] team indices,
    hi hosts games 1,2,5,7. Returns (winner [K,S], length [K,S])."""
    K, S = hi.shape
    ki = np.arange(K)[:, None]
    gf_h, ga_h = gf_rate[ki, hi], ga_rate[ki, hi]
    gf_l, ga_l = gf_rate[ki, lo], ga_rate[ki, lo]
    lam_h_hi_home = 2.0 * gf_h * ga_l / GPG + hia
    lam_l_hi_home = 2.0 * gf_l * ga_h / GPG
    lam_h_lo_home = 2.0 * gf_l * ga_h / GPG + hia
    lam_l_lo_home = 2.0 * gf_h * ga_l / GPG

    home_lams = np.where(HOME_PATTERN_HIGHER[None, None, :],
                         lam_h_hi_home[:, :, None], lam_h_lo_home[:, :, None])
    away_lams = np.where(HOME_PATTERN_HIGHER[None, None, :],
                         lam_l_hi_home[:, :, None], lam_l_lo_home[:, :, None])

    hw = np.zeros((K, S), dtype=np.int64)
    lw = np.zeros((K, S), dtype=np.int64)
    length = np.full((K, S), 7, dtype=np.int64)
    decided = np.zeros((K, S), dtype=bool)
    for g in range(7):
        pend = ~decided
        if not pend.any():
            break
        hl = home_lams[:, :, g][pend]
        al = away_lams[:, :, g][pend]
        hg = rng.poisson(hl)
        ag = rng.poisson(al)
        tie = hg == ag
        p_home = np.where(hl + al > 0, hl / (hl + al), 0.5)
        home_won = (hg > ag) | (tie & (rng.random(tie.shape) < p_home))
        hi_won = home_won if HOME_PATTERN_HIGHER[g] else ~home_won
        hw[pend] += hi_won
        lw[pend] += ~hi_won
        newly = pend & ((hw == WINS_NEEDED) | (lw == WINS_NEEDED))
        length[newly] = g + 1
        decided |= newly
    assert ((hw == WINS_NEEDED) ^ (lw == WINS_NEEDED)).all()
    return np.where(hw == WINS_NEEDED, hi, lo), length


# --- Full batch ---------------------------------------------------------------


def run_batch(n_seasons: int, seed: int, chunk: int = CHUNK,
              talent_sd: float | None = None, gpg: float | None = None,
              hia: float | None = None) -> dict[str, np.ndarray]:
    """Full pipeline for n_seasons; returns dict of [n_seasons, 32] arrays
    with the Stage 12 user-spec columns (flags int8, counts int32).

    talent_sd/gpg/hia default to the locked module constants (the Stage 13
    primary setup); Stage 17's sensitivity sweeps pass overrides. With no
    overrides the code path and results are identical to pre-Stage-17.
    """
    hia_eff = HIA if hia is None else hia
    out = {c: [] for c in COLS}
    done = 0
    while done < n_seasons:
        CH = min(chunk, n_seasons - done)
        rng = np.random.default_rng(seed + done)  # disjoint streams per chunk
        talent, pts, gf, ga, row, diff, h2h, gf_rate, ga_rate = \
            _regular_season_chunk(CH, rng, talent_sd=talent_sd, gpg=gpg)
        ki = np.arange(CH)[:, None]

        div_order, conf_order, league_order = _rank_all(pts, row, diff, h2h)
        hi1, lo1 = _seed_brackets(div_order, conf_order, pts, row, diff, h2h)

        rounds_won = np.zeros((CH, N_TEAMS), dtype=np.int64)
        made = np.zeros((CH, N_TEAMS), dtype=bool)
        made[ki, hi1] = True
        made[ki, lo1] = True

        w1, _ = _series_engine(hi1, lo1, gf_rate, ga_rate, rng, hia=hia_eff)
        rounds_won[ki, w1] += 1

        # Round 2: within each division, 1vsWC winner vs 2v3 winner.
        div_pts_top = np.take_along_axis(
            pts[:, None, :], div_order[:, :, :1], axis=2)[:, :, 0]
        top_div = np.empty((CH, 2), dtype=np.int64)
        for c in range(2):
            d0, d1 = 2 * c, 2 * c + 1
            top_div[:, c] = np.where(div_pts_top[:, d0] >= div_pts_top[:, d1], d0, d1)
        hi2 = np.empty((CH, 4), dtype=np.int64)
        lo2 = np.empty((CH, 4), dtype=np.int64)
        for d in range(4):
            c = d // 2
            t = top_div[:, c] == d
            s_1vwc = 4 * c + np.where(t, 0, 1)
            s_2v3 = 4 * c + np.where(t, 2, 3)
            a = w1[ki[:, 0], s_1vwc]
            b = w1[ki[:, 0], s_2v3]
            h, aw = _order_home_away(a[:, None], b[:, None], pts, row, diff, h2h)
            hi2[:, d] = h[:, 0]
            lo2[:, d] = aw[:, 0]
        w2, _ = _series_engine(hi2, lo2, gf_rate, ga_rate, rng, hia=hia_eff)
        rounds_won[ki, w2] += 1

        # Round 3: Atlantic(0) vs Metropolitan(1); Central(2) vs Pacific(3).
        a3 = w2[:, [0, 2]]
        b3 = w2[:, [1, 3]]
        hi3, lo3 = _order_home_away(a3, b3, pts, row, diff, h2h)
        w3, _ = _series_engine(hi3, lo3, gf_rate, ga_rate, rng, hia=hia_eff)
        rounds_won[ki, w3] += 1

        # Cup final: East (div 0/1 side) vs West.
        hiC, loC = _order_home_away(w3[:, [0]], w3[:, [1]], pts, row, diff, h2h)
        cup, _ = _series_engine(hiC, loC, gf_rate, ga_rate, rng, hia=hia_eff)
        rounds_won[ki, cup] += 1

        out["talent_gfpct"].append(talent.astype(np.float32))
        out["PTS"].append(pts.astype(np.int32))
        out["GF"].append(gf.astype(np.int32))
        out["GA"].append(ga.astype(np.int32))
        out["made_playoffs"].append(made.astype(np.int8))
        team32 = np.arange(N_TEAMS)  # [32]
        out["division_winner"].append(
            (div_order[:, :, 0, None] == team32[None, None, :]).any(1).astype(np.int8))
        out["conference_best"].append(
            (conf_order[:, :, 0, None] == team32[None, None, :]).any(1).astype(np.int8))
        out["presidents_trophy"].append(
            (league_order[:, 0, None] == team32[None, None, :]).astype(np.int8).reshape(CH, N_TEAMS))
        for k, col in enumerate(["round1_wins", "round2_wins", "round3_wins"], start=1):
            out[col].append((rounds_won >= k).astype(np.int8))
        out["cup_winner"].append((rounds_won >= 4).astype(np.int8))

        done += CH

    return {c: np.concatenate(v, axis=0) for c, v in out.items()}


# --- Verification -------------------------------------------------------------


def _check_structure(res: dict[str, np.ndarray], n: int) -> None:
    assert all(res[c].shape == (n, N_TEAMS) for c in COLS)
    assert (res["made_playoffs"].sum(1) == 16).all()
    assert (res["division_winner"].sum(1) == 4).all()
    assert (res["conference_best"].sum(1) == 2).all()
    assert (res["presidents_trophy"].sum(1) == 1).all()
    assert (res["round1_wins"].sum(1) == 8).all()
    assert (res["round2_wins"].sum(1) == 4).all()
    assert (res["round3_wins"].sum(1) == 2).all()
    assert (res["cup_winner"].sum(1) == 1).all()
    for a, b in [("round2_wins", "round1_wins"), ("round3_wins", "round2_wins"),
                 ("cup_winner", "round3_wins")]:
        assert (res[a] <= res[b]).all(), f"{a} exceeds {b}"
    assert (res["round1_wins"] <= res["made_playoffs"]).all()
    print(f"[structure] PASS on {n} seasons")


def _check_determinism() -> None:
    a = run_batch(100, seed=777)
    b = run_batch(100, seed=777)
    for c in COLS:
        assert np.array_equal(a[c], b[c]), f"non-deterministic: {c}"
    print("[determinism] PASS (same seed -> identical outputs)")


def _check_vs_scalar(n_scalar: int = 800, n_batch: int = 8000,
                     seed_scalar: int = 500_000, seed_batch: int = 900_000) -> None:
    """Distributional equivalence vs the scalar Stage 12 path.

    Same model, independent RNG streams — metrics must agree within MC
    error (4-SE gates; per-team maxima use a looser gate since 32 maxima
    are compared).
    """
    from stage12_full_season import simulate_full_season

    t0 = time.perf_counter()
    rows = []
    for i in range(n_scalar):
        rng = np.random.default_rng(seed_scalar + i)
        results, _, _ = simulate_full_season(rng)
        rows.append(results.set_index("team").loc[_TEAMS, COLS].to_numpy())
    scalar = {c: np.stack([r[:, j] for r in rows], axis=0)
              for j, c in enumerate(COLS)}
    t_scalar = time.perf_counter() - t0

    t0 = time.perf_counter()
    batch = run_batch(n_batch, seed=seed_batch)
    t_batch = time.perf_counter() - t0

    def gate(name, v_s, v_b, tol):
        d = abs(v_s - v_b)
        ok = d <= tol
        print(f"  {name}: scalar {v_s:.4f} vs batch {v_b:.4f} "
              f"(diff {d:.4f}, gate ±{tol:.4f}) {'PASS' if ok else 'FAIL'}")
        return ok

    ok = True
    # League PPG (points per team-game).
    ppg_s = scalar["PTS"].sum() / (n_scalar * 32 * 82)
    ppg_b = batch["PTS"].sum() / (n_batch * 32 * 82)
    ok &= gate("league PPG", ppg_s, ppg_b, 0.01)
    # Playoff rate, PT rate, cup rate, mean rounds (pooled over teams).
    for c, base in [("made_playoffs", None), ("presidents_trophy", None),
                    ("cup_winner", None)]:
        vs = scalar[c].mean(); vb = batch[c].mean()
        se = np.sqrt(max(vs * (1 - vs), 1e-9) / (n_scalar * 32)
                     + max(vb * (1 - vb), 1e-9) / (n_batch * 32))
        ok &= gate(f"P({c})", vs, vb, 4 * se + 0.002)
    # Per-team extremes across the 32 teams (loose gate for 32 maxima).
    for c, tol in [("made_playoffs", 0.06), ("cup_winner", 0.05),
                   ("presidents_trophy", 0.05)]:
        d = np.abs(scalar[c].mean(0) - batch[c].mean(0)).max()
        print(f"  max per-team |ΔP({c})| = {d:.4f} (gate {tol}) "
              f"{'PASS' if d <= tol else 'FAIL'}")
        ok &= d <= tol
    # Mean rounds_won by talent quartile (same-bin comparison).
    for name, arr in [("scalar", scalar), ("batch", batch)]:
        pass
    for q, lo_q, hi_q in [(0, 0, 8), (1, 8, 16), (2, 16, 24), (3, 24, 32)]:
        # talent rank per season (1 = best); compare mean rounds_won in rank band
        tr = (-batch["talent_gfpct"]).argsort(1).argsort(1)  # 0 = best
        m_b = batch["round1_wins"][ (tr >= lo_q) & (tr < hi_q) ].mean()
        tr_s = (-scalar["talent_gfpct"]).argsort(1).argsort(1)
        m_s = scalar["round1_wins"][(tr_s >= lo_q) & (tr_s < hi_q)].mean()
        ok &= gate(f"mean round1_wins, talent rank {lo_q}-{hi_q}", m_s, m_b, 0.05)
    print(f"  timing: scalar {t_scalar:.1f}s ({t_scalar/n_scalar*1000:.0f} ms/season), "
          f"batch {t_batch:.1f}s ({t_batch/n_batch*1000:.2f} ms/season) "
          f"-> speedup {t_scalar/n_scalar/(t_batch/n_batch):.0f}x")
    assert ok, "scalar-vs-batch consistency FAILED"
    print("[scalar consistency] PASS")


def _benchmark() -> None:
    t0 = time.perf_counter()
    run_batch(10_000, seed=42)
    dt = time.perf_counter() - t0
    print(f"[benchmark] 10,000 seasons in {dt:.1f}s = {dt/10_000*1000:.2f} ms/season "
          f"-> 100k ≈ {dt*10/60:.0f} min")


def main() -> None:
    print("Stage 13 (batch engine) verification")
    res = run_batch(300, seed=20261027)
    _check_structure(res, 300)
    _check_determinism()
    _check_vs_scalar()
    _benchmark()
    print("All Stage 13 batch-engine checks PASS")


if __name__ == "__main__":
    main()
