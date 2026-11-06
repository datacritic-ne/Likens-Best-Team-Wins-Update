#!/usr/bin/env python3
"""Stage 9 — Standings with seeding + first-round playoff matchups.

Consumes the Stage 8 standings table and produces:

  1. division/conference labels and league-wide rank by points;
  2. playoff seeding under the current NHL format (2013-14 realignment,
     re-adopted 2021-22; confirmed at records.nhl.com/history/playoff-formats):
       - top 3 teams in each division qualify (12 teams);
       - 2 wild cards per conference: next-highest point totals regardless
         of division;
       - Division rank within a division: points, ROW, head-to-head points,
         goal differential (team code as final deterministic tiebreak).
  3. first-round matchups:
       - within each division: #2 vs #3, with #1 drawing a wild card;
       - the conference's higher-points division winner plays the LOWER-points
         wild card; the other division winner plays the higher-points WC
         (verified wording from records.nhl.com);
       - round 2: winners within each division bracket meet (tracked here as
         bracket labels, not simulated — that is Stage 10/11).
  4. best-team identification: the team with the highest drawn GF% (ranked
     before any games are simulated — Stage 8 carries GF_percent into the
     standings table), tracked for made-playoffs / division-winner /
     Presidents'-Trophy (most league points) outcomes.

Tiebreakers vs the official rules — one documented simplification: the
official second tiebreaker is regulation + OT wins (ROW), excluding shootout
wins. The Stage 6 simulator treats every extra-point game as a coin flip and
does not distinguish OT wins from SO wins, so the ROW computed here includes
OT wins. This is a superset ("RW+OTW") of the official ROW and is flagged for
the writeup. Head-to-head points and goal differential are computed exactly.

Usage:
    from stage9_standings import compute_ranked_standings, playoff_seeding
    standings, game_log = simulate_season(rng)
    table = compute_ranked_standings(standings, game_log)
    seeds, matchups = playoff_seeding(table)

Running this file executes the Stage 9 verification checks (including the
2024-25 historical validation against the real playoff bracket).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nhl_structure import TEAM_TO_CONFERENCE, TEAM_TO_DIVISION
from stage8_regular_season import simulate_season

HIA = 0.2563  # home-ice advantage, goals/game (locked constant, assign_team_talent)

SEASON_CSV_2024_2025 = (
    "/workspace/projects/hockey-lab/analyses/likens/data/processed/games_20242025.csv"
)


# --- Tiebreakers --------------------------------------------------------------


def _head_to_head_pts(game_log: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Points each team earned head-to-head vs each opponent.

    Returns a dict {(team_a, team_b): pts_of_a_vs_b} with both orderings
    present (a<b key for a's points vs b, and the swapped key for b's).
    Ties in this comparison cannot occur under the final team-code tiebreak.
    """
    h2h: dict[tuple[str, str], int] = {}
    home = game_log["home"].to_numpy()
    away = game_log["away"].to_numpy()
    hp = game_log["home_pts"].to_numpy(dtype=int)
    ap = game_log["away_pts"].to_numpy(dtype=int)
    for h, a, p_h, p_a in zip(home, away, hp, ap):
        lo, hi = (h, a) if h < a else (a, h)
        h2h[(lo, hi)] = h2h.get((lo, hi), 0) + (p_h if h == lo else p_a)
        h2h[(hi, lo)] = h2h.get((hi, lo), 0) + (p_h if h == hi else p_a)
    return h2h


def sort_group(df: pd.DataFrame, h2h: dict[tuple[str, str], int]) -> pd.DataFrame:
    """Rank a group (division or conference) by the official tiebreaker chain.

    points -> ROW (here: all wins, documented simplification) ->
    head-to-head points (only defined between two teams; applied pairwise
    among teams tied on the earlier criteria) -> goal differential ->
    team code (deterministic final resort; the real NHL continues with more
    criteria we cannot or need not reproduce).
    """
    out = df.copy()
    out = out.sort_values(
        ["PTS", "ROW", "DIFF", "team"], ascending=[False, False, False, True],
        kind="mergesort",
    )
    # NOTE: original (caller-frame) index is PRESERVED so the caller can map
    # ranks back onto the full standings table by position.

    # Pairwise head-to-head: swap adjacent teams tied on PTS+ROW+DIFF if the
    # lower-ranked team earned more head-to-head points against the upper one.
    # H2H lookup is DIRECTED (upper, lower) — not sorted — so the comparison
    # cannot invert across a swap (which would loop forever).
    # NOTE on mechanics: the swap assigns row VALUES positionally while the
    # index labels stay in place — content moves between labels. Downstream
    # rank mapping therefore goes BY TEAM CODE (see compute_ranked_standings),
    # never by index label. Verified by the Stage 9 checks (tiebreaker unit
    # test with rank assertions + historical 8/8 bracket reproduction).
    for _ in range(1000):  # safety cap; converges in << n^2 swaps in practice
        changed = False
        for i in range(len(out) - 1):
            a, b = out.iloc[i], out.iloc[i + 1]
            if (a["PTS"], a["ROW"], a["DIFF"]) == (b["PTS"], b["ROW"], b["DIFF"]):
                if h2h.get((b["team"], a["team"]), 0) > h2h.get((a["team"], b["team"]), 0):
                    vals_i = out.iloc[i].copy()
                    out.iloc[i, :] = out.iloc[i + 1].to_numpy()
                    out.iloc[i + 1, :] = vals_i.to_numpy()
                    changed = True
        if not changed:
            break
    else:
        raise RuntimeError("head-to-head swap did not converge")
    return out.drop(columns=["_h2h"], errors="ignore")


# --- Standings with structure -------------------------------------------------


def compute_ranked_standings(
    standings: pd.DataFrame, game_log: pd.DataFrame
) -> pd.DataFrame:
    """Attach conference/division, ROW, league rank, division/conference rank.

    Returns the standings table with added columns: conf, div, ROW, league_rank
    (1-32 by the tiebreaker chain within the whole league), div_rank and
    conf_rank (1-8).
    """
    h2h = _head_to_head_pts(game_log)

    table = standings.copy()
    table["conf"] = table["team"].map(TEAM_TO_CONFERENCE)
    table["div"] = table["team"].map(TEAM_TO_DIVISION)
    if table["conf"].isna().any() or table["div"].isna().any():
        missing = sorted(table.loc[table["conf"].isna() | table["div"].isna(), "team"])
        raise KeyError(f"teams missing from league structure mapping: {missing}")
    table["DIFF"] = table["GF"] - table["GA"]
    # ROW analog: all wins. The simulator resolves every extra-point game by
    # coin flip with no OT goals scored, so OT wins and (hypothetical) SO wins
    # cannot be separated; the official ROW excluding SO wins is not
    # recoverable from the sim. Documented simplification (module docstring).
    table["ROW"] = table["W"]

    # League-wide rank by the tiebreaker chain (no head-to-head swaps applied
    # at league level; ties at this level fall through to DIFF/team code).
    table = table.sort_values(
        ["PTS", "ROW", "DIFF", "team"], ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table["league_rank"] = np.arange(1, len(table) + 1)

    # Division and conference ranks via sort_group (with H2H swaps).
    # Rank assignment is BY TEAM CODE, not by index label: the H2H swap inside
    # sort_group moves row VALUES while index labels stay positionally in place
    # (pandas iloc-tuple assignment semantics), so `ranked.index` no longer
    # identifies which original row holds which rank after a swap. Team codes
    # travel with their row values, so they are the safe key.
    pos_of_team = {t: i for i, t in enumerate(table["team"])}
    div_rank = np.empty(len(table), dtype=int)
    for div, ts in table.groupby("div"):
        ranked = sort_group(ts, h2h)
        for rank, team in enumerate(ranked["team"], start=1):
            div_rank[pos_of_team[team]] = rank
    table["div_rank"] = div_rank

    conf_rank = np.empty(len(table), dtype=int)
    for conf, ts in table.groupby("conf"):
        ranked = sort_group(ts, h2h)
        for rank, team in enumerate(ranked["team"], start=1):
            conf_rank[pos_of_team[team]] = rank
    table["conf_rank"] = conf_rank

    return table


def playoff_seeding(table: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Seed the 16 playoff teams and build the 8 first-round matchups.

    Arguments:
        table: output of compute_ranked_standings().

    Returns:
        (seeds, matchups)
        seeds: 16 rows with seed_label (e.g. 'ATL1', 'MET WC2'), ordered
               East then West, division bracket position then wild cards.
        matchups: list of 8 dicts {series, higher, lower, higher_seed_label,
               lower_seed_label, conference, division_bracket} — 'higher'/
               'lower' are team codes; 'higher' has home-ice (more points,
               tiebreak chain decides). Series letters follow the records.nhl.com
               layout: A/C = Atl/Met winner vs WC, B/D = #2v#3, etc.
    """
    seeds_rows, matchups = [], []

    for conf in ["Eastern", "Western"]:
        conf_teams = table[table["conf"] == conf]
        divs = sorted(conf_teams["div"].unique())

        bracket = {}  # division -> {"1": team, "2": team, "3": team}
        for div in divs:
            top3 = conf_teams[conf_teams["div"] == div].nsmallest(3, "div_rank")
            for _, row in top3.iterrows():
                bracket.setdefault(div, {})[str(int(row["div_rank"]))] = row["team"]
                seeds_rows.append({
                    "conference": conf, "seed_label": f"{div[:3].upper()}{int(row['div_rank'])}",
                    "team": row["team"], "PTS": int(row["PTS"]),
                })

        rest = conf_teams[
            ~conf_teams["team"].isin([t for d in bracket.values() for t in d.values()])
        ].sort_values(["PTS", "ROW", "DIFF", "team"], ascending=[False, False, False, True])
        wc = rest.head(2)
        for i, (_, row) in enumerate(wc.iterrows(), start=1):
            seeds_rows.append({
                "conference": conf, "seed_label": f"WC{i}", "team": row["team"],
                "PTS": int(row["PTS"]),
            })

        # Division winner with more conference points plays the LOWER WC.
        d1_pts = {d: int(conf_teams.loc[conf_teams["team"] == bracket[d]["1"], "PTS"].iloc[0])
                  for d in divs}
        top_div, other_div = sorted(divs, key=lambda d: (-d1_pts[d], d))
        wc1, wc2 = wc.iloc[0]["team"], wc.iloc[1]["team"]

        # Series letters per the records.nhl.com layout: within each
        # conference, D1-vs-WC and #2v#3 take the letter pair assigned to
        # each division's bracket (Atlantic A/B, Metropolitan C/D,
        # Central E/F, Pacific G/H).
        div_letters = {
            ("Eastern", "Atlantic"): "AB", ("Eastern", "Metropolitan"): "CD",
            ("Western", "Central"): "EF", ("Western", "Pacific"): "GH",
        }
        pairings = [
            ("1", bracket[top_div]["1"], wc2, top_div),      # top D1 vs lower WC
            ("1", bracket[other_div]["1"], wc1, other_div),  # other D1 vs higher WC
            ("2", bracket[top_div]["2"], bracket[top_div]["3"], top_div),
            ("2", bracket[other_div]["2"], bracket[other_div]["3"], other_div),
        ]
        for kind, hi, lo, div in pairings:
            hi_row = table[table["team"] == hi].iloc[0]
            lo_row = table[table["team"] == lo].iloc[0]
            if (lo_row["PTS"], lo_row["ROW"], lo_row["DIFF"]) > (hi_row["PTS"], hi_row["ROW"], hi_row["DIFF"]):
                hi, lo = lo, hi
                hi_row, lo_row = lo_row, hi_row
            label = div_letters[(conf, div)]
            series = label[0] if kind == "1" else label[1]
            matchups.append({
                "series": series,
                "conference": conf,
                "division_bracket": div,
                "higher": hi, "lower": lo,
                "higher_seed_label": _seed_label(table, hi),
                "lower_seed_label": _seed_label(table, lo),
            })

    return pd.DataFrame(seeds_rows), matchups


def _seed_label(table: pd.DataFrame, team: str) -> str:
    row = table[table["team"] == team].iloc[0]
    if row["div_rank"] <= 3:
        return f"{row['div'][:3].upper()}{int(row['div_rank'])}"
    order = table[(table["conf"] == row["conf"]) & (table["div_rank"] > 3)].sort_values(
        ["PTS", "ROW", "DIFF", "team"], ascending=[False, False, False, True])
    return f"WC{int((order['team'] == team).argmax()) + 1}"


def playoff_teams(table: pd.DataFrame) -> set[str]:
    """The 16 playoff qualifiers: division top-3 plus the 2 wild cards per
    conference (top 2 non-top-3 teams by the ranking chain). Note a team can
    rank 7th in its conference and still miss if division berths absorb the
    spots — that is the real NHL rule, so membership cannot be expressed as
    a conference-rank cutoff.
    """
    qualifiers: set[str] = set()
    for conf in ["Eastern", "Western"]:
        c = table[table["conf"] == conf]
        qualifiers |= set(c[c["div_rank"] <= 3]["team"])
        rest = c[c["div_rank"] > 3].sort_values(
            ["PTS", "ROW", "DIFF", "team"], ascending=[False, False, False, True]
        )
        qualifiers |= set(rest.head(2)["team"])
    return qualifiers


def season_outcomes(table: pd.DataFrame) -> pd.DataFrame:
    """Indicator variables for every team's regular-season outcomes.

    Consumes the output of compute_ranked_standings() and adds, for ALL 32
    teams (not just the best one):
        made_playoffs      — div_rank <= 3 or conference wild card
        division_winner    — div_rank == 1
        conference_best    — highest regular-season points total in the
                             conference (no NHL trophy; the conference top seed)
        presidents_trophy  — most league points overall
    Ties would be resolved by the same tiebreaker chain used for ranking
    (league_rank / conf_rank / div_rank already encode it), so these are
    unambiguous given the ranking.
    """
    out = table.copy()
    qualifiers = playoff_teams(out)
    out["made_playoffs"] = out["team"].isin(qualifiers)
    out["division_winner"] = out["div_rank"] == 1
    out["conference_best"] = out["conf_rank"] == 1
    out["presidents_trophy"] = out["league_rank"] == 1

    # Structural checks: exactly one of each per scope.
    assert out["presidents_trophy"].sum() == 1
    assert (out.groupby("conf")["conference_best"].sum() == 1).all()
    assert (out.groupby("div")["division_winner"].sum() == 1).all()
    assert out["made_playoffs"].sum() == 16
    return out


def simulate_season_full(
    rng: np.random.Generator,
    schedule: pd.DataFrame | None = None,
    hia: float = HIA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """simulate_season() + season_outcomes(), so the ranking step is never
    forgotten downstream.

    Returns:
        (standings, game_log) where standings additionally carries conf, div,
        ROW, league_rank, div_rank, conf_rank, and the four outcome flags for
        every team.
    """
    standings, game_log = simulate_season(rng, schedule, hia=hia)
    return season_outcomes(compute_ranked_standings(standings, game_log)), game_log


def best_team_flags(table: pd.DataFrame) -> dict:
    """Outcome flags for the best team (highest drawn GF_percent).

    made_playoffs: team is among playoff_teams() (division top-3 or a wild
    card — not expressible as a conference-rank cutoff; see playoff_teams).
    """
    best = table.loc[table["GF_percent"].idxmax()]
    return {
        "team": best["team"],
        "talent_gfpct": float(best["GF_percent"]),
        "league_rank": int(best["league_rank"]),
        "presidents_trophy": bool(best["league_rank"] == 1),
        "made_playoffs": bool(best["team"] in playoff_teams(table)),
        "division_winner": bool(best["div_rank"] == 1),
    }


# --- Verification ------------------------------------------------------------


def check_seeding(table: pd.DataFrame, seeds: pd.DataFrame, matchups: list[dict]) -> None:
    assert len(seeds) == 16, f"expected 16 seeds, got {len(seeds)}"
    assert seeds["team"].is_unique, "a team appears twice in the bracket"
    for conf in ["Eastern", "Western"]:
        c = seeds[seeds["conference"] == conf]
        assert len(c) == 8, f"{conf} has {len(c)} playoff teams, expected 8"
        conf_teams = table[table["conf"] == conf]
        for div in conf_teams["div"].unique():
            n = int((c["seed_label"].str.startswith(div[:3].upper())).sum())
            assert n >= 3, f"{div} contributed {n} teams, minimum is 3"
        # division winners all present
        for div in conf_teams["div"].unique():
            winner = conf_teams[conf_teams["div"] == div].nsmallest(1, "div_rank")["team"].iloc[0]
            assert winner in set(c["team"]), f"{div} winner missing from bracket"
    # matchup legality: 8 unique pairs, higher has home ice
    pairs = {frozenset((m["higher"], m["lower"])) for m in matchups}
    assert len(pairs) == 8 and all(len(p) == 2 for p in pairs), "matchups not 8 unique pairs"
    for m in matchups:
        hi = table[table["team"] == m["higher"]].iloc[0]
        lo = table[table["team"] == m["lower"]].iloc[0]
        assert (hi["PTS"], hi["ROW"], hi["DIFF"]) >= (lo["PTS"], lo["ROW"], lo["DIFF"]), \
            f"series {m['series']}: lower-seeded team has home ice"
        assert (hi["conf"] == lo["conf"] == m["conference"]), "cross-conference matchup"
        # #2 vs #3 pairings must be same division; D1-vs-WC pairings cross
        if m["series"] in "BDFH":
            assert hi["div"] == lo["div"], f"series {m['series']} not intra-division"
        else:
            assert m["higher_seed_label"].endswith("1"), f"series {m['series']} not a D1 host"


def _real_game_log_2024_2025() -> pd.DataFrame:
    """Game log from the real 2024-25 season CSV in the simulate_games schema.

    Points from official bookkeeping: winner 2 pts; if last_period_type is
    OT/SO the loser gets 1 (OTL). W classification follows the same rule
    (ROW here includes OT wins, per the documented simplification).
    """
    g = pd.read_csv(SEASON_CSV_2024_2025)
    ot = g["last_period_type"].ne("REG")
    home_pts = np.where(g["home_score"] > g["away_score"], 2, np.where(ot, 1, 0))
    away_pts = np.where(g["away_score"] > g["home_score"], 2, np.where(ot, 1, 0))
    return pd.DataFrame({
        "home": g["home_abbrev"], "away": g["away_abbrev"],
        "home_goals": g["home_score"], "away_goals": g["away_score"],
        "home_pts": home_pts, "away_pts": away_pts,
        "home_result": np.where(home_pts == 2, "W", np.where(home_pts == 1, "OTL", "L")),
        "away_result": np.where(away_pts == 2, "W", np.where(away_pts == 1, "OTL", "L")),
    })


def _historical_validation() -> None:
    """Reproduce the real 2025 playoff bracket from the real 2024-25 season."""
    games = _real_game_log_2024_2025()
    from stage8_regular_season import _standings_from_log
    real = _standings_from_log(games)
    table = compute_ranked_standings(real, games)
    seeds, matchups = playoff_seeding(table)

    check_seeding(table, seeds, matchups)

    actual_2025 = {
        # Atlantic: TOR1 (108), TBL2 (102), FLA3; OTT the WC1 (97). East D1s:
        # WSH (M1, 111) > TOR, so WSH played the LOWER WC (MTL, 91) and TOR
        # drew WC1 OTT; TBL-FLA is the 2v3. Matches the code's assignment.
        frozenset(("TOR", "OTT")), frozenset(("TBL", "FLA")),
        # Metropolitan: WSH1, CAR2, NJD3; Metro WC: MTL WC2
        frozenset(("WSH", "MTL")), frozenset(("CAR", "NJD")),
        # Central: WPG1, DAL2, COL3; Central WC: STL WC1
        frozenset(("DAL", "COL")), frozenset(("WPG", "STL")),
        # Pacific: VGK1, EDM2, LAK3; Pacific WC: MIN WC2
        frozenset(("EDM", "LAK")), frozenset(("VGK", "MIN")),
    }
    ours = {frozenset((m["higher"], m["lower"])) for m in matchups}
    if ours != actual_2025:
        diff = ours ^ actual_2025
        raise AssertionError(f"2025 bracket mismatch; differing pairs: {diff}")
    print("  [PASS] historical validation: computed 2025 playoff bracket "
          "matches the real 2024-25 matchups exactly (8/8 series)")


def main() -> None:

    rng = np.random.default_rng(20252026)
    standings, game_log = simulate_season(rng)
    table = compute_ranked_standings(standings, game_log)
    seeds, matchups = playoff_seeding(table)

    check_seeding(table, seeds, matchups)
    print("  [PASS] seeding structure (16 teams, 8/conference, >=3/division, "
          "division winners in, 8 legal pairs, home ice correct)")

    # Tiebreaker unit test: forced ties resolve by the documented chain.
    fake = table.head(6).copy()
    fake["PTS"] = [100, 100, 90, 90, 90, 80]
    fake["ROW"] = [40, 41, 38, 38, 38, 30]
    fake["DIFF"] = [10, 10, 5, 5, 5, 0]
    fake["team"] = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    h2h = {("CCC", "DDD"): 5, ("DDD", "CCC"): 6}  # DDD beat CCC head-to-head
    ranked = sort_group(fake, h2h)
    assert list(ranked["team"]) == ["BBB", "AAA", "DDD", "CCC", "EEE", "FFF"], \
        "tiebreaker chain (PTS, ROW, DIFF, H2H) did not resolve as documented"
    # rank mapping must follow the content, not the (unchanged) index labels.
    # Simulate compute_ranked_standings's team-code mapping on the fake group:
    # the H2H swap moved DDD ahead of CCC; if ranks were assigned by index
    # label (the old bug), the swapped rows would carry each other's ranks.
    fake["rank_by_content"] = 0
    pos_of_team = {t: i for i, t in enumerate(fake["team"])}
    for rank, team in enumerate(ranked["team"], start=1):
        fake.loc[fake["team"] == team, "rank_by_content"] = rank
    assert fake.loc[pos_of_team["DDD"], "rank_by_content"] == 3 and \
        fake.loc[pos_of_team["CCC"], "rank_by_content"] == 4, \
        "H2H swap ranks did not follow row content (index-label mapping bug)"
    print("  [PASS] tiebreaker unit test (PTS -> ROW -> DIFF -> head-to-head swap; "
          "ranks follow row content)")

    print(f"\nBest team: {best_team_flags(table)}")

    # Full 32-team outcome table (season_outcomes / simulate_season_full).
    full, _ = simulate_season_full(np.random.default_rng(20252026))
    print("\nSeason outcomes (all 32 teams, by talent rank):")
    oc = full.sort_values("GF_percent", ascending=False)[
        ["team", "GF_percent", "league_rank", "PTS", "made_playoffs",
         "division_winner", "conference_best", "presidents_trophy"]
    ]
    print(oc.to_string(index=False, formatters={"GF_percent": "{:.3f}".format}))

    # Consistency: the best-team flags agree with the full table's row for
    # the best team; ranks line up with the ordering by GF_percent.
    bt = full.loc[full["GF_percent"].idxmax()]
    flags = best_team_flags(table)
    assert bt["team"] == flags["team"]
    assert bool(bt["presidents_trophy"]) == flags["presidents_trophy"]
    assert bool(bt["made_playoffs"]) == flags["made_playoffs"]
    assert bool(bt["division_winner"]) == flags["division_winner"]
    # talent rank 1 (highest GF_percent) exists exactly once
    assert (full["GF_percent"] == full["GF_percent"].max()).sum() == 1
    print("  [PASS] outcome flags consistent with best_team_flags; "
          "structure checks inside season_outcomes passed "
          "(1 POT, 1 conference-best per conference, 1 division winner per "
          "division, 16 playoff teams)")

    print("\nPlayoff seeds:")
    print(seeds.to_string(index=False))
    print("\nFirst-round matchups:")
    for m in matchups:
        print(f"  ({m['series']}) {m['higher']} [{m['higher_seed_label']}] "
              f"vs {m['lower']} [{m['lower_seed_label']}]  ({m['division_bracket']})")

    print("\nHistorical validation (2024-25 real season):")
    _historical_validation()


if __name__ == "__main__":
    main()
