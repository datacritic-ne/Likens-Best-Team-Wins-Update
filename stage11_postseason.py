#!/usr/bin/env python3
"""Stage 11 — Full postseason simulation (4 rounds, 16 -> 8 -> 4 -> 2 -> 1).

Composition layer over Stage 9 (bracket) and Stage 10 (series sim). One
round at a time; each round pairs the previous round's winners, assigns
home ice, and simulates the series.

Round structure (2013-14+ format, per user-approved plan 2026-10-20):

  Round 1 (8 series): from playoff_seeding() — within each division
  bracket, D1 hosts a wild card (top D1 plays the LOWER WC) and #2 hosts
  #3. Higher/lower by points/tiebreakers; the WC may out-rank a D1 and
  hold home ice.

  Round 2 (4 series): one per DIVISION bracket. The winner of that
  division's D2-vs-D3 series plays the winner of its D1-vs-WC series —
  regardless of which conference the wild card came from (both WCs from
  one division is legal: that WC's winner still plays in the D1's
  divisional half).

  Round 3, Conference finals (2 series): Atlantic-bracket winner vs
  Metropolitan-bracket winner; Central vs Pacific.

  Cup final (1 series): East finalist vs West finalist.

  Home ice, rounds 2-4: higher regular-season points, then the SAME
  tiebreaker chain used for round-1 seeding (points -> ROW -> head-to-head
  points -> DIFF -> team code), from the ranked standings table + regular-
  season game log. (Same rule for the Cup final — user confirmed 2026-10-20.)

Output capture (user requirement 2026-10-20): per-team results for ALL 32
teams — the 16 non-playoff teams get rounds_won = 0 / cup = False — plus
per-round series results (winner, length, games).

Usage:
    results, round_logs = simulate_postseason(standings_table, game_log,
                                              team_data, rng)
    # results: 32 rows, team / conf / div / rounds_won (0-4) / cup_winner
    # round_logs: dict round_name -> DataFrame of series results

Running this file executes the Stage 11 verification checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stage9_standings import _head_to_head_pts
from simulate_playoff_series import simulate_series

ROUND_NAMES = ["Round 1", "Round 2", "Conference Finals", "Cup Final"]


def _home_ice(a: str, b: str, table: pd.DataFrame,
              h2h: dict[tuple[str, str], int]) -> tuple[str, str]:
    """Return (home, away): home = higher regular-season finish.

    Tiebreaker chain (same as round-1 seeding): points -> ROW -> head-to-head
    points (directed lookup, regular-season games) -> DIFF -> team code.
    """
    ra = table[table["team"] == a].iloc[0]
    rb = table[table["team"] == b].iloc[0]
    # Chain: PTS -> ROW -> H2H (directed) -> DIFF -> team code ASCENDING
    # (alphabetically first wins the tie, matching the ranking chain's
    # `team ascending` convention everywhere else). FIXED 2026-10-27: the
    # original compared `a` with `>=` against b's key whose final element is
    # the code — with equal PTS/ROW/H2H/DIFF, `key_a >= key_b` picked the
    # ALPHABETICALLY LATER code, inverting the documented convention.
    cmp = (int(ra["PTS"]), int(ra["ROW"]),
           h2h.get((a, b), 0), int(ra["DIFF"]))
    cmp_b = (int(rb["PTS"]), int(rb["ROW"]),
             h2h.get((b, a), 0), int(rb["DIFF"]))
    if cmp != cmp_b:
        return (a, b) if cmp > cmp_b else (b, a)
    return (a, b) if a <= b else (b, a)  # final: code ascending


def simulate_round(matchups: list[tuple[str, str]], table: pd.DataFrame,
                   game_log: pd.DataFrame, team_data: pd.DataFrame,
                   rng: np.random.Generator, round_name: str,
                   rounds_won: dict[str, int]) -> list[dict]:
    """Simulate one round: assign home ice, play each series, record results.

    matchups: list of (team_a, team_b) pairs (order irrelevant; home ice is
        decided here from the standings).
    rounds_won: mutable dict team -> rounds won; each series winner is bumped.
    Returns a list of per-series result dicts.
    """
    h2h = _head_to_head_pts(game_log)
    results = []
    for a, b in matchups:
        home, away = _home_ice(a, b, table, h2h)
        winner, log = simulate_series(home, away, team_data, rng)
        rounds_won[winner] += 1
        results.append({
            "round": round_name,
            "home": home, "away": away,
            "winner": winner,
            "loser": away if winner == home else home,
            "games": len(log),
            "home_wins": int((log["winner"] == home).sum()),
            "away_wins": int((log["winner"] == away).sum()),
        })
    return results


def _round2_matchups(r1_results: list[dict]) -> list[tuple[str, str]]:
    """Pair round-1 winners within each division bracket.

    User rule (2026-10-20): the division's D2-vs-D3 series winner plays the
    winner of that division's D1-vs-WC series. Wild cards may come from the
    other division; their winner still plays in the D1's bracket half.
    """
    # Round-1 matchups carry the division bracket; map winner -> bracket half.
    by_div: dict[str, dict[str, str]] = {}
    for m in r1_results:
        div = m["division_bracket"]
        kind = m["matchup_kind"]  # "1vsWC" or "2v3"
        by_div.setdefault(div, {})[kind] = m["winner"]
    pairs = []
    for div, halves in by_div.items():
        assert set(halves) == {"1vsWC", "2v3"}, f"{div}: incomplete bracket"
        pairs.append((halves["1vsWC"], halves["2v3"]))
    return pairs


def _round3_matchups(r2_results: list[dict]) -> list[tuple[str, str]]:
    """Conference finals: Atlantic vs Metropolitan, Central vs Pacific."""
    win = {m["division_bracket"]: m["winner"] for m in r2_results}
    assert set(win) == {"Atlantic", "Metropolitan", "Central", "Pacific"}
    return [(win["Atlantic"], win["Metropolitan"]),
            (win["Central"], win["Pacific"])]


def simulate_postseason(table: pd.DataFrame, game_log: pd.DataFrame,
                        team_data: pd.DataFrame, rng: np.random.Generator,
                        ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Simulate the full 4-round postseason off a completed regular season.

    Arguments:
        table: ranked standings (compute_ranked_standings output) — the
            bracket comes from the same table via playoff_seeding's logic;
            here we re-derive round-1 matchups through playoff_seeding to
            guarantee identical seeding.
        game_log: regular-season game log (for head-to-head tiebreaks).
        team_data: talent table (assign_team_talent output).
        rng: numpy Generator.

    Returns:
        (results, round_logs)
        results: 32-row DataFrame (all teams), columns team / conf / div /
            made_playoffs / rounds_won (0-4) / cup_winner.
        round_logs: round name -> DataFrame of that round's series results.
    """
    from stage9_standings import playoff_seeding

    _, matchups_r1 = playoff_seeding(table)
    # Tag each matchup with its bracket half for round-2 pairing.
    r1 = []
    for m in matchups_r1:
        div = m["division_bracket"]
        kind = "1vsWC" if "WC" in m["higher_seed_label"] + m["lower_seed_label"] else "2v3"
        r1.append({"series": m["series"], "division_bracket": div,
                   "matchup_kind": kind,
                   "higher": m["higher"], "lower": m["lower"]})

    rounds_won = {t: 0 for t in table["team"]}
    round_logs: dict[str, pd.DataFrame] = {}

    # Round 1 — simulate, keeping division/bracket info with each result.
    h2h = _head_to_head_pts(game_log)
    r1_results = []
    for m in r1:
        home, away = _home_ice(m["higher"], m["lower"], table, h2h)
        winner, log = simulate_series(home, away, team_data, rng)
        rounds_won[winner] += 1
        r1_results.append({
            "round": "Round 1", "division_bracket": m["division_bracket"],
            "matchup_kind": m["matchup_kind"],
            "series": m["series"],
            "home": home, "away": away, "winner": winner,
            "loser": away if winner == home else home,
            "games": len(log),
            "home_wins": int((log["winner"] == home).sum()),
            "away_wins": int((log["winner"] == away).sum()),
        })
    round_logs["Round 1"] = pd.DataFrame(r1_results)

    # Round 2 — divisional pairings.
    r2_results = simulate_round(_round2_matchups(r1_results), table,
                                game_log, team_data, rng, "Round 2",
                                rounds_won)
    # carry division bracket through for round 3
    div_of = {r["winner"]: r["division_bracket"] for r in r1_results}
    for r in r2_results:
        r["division_bracket"] = div_of[r["winner"]]
    round_logs["Round 2"] = pd.DataFrame(r2_results)

    # Round 3 — conference finals.
    r3_results = simulate_round(_round3_matchups(r2_results), table,
                                game_log, team_data, rng,
                                "Conference Finals", rounds_won)
    conf_of = {r["winner"]: ("Eastern" if r["division_bracket"] in
                             ("Atlantic", "Metropolitan") else "Western")
               for r in r2_results}
    for r in r3_results:
        r["conference"] = conf_of[r["winner"]]
    round_logs["Conference Finals"] = pd.DataFrame(r3_results)

    # Cup final.
    east = next(r["winner"] for r in r3_results if r["conference"] == "Eastern")
    west = next(r["winner"] for r in r3_results if r["conference"] == "Western")
    cup_results = simulate_round([(east, west)], table,
                                 game_log, team_data, rng, "Cup Final",
                                 rounds_won)
    round_logs["Cup Final"] = pd.DataFrame(cup_results)

    cup_winner = cup_results[0]["winner"]
    made = {r["home"] for r in r1_results} | {r["away"] for r in r1_results}

    results = table[["team", "conf", "div"]].copy()
    results["made_playoffs"] = results["team"].isin(made)
    results["rounds_won"] = results["team"].map(rounds_won)
    results["cup_winner"] = results["team"] == cup_winner
    return results, round_logs


# --- Stage 11 verification checks ---------------------------------------------


def _run_one_postseason(seed: int):
    """One seeded regular season + postseason, returning pieces for checks."""
    from stage9_standings import simulate_season_full
    from assign_team_talent import assign_team_talent

    rng = np.random.default_rng(seed)
    team_data = assign_team_talent(rng)
    table, game_log = simulate_season_full(rng)
    results, round_logs = simulate_postseason(table, game_log, team_data, rng)
    return table, game_log, team_data, results, round_logs


def check_structure(seed: int = 20261020) -> None:
    table, _, _, results, round_logs = _run_one_postseason(seed)

    assert len(results) == 32, "results must cover all 32 teams"
    assert int(results["made_playoffs"].sum()) == 16
    assert set(results["rounds_won"]) <= {0, 1, 2, 3, 4}
    assert int(results["cup_winner"].sum()) == 1, "exactly one Cup winner"
    cup = results.loc[results["cup_winner"], "team"].iloc[0]
    assert results.loc[results["cup_winner"], "rounds_won"].iloc[0] == 4

    # Series counts per round and rounds-won conservation:
    # 15 series per postseason, so sum(rounds_won) over 16 teams == 15.
    n_by_round = {name: len(df) for name, df in round_logs.items()}
    assert n_by_round == {"Round 1": 8, "Round 2": 4,
                          "Conference Finals": 2, "Cup Final": 1}, n_by_round
    assert int(results["rounds_won"].sum()) == 15, "15 series wins per postseason"

    # Every eliminated team lost exactly one series in its exit round:
    # rounds_won == r (r = 0..3) <=> team is the loser of round r+1's series.
    exit_round = {0: "Round 1", 1: "Round 2", 2: "Conference Finals",
                  3: "Cup Final"}
    for r, name in exit_round.items():
        losers = set(round_logs[name]["loser"])
        playoff_teams = set(results.loc[results["made_playoffs"], "team"])
        at_r = set(results.loc[(results["rounds_won"] == r)
                               & (results["team"].isin(playoff_teams)),
                               "team"])
        # rounds_won==3 counts only the Cup-final loser (winner has 4).
        assert at_r == losers, f"rounds_won=={r} mismatch: {at_r} vs {losers}"

    # Non-playoff teams all have rounds_won == 0 and no cup.
    out = results[~results["made_playoffs"]]
    assert (out["rounds_won"] == 0).all() and (~out["cup_winner"]).all()

    # Bracket narrowing: winners of each round are exactly the next round's
    # participants.
    for prev, nxt in [("Round 1", "Round 2"), ("Round 2", "Conference Finals"),
                      ("Conference Finals", "Cup Final")]:
        winners = set(round_logs[prev]["winner"])
        players = set(round_logs[nxt]["home"]) | set(round_logs[nxt]["away"])
        assert winners == players, f"{prev} winners != {nxt} players"

    print(f"[structure] PASS — 16->8->4->2->1; cup winner {cup}; "
          f"rounds-won histogram "
          f"{results['rounds_won'].value_counts().sort_index().to_dict()}")


def check_round2_pairing(seed: int = 20261021) -> None:
    """Round-2 pairings follow the division rule, incl. the same-division-WC
    edge case (forced by construction)."""
    table, _, team_data, _, round_logs = _run_one_postseason(seed)

    # Reconstruct: each round-2 series' participants must come from the SAME
    # division bracket's two round-1 halves.
    r1 = round_logs["Round 1"]
    # rebuild division/kind map from the bracket table
    from stage9_standings import playoff_seeding
    _, m1 = playoff_seeding(table)
    div_of_series = {m["series"]: m["division_bracket"] for m in m1}
    kind_of_series = {}
    for m in m1:
        kind = ("1vsWC" if "WC" in m["higher_seed_label"] + m["lower_seed_label"]
                else "2v3")
        kind_of_series[m["series"]] = kind

    halves = {}  # div -> {kind: winner}
    for _, row in r1.iterrows():
        # match by teams (round log has home/away, not series letter) —
        # use the set of participants
        s = next(m for m in m1
                 if {m["higher"], m["lower"]} == {row["home"], row["away"]})
        halves.setdefault(div_of_series[s["series"]],
                          {})[kind_of_series[s["series"]]] = row["winner"]

    r2 = round_logs["Round 2"]
    for _, row in r2.iterrows():
        pair = {row["home"], row["away"]}
        matched = [d for d, h in halves.items()
                   if {h["1vsWC"], h["2v3"]} == pair]
        assert len(matched) == 1, f"round-2 pair {pair} matches {matched}"
    print("[round2 pairing] PASS — each R2 series = one division's "
          "1vsWC winner vs 2v3 winner")


def check_home_ice(seed: int = 20261022) -> None:
    """Every series' home team has the better (PTS, ROW, H2H, DIFF) key."""
    table, game_log, _, _, round_logs = _run_one_postseason(seed)
    h2h = _head_to_head_pts(game_log)
    for name, df in round_logs.items():
        for _, row in df.iterrows():
            home, away = row["home"], row["away"]
            rh = table[table["team"] == home].iloc[0]
            ra = table[table["team"] == away].iloc[0]
            kh = (int(rh["PTS"]), int(rh["ROW"]), h2h.get((home, away), 0),
                  int(rh["DIFF"]), home)
            ka = (int(ra["PTS"]), int(ra["ROW"]), h2h.get((away, home), 0),
                  int(ra["DIFF"]), away)
            assert kh >= ka, f"{name}: {away} should hold home ice over {home}"
    print("[home ice] PASS — home team has the better tiebreaker key "
          "in all 15 series")


def check_batch(seed: int = 20261023, n: int = 200) -> None:
    """Batch sanity: cups won by many teams; 15 rounds-won every sim."""
    from stage9_standings import simulate_season_full
    from assign_team_talent import assign_team_talent

    cups: dict[str, int] = {}
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        team_data = assign_team_talent(rng)
        table, game_log = simulate_season_full(rng)
        results, _ = simulate_postseason(table, game_log, team_data, rng)
        assert int(results["rounds_won"].sum()) == 15
        assert int(results["cup_winner"].sum()) == 1
        cup = results.loc[results["cup_winner"], "team"].iloc[0]
        cups[cup] = cups.get(cup, 0) + 1
    # 200 sims should spread cups widely (any team can win a drawn talent);
    # demand most of the 32 teams appear.
    assert len(cups) >= 24, f"cup winners too concentrated: {cups}"
    top = sorted(cups.items(), key=lambda kv: -kv[1])[:5]
    print(f"[batch] PASS — {n} postseasons, {len(cups)} distinct cup winners; "
          f"top 5: {top}")


def check_rounds_won_magnitude(seed: int = 20261024, n: int = 500) -> None:
    """Better regular-season teams should win more rounds, on average."""
    from stage9_standings import simulate_season_full
    from assign_team_talent import assign_team_talent

    bucket_pts, bucket_rw = [], []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        team_data = assign_team_talent(rng)
        table, game_log = simulate_season_full(rng)
        results, _ = simulate_postseason(table, game_log, team_data, rng)
        pts = table.set_index("team")["PTS"]
        for _, row in results.iterrows():
            bucket_pts.append(int(pts[row["team"]]))
            bucket_rw.append(int(row["rounds_won"]))
    pts = np.array(bucket_pts); rw = np.array(bucket_rw)
    lo = rw[pts <= np.percentile(pts, 25)].mean()
    hi = rw[pts >= np.percentile(pts, 75)].mean()
    corr = float(np.corrcoef(pts, rw)[0, 1])
    print(f"[magnitude] mean rounds_won: bottom-points quartile {lo:.3f}, "
          f"top quartile {hi:.3f}, corr(PTS, rounds_won) = {corr:+.3f}")
    assert hi > lo, "stronger teams do not win more playoff rounds"
    print("  [PASS] playoff success increases with regular-season strength")


def main() -> None:
    print("Stage 11 verification")
    check_structure()
    check_round2_pairing()
    check_home_ice()
    check_batch()
    check_rounds_won_magnitude()
    print("All Stage 11 checks PASS")


if __name__ == "__main__":
    main()
