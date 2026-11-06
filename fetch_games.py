#!/usr/bin/env python3
"""Stage 2 — Likens analysis: fetch NHL game logs from api-web.nhle.com/v1.

Fetches team schedules to enumerate regular-season game ids, then play-by-play
for each game. Caches raw JSON under data/raw/ (idempotent: reruns only fetch
missing games), and writes one cleaned CSV per season under data/processed/.

Etiquette (user-approved, 2026-10-05):
  - courtesy time.sleep() between calls
  - honest custom User-Agent
  - retry with exponential backoff on HTTP/URL errors; failures recorded
  - per-season completeness report; season must pass before counting as done

Usage:
  python fetch_games.py --season 20052006           # one season
  python fetch_games.py --season 20052006 --limit 5 # smoke test, 5 games
  python fetch_games.py --all                       # every season in SEASONS
"""

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://api-web.nhle.com/v1"
PROJECT = Path(__file__).resolve().parent.parent  # analyses/likens/
RAW_DIR = PROJECT / "data" / "raw"
PROC_DIR = PROJECT / "data" / "processed"
REPORT_DIR = PROJECT / "outputs" / "reports"

DELAY_S = 0.3          # courtesy delay between API calls
RETRY_DELAYS = (2, 4, 8)  # exponential backoff on errors
UA = "likens-replication/0.1 (personal research; python-urllib)"

# Per-era team abbreviations. Era splits verified against the API
# (2026-10-06): /club-schedule-season accepts these exact codes per season.
# Notes:
#  - Anaheim: "Mighty Ducks of Anaheim" in 2005-06, but the API uses ANA for
#    that season too (MDA returns 404), so ANA throughout.
#  - Atlanta Thrashers (ATL) -> Winnipeg Jets (WPG) after 2010-11.
#  - Phoenix Coyotes (PHX) -> Arizona Coyotes (ARI) after 2013-14
#    -> Utah HC (UTA) after 2023-24.
#  - Vegas (VGK) joined 2017-18; Seattle (SEA) joined 2021-22.

def _era(base: list[str], **subs) -> list[str]:
    """Derive an era's team list by renaming (out_abbrev=in_abbrev) or
    adding (add_abbrev=True) teams on a copy of the base list."""
    out = list(base)
    for old, new in subs.items():
        if new is True:
            out.append(old)
        else:
            out[out.index(old)] = new
    return sorted(out)

_TEAMS_BASE_30 = [  # 2005-06 .. 2010-11: ATL era, PHX; 30 teams
    "ANA", "ATL", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
    "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR",
    "OTT", "PHX", "PHI", "PIT", "SJS", "STL", "TBL", "TOR", "VAN", "WSH"]
TEAMS_30_0511 = _TEAMS_BASE_30                                    # ATL, PHX
TEAMS_30_1114 = _era(_TEAMS_BASE_30, ATL="WPG")                   # WPG, PHX
TEAMS_30_1417 = _era(_TEAMS_BASE_30, ATL="WPG", PHX="ARI")        # WPG, ARI
TEAMS_31_1719 = _era(_TEAMS_BASE_30, ATL="WPG", PHX="ARI", VGK=True)
TEAMS_32_2124 = _era(_TEAMS_BASE_30, ATL="WPG", PHX="ARI", VGK=True, SEA=True)
TEAMS_32_2426 = _era(_TEAMS_BASE_30, ATL="WPG", PHX="UTA", VGK=True, SEA=True)

# season -> (team list, expected regular-season games)
SEASONS = {}
for s in range(2005, 2011):   # 2005-06 .. 2010-11: ATL era, PHX
    SEASONS[f"{s}{s+1}"] = (TEAMS_30_0511, 1230)
for s in (2011, 2013):       # 2011-12, 2013-14: WPG era, still PHX
                             # 2012-13 EXCLUDED: lockout, 48-game season (plan §Stage 2: 82-game seasons only)
    SEASONS[f"{s}{s+1}"] = (TEAMS_30_1114, 1230)
for s in range(2014, 2017):   # 2014-15 .. 2016-17: WPG + ARI era
    SEASONS[f"{s}{s+1}"] = (TEAMS_30_1417, 1230)
for s in (2017, 2018):        # 2017-18, 2018-19: VGK joins, 31 teams
    SEASONS[f"{s}{s+1}"] = (TEAMS_31_1719, 1271)
for s in range(2021, 2024):   # 2021-22 .. 2023-24: SEA joins, ARI era
    SEASONS[f"{s}{s+1}"] = (TEAMS_32_2124, 1312)
for s in (2024, 2025):        # 2024-25, 2025-26: Coyotes moved to Utah
    SEASONS[f"{s}{s+1}"] = (TEAMS_32_2426, 1312)

errors_log = []  # (season, game_id, error, retries_used)


def http_get_json(url: str) -> dict | None:
    """GET with honest UA and exponential-backoff retries. None if all fail."""
    for attempt, delay in enumerate((0,) + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                json.JSONDecodeError) as e:
            print(f"    attempt {attempt + 1} failed for {url}: {e}", flush=True)
    return None


def season_game_ids(season: str, teams: list[str]) -> set[int]:
    """Enumerate regular-season game ids from every team's schedule."""
    ids: set[int] = {}
    for i, team in enumerate(teams):
        sched = http_get_json(f"{BASE}/club-schedule-season/{team}/{season}")
        if sched is None:
            raise RuntimeError(f"schedule fetch failed for {team} {season}")
        for g in sched.get("games", []):
            if g.get("gameType") == 2 and g.get("gameState") in ("FINAL", "OFF"):
                ids[g["id"]] = True
        print(f"  schedule {i + 1}/{len(teams)}: {team} ok, "
              f"{len(ids)} unique games so far", flush=True)
        time.sleep(DELAY_S)
    return set(ids)


def fetch_pbp(game_id: int) -> dict | None:
    """Play-by-play with on-disk cache. None if unobtainable after retries."""
    cache = RAW_DIR / str(game_id) / "play-by-play.json"
    if cache.exists():
        return json.loads(cache.read_text())
    data = http_get_json(f"{BASE}/gamecenter/{game_id}/play-by-play")
    if data is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
    return data


def parse_game(game_id: int, pbp: dict) -> dict:
    """Extract one row per game: scores + EN/SO-excluded GF/GA for both teams."""
    goals = [p for p in pbp["plays"] if p.get("typeDescKey") == "goal"]
    home_id, away_id = pbp["homeTeam"]["id"], pbp["awayTeam"]["id"]

    def goals_for(team_id: int, include_en: bool, include_so: bool) -> int:
        n = 0
        for p in goals:
            d = p["details"]
            if d.get("eventOwnerTeamId") != team_id:
                continue
            if p["periodDescriptor"]["periodType"] == "SO" and not include_so:
                continue
            if d.get("goalieInNetId") is None and not include_en:
                continue
            n += 1
        return n

    outcome = pbp.get("gameOutcome") or {}
    return {
        "game_id": game_id,
        "season": pbp.get("season"),
        "game_date": pbp.get("gameDate"),
        "home_abbrev": pbp["homeTeam"].get("abbrev"),
        "away_abbrev": pbp["awayTeam"].get("abbrev"),
        "home_score": pbp["homeTeam"].get("score"),
        "away_score": pbp["awayTeam"].get("score"),
        "last_period_type": outcome.get("lastPeriodType", ""),
        "n_goal_plays": len(goals),
        # cleaned (JLikens-style) GF/GA: exclude empty-net and shootout goals
        "home_gf_clean": goals_for(home_id, include_en=False, include_so=False),
        "away_gf_clean": goals_for(away_id, include_en=False, include_so=False),
        "home_ga_clean": goals_for(away_id, include_en=False, include_so=False),
        "away_ga_clean": goals_for(home_id, include_en=False, include_so=False),
        # EN/SO goal counts for audit
        "home_en_goals": goals_for(home_id, include_en=True, include_so=False) - goals_for(home_id, include_en=False, include_so=False),
        "away_en_goals": goals_for(away_id, include_en=True, include_so=False) - goals_for(away_id, include_en=False, include_so=False),
        "home_so_goals": goals_for(home_id, include_en=False, include_so=True) - goals_for(home_id, include_en=False, include_so=False),
        "away_so_goals": goals_for(away_id, include_en=False, include_so=True) - goals_for(away_id, include_en=False, include_so=False),
    }


def completeness_report(season: str, expected: int, got: int, parsed: list[dict],
                        missing: list[int], elapsed: float) -> bool:
    """Print + save the per-season report. Returns True iff clean."""
    ok = (len(parsed) == got == expected) and not missing and not errors_log
    proc = PROC_DIR / f"games_{season}.csv"
    report = REPORT_DIR / f"stage2_report_{season}.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Stage 2 completeness report — season {season}",
        f"  expected regular-season games : {expected}",
        f"  enumerated via schedules      : {got}",
        f"  play-by-play parsed           : {len(parsed)}",
        f"  missing (enumerated, no PBP)  : {len(missing)} {missing[:10]}",
        f"  http/url errors logged        : {len(errors_log)}",
        f"  elapsed                       : {elapsed / 60:.1f} min",
        f"  processed file                : {proc}",
        f"  RESULT                        : {'CLEAN' if ok else 'INCOMPLETE — rerun this season'}",
    ]
    if errors_log:
        lines.append("  errors:")
        for season_, gid, err, rt in errors_log:
            lines.append(f"    {season_} game {gid}: {err} (retried {rt}x)")
    text = "\n".join(lines)
    print(text, flush=True)
    report.write_text(text + "\n")
    return ok


def run_season(season: str, limit: int | None = None) -> bool:
    errors_log.clear()  # FIX 2026-10-27: reset per season — the module-level
    # log previously accumulated across seasons, so a late --all run could
    # fail every subsequent season's completeness report on stale errors.
    teams, expected = SEASONS[season]
    print(f"=== Season {season}: {len(teams)} teams, expect {expected} games ===", flush=True)
    t0 = time.perf_counter()
    try:
        gids = season_game_ids(season, teams)
    except RuntimeError as e:
        print(f"FATAL: {e}", flush=True)
        return False
    if limit:
        gids = set(sorted(gids)[:limit])
        print(f"  (limit mode: fetching only {len(gids)} games)", flush=True)
    parsed = []
    for i, gid in enumerate(sorted(gids)):
        pbp = fetch_pbp(gid)
        if pbp is None:
            errors_log.append((season, gid, "unobtainable after retries", len(RETRY_DELAYS)))
        else:
            parsed.append(parse_game(gid, pbp))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(gids)} games fetched", flush=True)
        time.sleep(DELAY_S)
    # write processed CSV
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(parsed[0].keys()) if parsed else []
    with open(PROC_DIR / f"games_{season}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(parsed)
    missing = sorted(gids - {p["game_id"] for p in parsed})
    ok = completeness_report(season, expected, len(gids), parsed, missing,
                             time.perf_counter() - t0)
    print(f"=== Season {season} {'OK' if ok else 'INCOMPLETE'} ===", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", help="e.g. 20052006")
    ap.add_argument("--limit", type=int, help="fetch only first N games (smoke test)")
    ap.add_argument("--all", action="store_true", help="run every season")
    args = ap.parse_args()

    if args.all:
        seasons = list(SEASONS)
    elif args.season:
        seasons = [args.season]
    else:
        ap.error("specify --season or --all")
    for s in seasons:
        if s not in SEASONS:
            sys.exit(f"unknown season {s}; known: {sorted(SEASONS)}")
    failures = [s for s in seasons if not run_season(s, args.limit)]
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
