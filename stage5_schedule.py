"""Stage 5 — ingest the 2025-26 regular-season schedule.

Fetches club schedules for all 32 teams from the NHL API
(https://api-web.nhle.com/v1/club-schedule-season/{team}/20252026),
keeps regular-season games (gameType 2), and writes the deduplicated
schedule table.

Output: data/processed/schedule_20252026.csv
        columns: game_id, game_date, away_abbrev, home_abbrev
Report:  outputs/reports/stage5_schedule_report.txt

Verification (in-script gate):
- 1,212 games (32 teams x 82 / 2) under the current format
- every team exactly 82 games, 41 home + 41 away
- meeting counts consistent with the current schedule format
  (each team: 3 games vs 8 opponents... per current matrix; we report
  the observed distribution for user review rather than hard-assert
  the full matrix, but assert total pairs and home/away splits)
"""
import csv
import json
import os
import time
import urllib.request

BASE_URL = "https://api-web.nhle.com/v1"
SEASON = "20252026"
TEAMS = ["ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
         "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI",
         "NYR", "OTT", "PHI", "PIT", "SJS", "SEA", "STL", "TBL", "TOR",
         "UTA", "VAN", "VGK", "WPG", "WSH"]
HEADERS = {"User-Agent": "hockey-lab-likens/0.1 (research; contact: local user)"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
REPORT = os.path.join(ROOT, "outputs", "reports", "stage5_schedule_report.txt")

EXPECTED_GAMES = 1312  # 32 x 82 / 2 (plan's 1,212 was an arithmetic slip, fixed 2026-10-08)


def http_get_json(url, retries=4):
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            print(f"  attempt {attempt} failed for {url}: {e}", flush=True)
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2


def main():
    os.makedirs(PROCESSED, exist_ok=True)
    games = {}
    errors = []
    for i, team in enumerate(TEAMS, 1):
        try:
            sched = http_get_json(f"{BASE_URL}/club-schedule-season/{team}/{SEASON}")
        except Exception as e:
            errors.append((team, str(e)))
            print(f"  schedule {i}/32: {team} FAILED: {e}", flush=True)
            continue
        n = 0
        for g in sched.get("games", []):
            if g.get("gameType") != 2:
                continue
            gid = g["id"]
            home = g["homeTeam"]["abbrev"]
            away = g["awayTeam"]["abbrev"]
            if gid not in games:
                games[gid] = {
                    "game_id": gid,
                    "game_date": g["gameDate"],
                    "away_abbrev": away,
                    "home_abbrev": home,
                }
                n += 1
        print(f"  schedule {i}/32: {team} ok, {n} new games "
              f"({len(games)} total)", flush=True)
        time.sleep(0.35)

    rows = sorted(games.values(), key=lambda r: (r["game_date"], r["game_id"]))
    out_path = os.path.join(PROCESSED, f"schedule_{SEASON}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game_id", "game_date",
                                          "away_abbrev", "home_abbrev"])
        w.writeheader()
        w.writerows(rows)

    # ---- verification
    lines = [f"Stage 5 schedule report — season {SEASON}",
             f"source: {BASE_URL}/club-schedule-season/{{team}}/{SEASON}",
             f"team schedule errors: {len(errors)} {errors if errors else ''}",
             f"unique regular-season games: {len(rows)} (expected {EXPECTED_GAMES})"]
    from collections import Counter
    home_c = Counter(r["home_abbrev"] for r in rows)
    away_c = Counter(r["away_abbrev"] for r in rows)
    total_c = Counter(home_c) + Counter(away_c)
    bad = {t: (total_c[t], home_c[t], away_c[t]) for t in TEAMS
           if not (total_c[t] == 82 and home_c[t] == 41 and away_c[t] == 41)}
    lines.append(f"teams with 82 games (41 home / 41 away): "
                 f"{32 - len(bad)}/32" + (f" VIOLATIONS: {bad}" if bad else ""))
    # meeting counts: pairs -> (away@home + home@away)
    pair = Counter()
    for r in rows:
        a, h = sorted((r["away_abbrev"], r["home_abbrev"]))
        pair[(a, h)] += 1
    dist = Counter(pair.values())
    lines.append(f"pair meeting-count distribution (games per pair): "
                 f"{dict(sorted(dist.items()))}")
    # sanity: total = sum of pair counts; each team's sum = 82 (covered above)
    lines.append(f"sum over pairs: {sum(pair.values())}")
    # date range + future games still scheduled (gameState-independent: we
    # kept gameType 2 regardless of state — report states observed)
    dates = sorted({r["game_date"] for r in rows})
    lines.append(f"date range: {dates[0]} .. {dates[-1]} "
                 f"({len(dates)} distinct dates)")

    ok = (len(rows) == EXPECTED_GAMES and not errors and not bad)
    lines.append("RESULT: " + ("CLEAN" if ok else "INCOMPLETE — rerun"))
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
