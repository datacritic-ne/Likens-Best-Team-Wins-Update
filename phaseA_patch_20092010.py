#!/usr/bin/env python
"""
Phase A patch — repair the 34 missing 2009-10 goals in data/processed/games_20092010.csv.

Background (see notes/phaseA_bookmark.md):
  The NHL API play-by-play for exactly 30 games of 2009-10 (all regular season) is
  missing goal plays upstream (34 goals total). The /boxscore endpoint DOES have
  complete per-player goal counts, but no EN flag. This script repairs the CSV by
  pulling each affected game's hockey-reference boxscore scoring table (single-page
  ad-hoc fetches with a generous delay — HR annotated "EN" was verified to match the
  NHL API's null-goalie EN rule exactly on 2009020004), matching the missing goals
  against the boxscore scorer set, and patching gf_clean / EN audit columns.

  Each repaired game must satisfy the project invariant afterwards:
      final score == clean GF + EN goals + SO-decider goals (per side).

  NO goals are inserted into the PBP raw JSON cache (data/raw/...) — the repair is
  at the processed-CSV level only, with a full audit trail:
      outputs/reports/phaseA_patch_report.txt
      notes/phaseA_missing_goal_repair_detail.csv
"""
import csv
import html as H
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = Path("/workspace/projects/hockey-lab/analyses/likens")
SEASON_CSV = BASE / "data/processed/games_20092010.csv"
MISSING_CSV = BASE / "notes/phaseA_missing_goal_games.csv"
REPORT = BASE / "outputs/reports/phaseA_patch_report.txt"
DETAIL = BASE / "notes/phaseA_missing_goal_repair_detail.csv"

HR_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
DELAY_S = 8.0  # responsible-crawl delay; 30 requests -> ~2.5 min

class RateLimited(Exception):
    pass

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

def load_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "hockey-lab research fetch"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def hr_scoring_rows(url):
    """Parse the HR boxscore scoring table -> list of (period, time, team, strength, scorer_cell)."""
    page = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": HR_UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                page = r.read().decode("utf-8", errors="ignore")
            break
        except Exception as e:
            log(f"  HR fetch attempt {attempt+1} failed ({e}); backoff...")
            time.sleep(10 * (attempt + 1))
    if page is None:
        raise RateLimited(f"HR fetch failed after retries: {url}")
    i = page.find('id="scoring"')
    if i < 0:
        raise ValueError(f"no scoring table at {url}")
    j = page.find("</table>", i)
    tbl = page[i:j]
    # Track period section headers
    out = []
    period = None
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
        cells = [" ".join(H.unescape(re.sub(r"<[^>]+>", " ", c)).split()) for c in cells]
        if not any(cells):
            continue
        if len(cells) == 1:  # section header like '2nd Period' / 'Shootout'
            txt = cells[0]
            if txt.lower().startswith("shootout"):
                period = "SO"
            else:
                period = txt
            continue
        if len(cells) >= 4 and re.match(r"^\d+:\d\d$", cells[0]):
            tm, strength, scorer = cells[1], cells[2], cells[3]
            if strength not in ("PP", "SH", "EN", "PS"):
                strength = "EV"
            # match team abbrev + scorer name
            m = re.match(r"^(.*?)\s*\(\d+\)$", scorer)
            name = m.group(1) if m else scorer
            out.append({"period": period, "time": cells[0], "team": tm,
                        "strength": strength, "scorer": name})
    return out

def api_pbp_goals(gid):
    pbp = load_json(f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play")
    return [p for p in pbp["plays"] if p["typeDescKey"] == "goal"]

def api_boxscore_scorers(gid):
    """{side: [(playerId, name, goals)]} plus final scores and team abbrevs."""
    bx = load_json(f"https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore")
    res = {}
    for side in ("awayTeam", "homeTeam"):
        scorers = []
        for grp in ("forwards", "defense"):
            for p in bx["playerByGameStats"][side][grp]:
                if p["goals"]:
                    name = p["name"]["default"] if isinstance(p.get("name"), dict) else str(p.get("name"))
                    scorers.append((p["playerId"], name, p["goals"]))
        res[side] = {"team": bx[side]["abbrev"], "score": bx[side]["score"], "scorers": scorers}
    return res

import unicodedata
def name_key(name):
    """Normalize a name for cross-source matching: 'Josh Gorges' vs 'J. Gorges', accents folded."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    parts = re.sub(r"[^.A-Za-z\- ']", "", name).replace(".", " ").split()
    parts = [p for p in parts if p.lower() not in ("jr", "sr")]
    if not parts:
        return ""
    last = parts[-1].lower()
    first = parts[0][0].lower() if parts[0] else ""
    return f"{first}|{last}"

def main():
    if "--fetch" not in sys.argv:
        log("NOTE: run with --fetch to hit hockey-reference; otherwise this prints plan only.")
    season = {}
    with open(SEASON_CSV) as f:
        for r in csv.DictReader(f):
            season[int(r["game_id"])] = r
    missing = list(csv.DictReader(open(MISSING_CSV)))
    log(f"Affected games: {len(missing)}, total missing goals: {sum(int(m['missing']) for m in missing)}")

    detail_rows = []
    patched = 0
    start = int(sys.argv[sys.argv.index("--start")+1]) if "--start" in sys.argv else 0
    for idx, m in enumerate(missing):
        if idx < start:
            continue
        gid = int(m["game_id"])
        row = season[gid]
        log(f"\n== {gid} {m['game_date']} {m['away_abbrev']}@{m['home_abbrev']} "
            f"final {m['away_score']}-{m['home_score']} missing {m['missing']}")
        if "--fetch" not in sys.argv:
            continue
        hr_url = f"https://www.hockey-reference.com/boxscores/{m['game_date'].replace('-', '')}0{m['home_abbrev']}.html"
        log(f"  URL: {hr_url}")
        try:
            hr = hr_scoring_rows(hr_url)
        except RateLimited:
            log(f"\nRATE-LIMITED at game index {idx} ({gid}). Resume with: --fetch --start {idx}")
            REPORT.write_text("\n".join(log_lines) + "\n")
            sys.exit(3)
        pbp = api_pbp_goals(gid)
        bx = api_boxscore_scorers(gid)
        time.sleep(DELAY_S)
        n_so_deciders = sum(1 for p in pbp if p["periodDescriptor"].get("periodType") == "SO")

        # PBP goal scorer id sets by side
        away_id = row["away_abbrev"]
        home_id = row["home_abbrev"]
        # map API team ids -> abbrevs via boxscore
        away_abbrev = bx["awayTeam"]["team"]
        home_abbrev = bx["homeTeam"]["team"]
        assert (away_abbrev, home_abbrev) == (away_id, home_id), f"abbrev mismatch {gid}"

        # HR goals excluding shootout section
        hr_reg = [g for g in hr if not str(g["period"]).lower().startswith("shootout")
                  and "ot" not in str(g["period"]).lower().replace("st ot", "ot") or True]
        # simpler: exclude shootout rows by period string
        hr_reg = [g for g in hr if "shootout" not in str(g["period"]).lower()]
        # OT-period rows are real goals; only the Shootout section (SO attempts) is excluded.
        assert len(hr_reg) == int(m["away_score"]) + int(m["home_score"]) - n_so_deciders, \
            f"{gid}: HR reg goals {len(hr_reg)} != final minus SO deciders ({m['away_score']}-{m['home_score']}, SO {n_so_deciders})"

        # PBP scorers by side (ids)
        pbp_ids = {"awayTeam": [], "homeTeam": []}
        for p in pbp:
            side = "homeTeam" if p["details"]["eventOwnerTeamId"] == bx["homeTeam"].get("id") else "awayTeam"
            # eventOwnerTeamId compare needs team ids; fall back: match by scorer id in box scorer lists
            pbp_ids[side].append(p["details"]["scoringPlayerId"])
        # Safer: classify by membership in boxscore scorer lists
        away_ids = {s[0] for s in bx["awayTeam"]["scorers"]}
        home_ids = {s[0] for s in bx["homeTeam"]["scorers"]}
        pbp_away = [p["details"]["scoringPlayerId"] for p in pbp
                    if p["details"]["scoringPlayerId"] in away_ids]
        pbp_home = [p["details"]["scoringPlayerId"] for p in pbp
                    if p["details"]["scoringPlayerId"] in home_ids]

        # side labels for the CSV: away_abbrev/home_abbrev
        def side_of(abbrev):
            return "away" if abbrev == away_abbrev else "home"

        # per-side missing counts
        miss_by_side = {}
        for abbrev, key in ((away_abbrev, "awayTeam"), (home_abbrev, "homeTeam")):
            hr_n = sum(1 for g in hr_reg if g["team"] == abbrev)
            api_n = len(pbp_away if key == "awayTeam" else pbp_home)
            miss_by_side[abbrev] = hr_n - api_n
            log(f"  {abbrev}: HR {hr_n} vs PBP {api_n} -> missing {hr_n - api_n}")
        assert sum(miss_by_side.values()) == int(m["missing"]), f"{gid}: side split mismatch"

        # identify missing scorer names: boxscore scorer counts minus PBP scorer counts, per side
        missing_goals = []
        for abbrev, key in ((away_abbrev, "awayTeam"), (home_abbrev, "homeTeam")):
            bx_counts = {pid: n for pid, name, n in bx[key]["scorers"]}
            pbp_counts = {}
            for pid in (pbp_away if key == "awayTeam" else pbp_home):
                pbp_counts[pid] = pbp_counts.get(pid, 0) + 1
            for pid, n in bx_counts.items():
                excess = n - pbp_counts.get(pid, 0)
                name = next(nm for p_, nm, _ in bx[key]["scorers"] if p_ == pid)
                if excess > 0:
                    missing_goals.append({"abbrev": abbrev, "playerId": pid, "name": name, "n": excess})
        assert sum(g["n"] for g in missing_goals) == int(m["missing"]), f"{gid}: scorer excess mismatch"

        # match each missing goal to an HR row: unmatched HR rows for that team, name-key match
        hr_left = list(hr_reg)
        repairs = []
        for g in missing_goals:
            for k in range(g["n"]):
                nk = name_key(g["name"])
                cand = [h for h in hr_left if h["team"] == g["abbrev"] and name_key(h["scorer"]) == nk]
                if len(cand) != 1:
                    log(f"  WARN {gid}: {g['name']} -> {len(cand)} HR candidates")
                if cand and any(h["strength"] == "EN" for h in cand) and not all(h["strength"] == "EN" for h in cand):
                    # mixed EN/EN-ambiguity: prefer the EN row only if this is the player's only excess goal and PBP EN flag missing
                    log(f"  NOTE {gid}: {g['name']} has mixed-strength candidates; strengths={[h['strength'] for h in cand]}")
                h = cand[0] if cand else None
                hr_left.remove(h)
                repairs.append((g, h))

        # EN-ness from HR strength flag
        en_by_side = {"away": 0, "home": 0}
        for g, h in repairs:
            is_en = h["strength"] == "EN"
            if h is None:
                raise RuntimeError(f"{gid}: unresolvable missing goal {g}")
            s = side_of(g["abbrev"])
            if is_en:
                en_by_side[s] += 1
            detail_rows.append({
                "game_id": gid, "team": g["abbrev"], "playerId": g["playerId"],
                "player_name": g["name"], "hr_period": h["period"], "hr_time": h["time"],
                "hr_strength": h["strength"], "is_en": int(is_en),
            })
            log(f"  missing -> {g['name']} ({g['abbrev']}) {h['period']} {h['time']} [{h['strength']}]"
                + ("  ** EN" if is_en else ""))

        # patch row: gf_clean excludes EN + SO. Missing goals are all REG/OT (verified: SO-deciders present).
        n_en_a, n_en_h = en_by_side["away"], en_by_side["home"]
        if n_en_a or n_en_h:
            row["away_en_goals"] = str(int(row["away_en_goals"]) + n_en_a)
            row["home_en_goals"] = str(int(row["home_en_goals"]) + n_en_h)
        row["away_gf_clean"] = str(int(row["away_score"]) - int(row["away_en_goals"]) - int(row["away_so_goals"]))
        row["home_gf_clean"] = str(int(row["home_score"]) - int(row["home_en_goals"]) - int(row["home_so_goals"]))
        row["away_ga_clean"] = row["home_gf_clean"]
        row["home_ga_clean"] = row["away_gf_clean"]
        row["n_goal_plays"] = str(int(row["n_goal_plays"]) + int(m["missing"]))
        patched += 1
        # invariant check for this game (Stage 2 SO convention)
        is_so = row["last_period_type"] == "SO"
        home_winner = int(row["home_score"]) > int(row["away_score"])
        for s in ("away", "home"):
            so_dec = 1 if is_so and (s == "home") == home_winner else 0
            if is_so:
                assert int(row[f"{s}_score"]) == int(row[f"{s}_gf_clean"]) + int(row[f"{s}_en_goals"]) + so_dec, \
                    f"{gid}: invariant fail {s}"
            else:
                assert int(row[f"{s}_score"]) == int(row[f"{s}_gf_clean"]) + int(row[f"{s}_en_goals"]) + int(row[f"{s}_so_goals"]), \
                    f"{gid}: invariant fail {s}"

    if "--fetch" not in sys.argv:
        log("DRY RUN done (no network, no writes).")
        return

    # write patched CSV + verify all 18 seasons
    with open(SEASON_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(season[2009020001].keys()))
        w.writeheader()
        for gid in sorted(season):
            w.writerow(season[gid])
    with open(DETAIL, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        for r in detail_rows:
            w.writerow(r)
    log(f"\nPatched {patched} games; detail rows: {len(detail_rows)}")

    # full invariant sweep, all 18 seasons
    # Convention (Stage 2): final score = gf_clean + EN goals + (1 SO decider to the winner if SO game).
    # For non-SO games so_goals column is 0. In SO games the decider is attributed via last_period_type.
    log("\n== Invariant sweep (all 18 seasons) ==")
    import glob
    bad = 0
    for path in sorted(glob.glob(str(BASE / "data/processed/games_*.csv"))):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        fails = []
        for r in rows:
            is_so = r["last_period_type"] == "SO"
            home_winner = int(r["home_score"]) > int(r["away_score"])
            ok = True
            for s in ("home", "away"):
                so_dec = 1 if is_so and (s == "home") == home_winner else 0
                expect = int(r[f"{s}_gf_clean"]) + int(r[f"{s}_en_goals"]) + so_dec
                if is_so:
                    if int(r[f"{s}_score"]) != expect:
                        ok = False
                else:
                    if int(r[f"{s}_score"]) != int(r[f"{s}_gf_clean"]) + int(r[f"{s}_en_goals"]) + int(r[f"{s}_so_goals"]):
                        ok = False
            if not ok:
                fails.append(r["game_id"])
        status = "OK" if not fails else f"FAIL {fails[:5]}"
        log(f"  {Path(path).name}: {len(rows)} games {status}")
        bad += len(fails)
    log(f"\nTOTAL invariant failures across all seasons: {bad}")
    if bad:
        sys.exit(1)

    REPORT.write_text("\n".join(log_lines) + "\n")
    log(f"Report written to {REPORT}")

if __name__ == "__main__":
    main()
