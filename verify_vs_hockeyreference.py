"""Stage 2 verification: match each processed season CSV 1:1 against the
hockey-reference Schedule & Results page for that season.

Match key: (date, away team, home team, away goals, home goals).
Reports per season: HR rows, CSV rows, matched, unmatched in each direction.
Writes a summary to outputs/reports/stage2_hr_verification.txt.
"""
import csv
import os
import re
import sys
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
PROCESSED = os.path.join(ROOT, "data", "processed")
REPORT = os.path.join(ROOT, "outputs", "reports", "stage2_hr_verification.txt")

PAIRS = [
    ("20062007", "https://www.hockey-reference.com/leagues/NHL_2007_games.html"),
    ("20072008", "https://www.hockey-reference.com/leagues/NHL_2008_games.html"),
    ("20082009", "https://www.hockey-reference.com/leagues/NHL_2009_games.html"),
    ("20092010", "https://www.hockey-reference.com/leagues/NHL_2010_games.html"),
    ("20102011", "https://www.hockey-reference.com/leagues/NHL_2011_games.html"),
    ("20112012", "https://www.hockey-reference.com/leagues/NHL_2012_games.html"),
    ("20132014", "https://www.hockey-reference.com/leagues/NHL_2014_games.html"),
    ("20142015", "https://www.hockey-reference.com/leagues/NHL_2015_games.html"),
    ("20152016", "https://www.hockey-reference.com/leagues/NHL_2016_games.html"),
    ("20162017", "https://www.hockey-reference.com/leagues/NHL_2017_games.html"),
    ("20172018", "https://www.hockey-reference.com/leagues/NHL_2018_games.html"),
    ("20182019", "https://www.hockey-reference.com/leagues/NHL_2019_games.html"),
    ("20212022", "https://www.hockey-reference.com/leagues/NHL_2022_games.html"),
    ("20222023", "https://www.hockey-reference.com/leagues/NHL_2023_games.html"),
    ("20232024", "https://www.hockey-reference.com/leagues/NHL_2024_games.html"),
    ("20242025", "https://www.hockey-reference.com/leagues/NHL_2025_games.html"),
    ("20252026", "https://www.hockey-reference.com/leagues/NHL_2026_games.html"),
]

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# hockey-reference team abbreviations are 3-letter; the CSV uses NHL API
# abbreviations. They are identical for all teams in scope, but a couple of
# historical HR codes differ (verified below); map CSV -> HR here.
ABBREV_MAP = {
    "VGK": "VEG",  # NHL API 'VGK' = hockey-reference 'VEG' (Vegas)
}


def parse_hr_games(html):
    """Parse the regular-season games table from a Schedule page.
    Returns list of (date_str_iso, away, home, away_goals, home_goals)."""
    soup = BeautifulSoup(html, "html.parser")
    # The main table has id 'games' (playoffs are in 'games_playoffs').
    table = soup.find("table", id="games")
    if table is None:
        # sometimes wrapped in a comment; search comments
        for c in soup.find_all(string=lambda t: isinstance(t, type(soup.new_string("")))):
            pass
        # fallback: find commented-out table
        from bs4 import Comment
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if "id=\"games\"" in comment:
                csoup = BeautifulSoup(comment, "html.parser")
                table = csoup.find("table", id="games")
                break
    if table is None:
        raise RuntimeError("games table not found")
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        if "thead" in tr.get("class", []):
            continue
        date_cell = tr.find("th", {"data-stat": "date_game"})
        vis = tr.find("td", {"data-stat": "visitor_team_name"})
        vgoals = tr.find("td", {"data-stat": "visitor_goals"})
        home = tr.find("td", {"data-stat": "home_team_name"})
        hgoals = tr.find("td", {"data-stat": "home_goals"})
        if not (date_cell and vis and vgoals and home and hgoals):
            continue
        link = vis.find("a")
        vis_abbr = link["href"].split("/")[2] if link else vis.get_text()
        link = home.find("a")
        home_abbr = link["href"].split("/")[2] if link else home.get_text()
        d = datetime.strptime(date_cell.get_text(), "%Y-%m-%d")
        rows.append((d.strftime("%Y-%m-%d"), vis_abbr, home_abbr,
                     int(vgoals.get_text()), int(hgoals.get_text())))
    return rows


def load_csv_games(season):
    path = os.path.join(PROCESSED, f"games_{season}.csv")
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = datetime.strptime(r["game_date"][:10], "%Y-%m-%d")
            rows.append((d.strftime("%Y-%m-%d"),
                         ABBREV_MAP.get(r["away_abbrev"], r["away_abbrev"]),
                         ABBREV_MAP.get(r["home_abbrev"], r["home_abbrev"]),
                         int(r["away_score"]), int(r["home_score"])))
    return rows


def main():
    out = open(REPORT, "w")
    overall_ok = True
    for season, url in PAIRS:
        t0 = time.time()
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            hr_rows = parse_hr_games(resp.text)
            time.sleep(3)  # courtesy delay between requests
        except Exception as e:
            print(f"{season}: FETCH/PARSE FAILED: {e}", flush=True)
            out.write(f"{season}: FETCH/PARSE FAILED: {e}\n")
            overall_ok = False
            continue
        csv_rows = load_csv_games(season)
        from collections import Counter
        hr_c = Counter(hr_rows)
        csv_c = Counter(csv_rows)
        only_csv = list((csv_c - hr_c).elements())
        only_hr = list((hr_c - csv_c).elements())
        matched = len(csv_rows) - len(only_csv)
        ok = (len(hr_rows) == len(csv_rows) == matched
              and not only_csv and not only_hr)
        if not ok:
            overall_ok = False
        line = (f"{season}: HR {len(hr_rows)} | CSV {len(csv_rows)} | "
                f"matched {matched} | csv-only {len(only_csv)} | "
                f"hr-only {len(only_hr)} | "
                f"{'MATCH' if ok else 'MISMATCH'} ({time.time()-t0:.1f}s)")
        print(line, flush=True)
        out.write(line + "\n")
        for r in only_csv[:10]:
            out.write(f"  CSV ONLY: {r}\n")
        for r in only_hr[:10]:
            out.write(f"  HR  ONLY: {r}\n")
    out.write("OVERALL: " + ("ALL MATCH" if overall_ok else "PROBLEMS FOUND") + "\n")
    out.close()
    print("OVERALL:", "ALL MATCH" if overall_ok else "PROBLEMS FOUND", flush=True)


if __name__ == "__main__":
    main()
