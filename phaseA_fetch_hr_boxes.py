"""Phase A repair v5: fetch hockey-reference boxscores for the 30 affected 2009-10 games.
Rewritten with requests.Session (curl+subprocess path kept hitting unexplained 404s).
Validates per-team goal counts against final scores. 5s courtesy delay (project rule 6)."""
import json
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = '/workspace/projects/hockey-lab/analyses/likens'
GAMES_CSV = f'{BASE}/notes/phaseA_missing_goal_games.csv'
OUT_JSON = f'{BASE}/notes/phaseA_hr_scoring_events.json'
UA = 'Mozilla/5.0 (X11; Linux x86_64) hockey-lab research script'

session = requests.Session()
session.headers['User-Agent'] = UA


def parse_scoring(html):
    soup = BeautifulSoup(html.replace('<!--', '').replace('-->', ''), 'html.parser')
    t = soup.find('table', id='scoring')
    if t is None:
        return []
    out = []
    for row in t.find_all('tr'):
        if row.find('th') and 'Period' in row.get_text():
            continue
        cells = [c.get_text(' ', strip=True) for c in row.find_all('td')]
        if len(cells) >= 4:
            out.append(cells)
    return out


def main():
    games = pd.read_csv(GAMES_CSV).set_index('game_id')
    rows = []
    for gid, g in games.iterrows():
        date = g['game_date'].replace('-', '')
        url = f'https://www.hockey-reference.com/boxscores/{date}{g["home_abbrev"]}.html'
        code, events = None, []
        for attempt in range(5):
            try:
                r = session.get(url, timeout=30)
                code = str(r.status_code)
                if code == '200':
                    events = parse_scoring(r.text)
                    break
            except Exception as e:
                code = f'ERR {e}'
            time.sleep(15)
        time.sleep(5)
        team_count = {}
        for ev in events:
            tm = ev[1]
            team_count[tm] = team_count.get(tm, 0) + 1
        got = {'away': team_count.get(g['away_abbrev'], 0),
               'home': team_count.get(g['home_abbrev'], 0)}
        match = got['away'] == g['away_score'] and got['home'] == g['home_score']
        rows.append({'game_id': gid, 'http': code, 'n_events': len(events),
                     'away_goals_hr': got['away'], 'home_goals_hr': got['home'],
                     'matches_final': match, 'events': events})
        print(gid, 'http', code, 'events', len(events), 'match', match, flush=True)
    df = pd.DataFrame(rows)
    df.to_json(OUT_JSON, orient='records')
    print('games where HR scoring totals match final score:',
          df['matches_final'].sum(), 'of', len(df))
    print('non-200:', (df['http'] != '200').sum())


if __name__ == '__main__':
    main()
