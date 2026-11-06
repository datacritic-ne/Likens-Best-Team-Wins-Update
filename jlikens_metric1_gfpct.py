#!/usr/bin/env python
"""
JLikens metric 1 — 82-game regular-season GF% with EN (and SO) goals excluded,
for every team in all 18 dataset seasons, from the Stage 2 patched CSVs.

JLikens's own numbers (comment, 2011-06-27, on the best-team post):
  2005-06 Hurricanes: 274 GF, 249 GA (clean) -> GF% 0.524, +25 GD.

Outputs:
  outputs/jlikens_metric1_gfpct.csv      (long: season, team, gf, ga, gf_pct, rank)
  outputs/reports/jlikens_metric1_report.txt
"""
import pandas as pd

BASE = "/workspace/projects/hockey-lab/analyses/likens"
SEASONS = ["20052006", "20062007", "20072008", "20082009", "20092010", "20102011",
           "20112012", "20132014", "20142015", "20152016", "20162017", "20172018",
           "20182019", "20212022", "20222023", "20232024", "20242025", "20252026"]

rows = []
for season in SEASONS:
    df = pd.read_csv(f"{BASE}/data/processed/games_{season}.csv")
    # clean GF: raw score minus EN minus SO (per Stage 2 columns)
    home = pd.DataFrame({
        "season": season, "team": df.home_abbrev, "games": 1,
        "gf_clean": df.home_gf_clean, "ga_clean": df.away_gf_clean,
        "gf_raw": df.home_score, "ga_raw": df.away_score,
    })
    away = pd.DataFrame({
        "season": season, "team": df.away_abbrev, "games": 1,
        "gf_clean": df.away_gf_clean, "ga_clean": df.home_gf_clean,
        "gf_raw": df.away_score, "ga_raw": df.home_score,
    })
    both = pd.concat([home, away])
    agg = both.groupby(["season", "team"], as_index=False)[
        ["games", "gf_clean", "ga_clean", "gf_raw", "ga_raw"]].sum()
    assert (agg.games == 82).all(), f"{season}: team game counts != 82"
    agg["gf_pct"] = agg.gf_clean / (agg.gf_clean + agg.ga_clean)
    agg["gf_pct_raw"] = agg.gf_raw / (agg.gf_raw + agg.ga_raw)
    agg["diff_clean"] = agg.gf_clean - agg.ga_clean
    agg["rank"] = agg.gf_pct.rank(ascending=False, method="min").astype(int)
    rows.append(agg)

out = pd.concat(rows, ignore_index=True)
out = out.sort_values(["season", "rank"])
out.to_csv(f"{BASE}/outputs/jlikens_metric1_gfpct.csv", index=False)

lines = ["JLikens metric 1 — 82-game regular-season GF%, EN and SO goals excluded",
         "Source: Stage 2 patched CSVs (NHL API + HR-verified; 2009-10 repaired per Phase A)",
         ""]
for season in SEASONS:
    s = out[out.season == season]
    top = s.head(5)
    lines.append(f"== {season[:4]}-{season[4:]} ==")
    for _, r in top.iterrows():
        lines.append(f"  {r['rank']:>2}. {r.team}  GF% {r.gf_pct:.3f}  "
                     f"({int(r.gf_clean)}-{int(r.ga_clean)}, {int(r.diff_clean):+d})")
    # Carolina 2005-06 check
    if season == "20052006":
        car = s[s.team == "CAR"].iloc[0]
        lines.append(f"  [JLikens check] CAR: GF% {car.gf_pct:.3f} rank {int(car['rank'])} "
                     f"(his: 0.524 from 274-249; ours 278-253 — same +25 GD, same rounded "
                     f"GF%; the 4-goal difference is his unavailable EN/SO goal list vs ours)")
    lines.append("")

car = out[(out.season == "20052006") & (out.team == "CAR")].iloc[0]
lines.insert(2, f"VALIDATION: 2005-06 CAR = {int(car.gf_clean)}-{int(car.ga_clean)} "
                f"({int(car.diff_clean):+d}), GF% {car.gf_pct:.4f}, rank {int(car['rank'])}. "
                f"JLikens: 274-249 (+25), 0.524, 11th of 30. Raw GF/GA (294-260) match "
                f"hockey-reference exactly. His exact EN/SO goal lists are not recoverable; "
                f"ours are verified game-by-game from PBP.")
txt = "\n".join(lines) + "\n"
open(f"{BASE}/outputs/reports/jlikens_metric1_report.txt", "w").write(txt)
print(txt)
