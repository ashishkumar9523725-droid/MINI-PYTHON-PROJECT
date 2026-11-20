"""
Sports Tournament Analysis
--------------------------
- Elo-based match model (with draws in the group stage)
- Group stage (round-robin) + knockout (semi-finals, final)
- Monte Carlo simulation to estimate each team's chances
- Standings analysis for a single sample tournament
- CSV export and optional bar chart (if matplotlib is installed)

Run:  python tournament_analysis.py
Optional args:  python tournament_analysis.py --sims 20000 --seed 7
Only the standard library is required.
"""

import argparse
import csv
import itertools
import math
import random
from collections import defaultdict

# --------------------------------------------------------------------------
# 1. Tournament setup  (edit teams / ratings / groups here)
# --------------------------------------------------------------------------
GROUPS = {
    "A": {"Lions": 1850, "Eagles": 1760, "Wolves": 1690, "Sharks": 1610},
    "B": {"Tigers": 1820, "Bears": 1740, "Falcons": 1670, "Cobras": 1590},
}
ADVANCE_PER_GROUP = 2          # top N advance (this script assumes 2 groups, top 2)
HOME_ADVANTAGE = 0             # Elo points; tournaments are usually neutral venue
DRAW_BASE = 0.26               # max draw probability for evenly matched teams
AVG_GOALS = 1.35               # baseline goals per team per match


# --------------------------------------------------------------------------
# 2. Match model
# --------------------------------------------------------------------------
def expected_score(r_a, r_b):
    """Elo expected score for A (win=1, draw=0.5, loss=0)."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def match_probs(r_a, r_b):
    """Return (P(A win), P(draw), P(B win)) for a group-stage match."""
    e = expected_score(r_a + HOME_ADVANTAGE, r_b)
    # Draws are likelier when teams are evenly matched
    p_draw = DRAW_BASE * (1 - abs(2 * e - 1) ** 1.5)
    p_a = e - p_draw / 2
    p_b = 1 - e - p_draw / 2
    return p_a, p_draw, p_b


def poisson(lam, rng):
    """Knuth's Poisson sampler (no numpy needed)."""
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def simulate_score(r_a, r_b, rng):
    """Sample a scoreline; Elo difference shifts expected goals."""
    e = expected_score(r_a + HOME_ADVANTAGE, r_b)
    # Map win expectancy to a goal-rate split
    lam_a = AVG_GOALS * (0.5 + e) ** 1.2
    lam_b = AVG_GOALS * (1.5 - e) ** 1.2
    return poisson(lam_a, rng), poisson(lam_b, rng)


def knockout_winner(a, b, ratings, rng):
    """Single-elimination match: no draws (ties broken by Elo-weighted coin)."""
    ga, gb = simulate_score(ratings[a], ratings[b], rng)
    if ga != gb:
        return a if ga > gb else b
    return a if rng.random() < expected_score(ratings[a], ratings[b]) else b


# --------------------------------------------------------------------------
# 3. Tournament simulation
# --------------------------------------------------------------------------
def play_group(group_name, teams, ratings, rng):
    """Round-robin. Returns sorted standings list of dicts."""
    table = {t: dict(team=t, group=group_name, P=0, W=0, D=0, L=0, GF=0, GA=0, Pts=0)
             for t in teams}
    for a, b in itertools.combinations(teams, 2):
        ga, gb = simulate_score(ratings[a], ratings[b], rng)
        for t, gf, ga_ in ((a, ga, gb), (b, gb, ga)):
            table[t]["P"] += 1
            table[t]["GF"] += gf
            table[t]["GA"] += ga_
        if ga > gb:
            table[a]["W"] += 1; table[a]["Pts"] += 3; table[b]["L"] += 1
        elif gb > ga:
            table[b]["W"] += 1; table[b]["Pts"] += 3; table[a]["L"] += 1
        else:
            for t in (a, b):
                table[t]["D"] += 1; table[t]["Pts"] += 1
    rows = list(table.values())
    for r in rows:
        r["GD"] = r["GF"] - r["GA"]
    # Tie-breakers: points, goal difference, goals for, random draw
    rows.sort(key=lambda r: (r["Pts"], r["GD"], r["GF"], rng.random()), reverse=True)
    return rows


def play_tournament(rng):
    ratings = {t: r for g in GROUPS.values() for t, r in g.items()}
    standings = {g: play_group(g, list(teams), ratings, rng) for g, teams in GROUPS.items()}

    qualified = {g: [r["team"] for r in rows[:ADVANCE_PER_GROUP]]
                 for g, rows in standings.items()}
    (ga, gb) = list(GROUPS.keys())[:2]
    # Cross semi-finals: A1 v B2, B1 v A2
    sf1 = (qualified[ga][0], qualified[gb][1])
    sf2 = (qualified[gb][0], qualified[ga][1])
    w1 = knockout_winner(*sf1, ratings, rng)
    w2 = knockout_winner(*sf2, ratings, rng)
    champion = knockout_winner(w1, w2, ratings, rng)

    return {
        "standings": standings,
        "qualified": [t for v in qualified.values() for t in v],
        "finalists": (w1, w2),
        "champion": champion,
        "semis": (sf1, sf2),
    }


# --------------------------------------------------------------------------
# 4. Monte Carlo analysis
# --------------------------------------------------------------------------
def monte_carlo(n_sims, seed):
    rng = random.Random(seed)
    all_teams = [t for g in GROUPS.values() for t in g]
    counts = {t: defaultdict(int) for t in all_teams}
    pts_total = defaultdict(float)
    pos_counts = {t: [0] * max(len(g) for g in GROUPS.values()) for t in all_teams}

    for _ in range(n_sims):
        res = play_tournament(rng)
        for rows in res["standings"].values():
            for pos, r in enumerate(rows):
                pts_total[r["team"]] += r["Pts"]
                pos_counts[r["team"]][pos] += 1
        for t in res["qualified"]:
            counts[t]["advance"] += 1
        for t in res["finalists"]:
            counts[t]["final"] += 1
        counts[res["champion"]]["champion"] += 1

    ratings = {t: r for g in GROUPS.values() for t, r in g.items()}
    group_of = {t: g for g, teams in GROUPS.items() for t in teams}
    results = []
    for t in all_teams:
        results.append({
            "team": t,
            "group": group_of[t],
            "elo": ratings[t],
            "avg_pts": pts_total[t] / n_sims,
            "win_group_%": 100 * pos_counts[t][0] / n_sims,
            "advance_%": 100 * counts[t]["advance"] / n_sims,
            "final_%": 100 * counts[t]["final"] / n_sims,
            "champion_%": 100 * counts[t]["champion"] / n_sims,
        })
    results.sort(key=lambda r: r["champion_%"], reverse=True)
    return results


# --------------------------------------------------------------------------
# 5. Reporting helpers
# --------------------------------------------------------------------------
def print_table(rows, cols, title):
    print(f"\n{title}\n" + "-" * len(title))
    widths = [max(len(c), *(len(f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c]))
                            for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(
            (f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c])).ljust(w)
            for c, w in zip(cols, widths)))


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {path}")


def plot_results(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping chart (pip install matplotlib).")
        return
    teams = [r["team"] for r in results]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(teams))
    w = 0.27
    ax.bar([i - w for i in x], [r["advance_%"] for r in results], w, label="Advance")
    ax.bar(list(x), [r["final_%"] for r in results], w, label="Reach final")
    ax.bar([i + w for i in x], [r["champion_%"] for r in results], w, label="Champion")
    ax.set_xticks(list(x))
    ax.set_xticklabels(teams, rotation=30)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Tournament outcome probabilities (Monte Carlo)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Sports tournament analysis")
    ap.add_argument("--sims", type=int, default=10000, help="number of simulations")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--out", default=".", help="output folder for CSV / chart")
    args = ap.parse_args()

    # One sample tournament for illustration
    sample = play_tournament(random.Random(args.seed))
    for g, rows in sample["standings"].items():
        print_table(rows, ["team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"],
                    f"Sample tournament - Group {g}")
    print(f"\nSemi-finals: {sample['semis'][0]} and {sample['semis'][1]}")
    print(f"Final: {sample['finalists'][0]} vs {sample['finalists'][1]}")
    print(f"Champion: {sample['champion']}")

    # Monte Carlo
    print(f"\nRunning {args.sims:,} simulations...")
    results = monte_carlo(args.sims, args.seed)
    print_table(results,
                ["team", "group", "elo", "avg_pts", "win_group_%",
                 "advance_%", "final_%", "champion_%"],
                "Monte Carlo results")

    fav = results[0]
    print(f"\nFavourite: {fav['team']} ({fav['champion_%']:.1f}% to win). "
          f"Most open group: "
          f"{min(GROUPS, key=lambda g: max(r['win_group_%'] for r in results if r['group'] == g))}")

    save_csv(results, f"{args.out}/tournament_results.csv")
    plot_results(results, f"{args.out}/tournament_probabilities.png")


if __name__ == "__main__":
    main()
