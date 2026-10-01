"""09c_what_humans_saw.py - the human ratings against ALL eleven features, not just arousal.
If punctuation beats the emotion lexicon at predicting what a reader calls emotional, that is the finding.
Read-only; writes validation/09c_report.txt."""
import os, glob, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
V = f"{ROOT}/validation"
FEATURES = ["arousal", "valence", "dominance", "attribution_density", "hedging", "certainty",
            "caps_ratio", "exclamation_rate", "flesch_kincaid", "mtld", "log_length", "arousal_allwords"]
lines = []


def out(s=""):
    print(s)
    lines.append(str(s))


files = sorted(glob.glob(f"{V}/returned_*.xlsx"))
if not files:
    print("STOP: no returned_*.xlsx files in validation/")
    sys.exit(1)
R = {}
for f in files:
    name = os.path.basename(f).replace("returned_", "").replace(".xlsx", "")
    d = pd.read_excel(f, "Articles", skiprows=2)[["item", "rating"]]
    d["rating"] = pd.to_numeric(d.rating, errors="coerce")
    R[name] = d.dropna().set_index("item").rating.astype(float)
M = pd.DataFrame(R)
M["human"] = M.mean(axis=1)

key = pd.read_csv(f"{V}/09_rating_key.csv").set_index("item")
t = pd.read_parquet(f"{ROOT}/pipeline/analysis_table.parquet")
cols = [c for c in FEATURES if c in t.columns]
j = M.join(key[["article_id"]], how="inner").merge(
    t[["article_id"] + cols], on="article_id", how="inner")
out(f"09c_what_humans_saw - {len(j)} rated items joined to their features, "
    f"{len(files)} coder(s): {', '.join(M.columns[:-1])}\n")

out("WHAT THE HUMAN RATING ACTUALLY TRACKS (Spearman with the human rating, 120 items)")
rows = []
for c in cols:
    r = spearmanr(j.human, j[c])
    rows.append({"feature": c, "rho": r.statistic, "p": r.pvalue,
                 "mean at human 0": j.loc[j.human.round() == 0, c].mean(),
                 "mean at human 2": j.loc[j.human.round() == 2, c].mean()})
tab = pd.DataFrame(rows).set_index("feature").reindex(
    pd.DataFrame(rows).set_index("feature").rho.abs().sort_values(ascending=False).index)
out(tab.to_string(float_format=lambda v: f"{v:+.3f}" if abs(v) < 10 else f"{v:.2f}"))
best = tab.index[0]
out(f"\n   strongest: {best} (rho {tab.rho.iloc[0]:+.3f}, p {tab.p.iloc[0]:.4f})"
    f"{'  <- the emotion lexicon is NOT the closest match to human judgement' if best != 'arousal' else ''}")

out("\nSPREAD BETWEEN THE HUMAN EXTREMES, in SDs of that feature across the whole corpus")
for c in cols:
    sd = float(t[c].std())
    d = (j.loc[j.human.round() == 2, c].mean() - j.loc[j.human.round() == 0, c].mean()) / sd
    out(f"   {c:<20}{d:+.2f} SD")

if M.shape[1] - 1 >= 2:
    a, b = M.iloc[:, 0].dropna(), M.iloc[:, 1].dropna()
    both = a.index.intersection(b.index)
    rel = spearmanr(a[both], b[both]).statistic
    ceiling = np.sqrt(max(rel, 0))
    out(f"\nATTENUATION: coder-to-coder Spearman {rel:.3f} over {len(both)} items, so the highest "
        f"correlation any\n   measure could reach against this criterion is about {ceiling:.2f}. "
        f"Compare that with the observed values above.")
else:
    out("\nATTENUATION: only one coder, so the reliability of the human criterion is unknown and the "
        "correlations\n   above cannot be corrected for it. This is the reason the second coder matters.")
open(f"{V}/09c_report.txt", "w").write("\n".join(lines) + "\n")
print("\nRESULT: wrote validation/09c_report.txt")
