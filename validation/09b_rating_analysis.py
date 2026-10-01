"""09b_rating_analysis.py - does the word-list arousal measure agree with human judgement?
Put each returned workbook in validation/ as returned_<name>.xlsx, then run. Read-only apart from the report.
Works with one coder (no agreement statistic) or several."""
import os, glob, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
V = f"{ROOT}/validation"
files = sorted(glob.glob(f"{V}/returned_*.xlsx"))
if not files:
    print("STOP: no returned_*.xlsx files in validation/")
    sys.exit(1)
key = pd.read_csv(f"{V}/09_rating_key.csv")
lines = []


def out(s=""):
    print(s)
    lines.append(str(s))


R = {}
for f in files:
    name = os.path.basename(f).replace("returned_", "").replace(".xlsx", "")
    d = pd.read_excel(f, "Articles", skiprows=2)[["item", "rating"]]
    d["rating"] = pd.to_numeric(d.rating, errors="coerce")
    d = d.dropna()
    R[name] = d.set_index("item").rating.astype(float)
    out(f"{name}: {len(d)} of {len(key[key.set == 'main'])} items rated | "
        + ", ".join(f"{v}: {int((d.rating == v).sum())}" for v in (0, 1, 2))
        + f" | mean {d.rating.mean():.2f}")
M = pd.DataFrame(R)

if M.shape[1] >= 2:
    a, b = M.iloc[:, 0].dropna(), M.iloc[:, 1].dropna()
    both = a.index.intersection(b.index)
    a, b = a[both], b[both]
    kappa = 1 - ((a - b) ** 2).sum() / ((np.subtract.outer(a.to_numpy(), b.to_numpy()) ** 2).mean() * len(a))
    out(f"\nAGREEMENT BETWEEN CODERS on {len(both)} items shared by "
        f"{M.columns[0]} and {M.columns[1]}")
    out(f"   exact agreement {np.mean(a.to_numpy() == b.to_numpy()):.0%} | "
        f"within one point {np.mean((a - b).abs() <= 1):.0%}")
    out(f"   quadratic-weighted kappa {kappa:.3f}")
    out(f"   means {a.mean():.2f} vs {b.mean():.2f}")
else:
    out("\n(only one coder's ratings present: no agreement statistic yet)")

k = key.set_index("item")
M["human"] = M.mean(axis=1)
j = M.join(k, how="inner")
out(f"\nAGREEMENT WITH THE WORD-LIST MEASURE ({len(j)} items)")
rho = spearmanr(j.human, j.arousal)
out(f"   Spearman of the human rating with lexicon arousal: {rho.statistic:.3f} (p {rho.pvalue:.4f})")
for v in sorted(j.human.round().unique()):
    s = j[j.human.round() == v]
    out(f"   human {v:.0f} ({len(s):3d} items): mean lexicon arousal {s.arousal.mean():+.4f}")
out("   by factuality class (human mean / lexicon mean):")
for f_, s in j.groupby("mbfc_factuality"):
    out(f"      {f_:<16}{s.human.mean():.2f} / {s.arousal.mean():+.4f}  ({len(s)} items)")
within = j.groupby("stratum")[["human", "arousal"]].apply(
    lambda s: spearmanr(s.human, s.arousal).statistic if s.human.nunique() > 1 else np.nan)
out("   within arousal strata: " + ", ".join(f"{i}: {v:+.2f}" for i, v in within.items()))
out(f"   drift check, mean rating by block of 20 items in the order they were rated: "
    + ", ".join(f"{M.human.to_numpy()[i * 20:(i + 1) * 20].mean():.2f}" for i in range(len(M) // 20)))

try:
    tp = pd.concat([pd.read_excel(f, "Topics", skiprows=2)[["topic", "rating"]] for f in files])
    tp["rating"] = pd.to_numeric(tp.rating, errors="coerce")
    tp = tp.dropna().groupby("topic").rating.mean()
    fr = pd.read_csv(f"{ROOT}/partisanship/cluster_framing.csv", index_col=0).framing
    both = tp.index.intersection(fr.dropna().index)
    if len(both) >= 10:
        r = spearmanr(tp[both], fr[both])
        out(f"\nTOPIC CONTESTEDNESS: human ratings vs framing divergence over {len(both)} topics: "
            f"Spearman {r.statistic:.3f} (p {r.pvalue:.4f})")
        out("   " + ", ".join(f"{int(c)}:{int(tp[c])}" for c in sorted(both)))
    else:
        out(f"\n(Topics tab: {len(tp)} topics rated, need 10 - not analysed)")
except Exception as e:
    out(f"\n(no usable Topics tab: {e})")
open(f"{V}/09b_report.txt", "w").write("\n".join(lines) + "\n")
print("\nRESULT: wrote validation/09b_report.txt")
