"""09d_three_coders.py - human validation with every returned coder, fixed rules, and a full sensitivity table.
Reads every validation/returned_<name>.xlsx, the rating key, the analysis table and cluster_framing.csv.
Read-only apart from validation/09d_report.txt and validation/09d_combinations.csv.

Decisions fixed in code, independent of the results:
  * PRIMARY article criterion = the mean of ALL returned coders (more raters, more reliable criterion).
    Agreement = the mean of the pairwise quadratic-weighted kappas; every pair is also reported.
  * TOPICS use only coders who were blind to the framing ranking. The first coder (tarik) has seen it,
    so his topic ratings are excluded (validation/09_interpretation_rule.md, point 4).
  * Every subset of coders is reported in the sensitivity table, but the branch is read off the primary only.
"""
import os, glob, itertools, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
V = f"{ROOT}/validation"
NOT_BLIND_FOR_TOPICS = {"tarik"}
FEATURES = ["arousal", "valence", "dominance", "attribution_density", "hedging", "certainty",
            "caps_ratio", "exclamation_rate", "flesch_kincaid", "mtld", "log_length", "arousal_allwords"]
lines = []


def out(s=""):
    print(s)
    lines.append(str(s))


def qwk(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 1 - ((a - b) ** 2).sum() / ((np.subtract.outer(a, b) ** 2).mean() * len(a))


files = sorted(glob.glob(f"{V}/returned_*.xlsx"))
if len(files) < 2:
    sys.exit("STOP: fewer than two returned_*.xlsx files in validation/")
A, T = {}, {}
for f in files:
    n = os.path.basename(f)[9:-5]
    a = pd.read_excel(f, "Articles", skiprows=2).set_index("item")
    A[n] = pd.to_numeric(a["rating"], errors="coerce")
    t = pd.read_excel(f, "Topics", skiprows=2).set_index("topic")
    T[n] = pd.to_numeric(t["rating"], errors="coerce")
A, T = pd.DataFrame(A), pd.DataFrame(T)
C = list(A.columns)
key = pd.read_csv(f"{V}/09_rating_key.csv").set_index("item")
key = key[key.set == "main"]
A = A.loc[A.index.intersection(key.index)]
out(f"09d_three_coders - coders: {', '.join(C)} | {len(A)} main items")

# identical-sheet guard: two independent coders never agree on every item
for x, y in itertools.combinations(C, 2):
    if (A[x] == A[y]).all():
        out(f"WARNING: {x} and {y} have IDENTICAL article ratings on all {len(A)} items - check the files")

out("\n1  EACH CODER")
for c in C:
    r = A[c]
    blocks = ", ".join(f"{r.iloc[i*20:(i+1)*20].mean():.2f}" for i in range(len(r) // 20))
    out(f"   {c:<8} rated {int(r.notna().sum())} | 0: {int((r==0).sum())}, 1: {int((r==1).sum())}, 2: {int((r==2).sum())}"
        f" | mean {r.mean():.2f} | drift by block of 20: {blocks}")

out("\n2  AGREEMENT BETWEEN CODERS (120 articles)")
pair_k, pair_r = [], []
for x, y in itertools.combinations(C, 2):
    k, r = qwk(A[x], A[y]), spearmanr(A[x], A[y]).statistic
    pair_k.append(k); pair_r.append(r)
    out(f"   {x:<6} vs {y:<6} exact {np.mean(A[x]==A[y]):.0%} | within one {np.mean((A[x]-A[y]).abs()<=1):.0%}"
        f" | quadratic-weighted kappa {k:.3f} | Spearman {r:.3f}")
kbar, rbar = float(np.mean(pair_k)), float(np.mean(pair_r))
out(f"   mean pairwise kappa {kbar:.3f} | mean pairwise Spearman {rbar:.3f}")


def rel_of_mean(k_coders, r):          # Spearman-Brown reliability of a k-coder mean
    return k_coders * r / (1 + (k_coders - 1) * r)


out("\n3  THE HUMAN CRITERION AGAINST LEXICON AROUSAL (Spearman, 120 articles)")
rows = []
for size in range(1, len(C) + 1):
    for sub in itertools.combinations(C, size):
        m = A[list(sub)].mean(axis=1)
        s = spearmanr(m, key.loc[A.index, "arousal"])
        ks = [qwk(A[x], A[y]) for x, y in itertools.combinations(sub, 2)]
        rs = [spearmanr(A[x], A[y]).statistic for x, y in itertools.combinations(sub, 2)]
        rel = rel_of_mean(size, np.mean(rs)) if rs else np.nan
        rows.append({"coders": "+".join(sub), "n_coders": size, "kappa": np.mean(ks) if ks else np.nan,
                     "rho_lexicon": s.statistic, "p": s.pvalue, "reliability_of_mean": rel,
                     "ceiling": np.sqrt(rel) if rs else np.nan, "primary": size == len(C)})
tab = pd.DataFrame(rows)
tab.to_csv(f"{V}/09d_combinations.csv", index=False, float_format="%.4f")
out(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

P = tab[tab.primary].iloc[0]
out(f"\n   PRIMARY (all {len(C)} coders): mean pairwise kappa {P.kappa:.3f}, rho {P.rho_lexicon:.3f} "
    f"(p {P.p:.4f}); reliability of the {len(C)}-coder mean {P.reliability_of_mean:.3f}, "
    f"so no measure can correlate above about {P.ceiling:.2f} with it")

out("\n4  WHAT THE PRIMARY CRITERION TRACKS (Spearman with the all-coder mean)")
t = pd.read_parquet(f"{ROOT}/pipeline/analysis_table.parquet")
hm = A.mean(axis=1).rename("human").to_frame().join(key[["article_id"]])
j = hm.merge(t[["article_id"] + [c for c in FEATURES if c in t.columns]], on="article_id")
fr = pd.DataFrame([{"feature": c, "rho": spearmanr(j.human, j[c]).statistic, "p": spearmanr(j.human, j[c]).pvalue}
                   for c in FEATURES if c in j.columns])
fr = fr.reindex(fr.rho.abs().sort_values(ascending=False).index)
out(fr.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
out(f"   strongest two: {fr.feature.iloc[0]}, {fr.feature.iloc[1]}")

out("\n5  TOPIC CONTESTEDNESS AGAINST FRAMING DIVERGENCE (blind coders only)")
blind = [c for c in T.columns if c not in NOT_BLIND_FOR_TOPICS and T[c].notna().sum() > 0]
out(f"   blind topic coders: {', '.join(blind) or 'none'} | excluded as not blind: "
    f"{', '.join(c for c in T.columns if c in NOT_BLIND_FOR_TOPICS and T[c].notna().sum() > 0) or 'none'}")
framing = pd.read_csv(f"{ROOT}/partisanship/cluster_framing.csv").set_index("cluster").framing
topic_rows = []
for c in blind:
    both = T[c].dropna().index.intersection(framing.dropna().index)
    s = spearmanr(T[c][both], framing[both])
    topic_rows.append((c, len(both), s.statistic, s.pvalue))
    out(f"   {c:<6} {len(both)} topics  Spearman {s.statistic:+.3f}  p {s.pvalue:.4f}")
if len(blind) >= 2:
    for x, y in itertools.combinations(blind, 2):
        m = T[x].notna() & T[y].notna()
        out(f"   agreement {x} vs {y} on {int(m.sum())} topics: kappa {qwk(T[x][m], T[y][m]):.3f}")
tm = T[blind].mean(axis=1).dropna()
both = tm.index.intersection(framing.dropna().index)
ts = spearmanr(tm[both], framing[both]) if len(both) >= 10 else None
if ts is not None:
    out(f"   PRIMARY (mean of blind coders): {len(both)} topics  Spearman {ts.statistic:+.3f}  p {ts.pvalue:.4f}")
else:
    out(f"   fewer than 10 topics rated - not analysed")

out("\n6  READING UNDER validation/09_interpretation_rule.md (primary only)")
band = "reliable" if P.kappa >= 0.60 else "moderately reliable" if P.kappa >= 0.40 else "unreliable"
if P.kappa < 0.40:
    branch = "C"
elif P.rho_lexicon >= 0.30:
    branch = "A"
else:
    branch = "B"
out(f"   agreement {P.kappa:.3f} -> {band} | rho {P.rho_lexicon:.3f} -> BRANCH {branch}")
if ts is not None:
    sup = ts.statistic > 0 and ts.pvalue < 0.05
    out(f"   topics: rho {ts.statistic:+.3f}, p {ts.pvalue:.4f} -> "
        f"{'SUPPORTS' if sup else 'does NOT support'} reading framing divergence as contestation")
style = {"caps_ratio", "exclamation_rate", "flesch_kincaid", "mtld", "log_length"}
out(f"   strongest two correlates are {'both style features' if set(fr.feature.iloc[:2]) <= style else 'not both style features'}"
    f" -> Branch B's bracketed style sentence {'KEPT' if set(fr.feature.iloc[:2]) <= style else 'DROPPED'} (if Branch B)")

out("\n7  EXPLORATORY (not pre-specified): does the human criterion track outlet factuality?")
fh = spearmanr(A.mean(axis=1), key.loc[A.index, "fact"])
fl = spearmanr(key.loc[A.index, "arousal"], key.loc[A.index, "fact"])
out(f"   all-coder mean vs factuality (0-4): Spearman {fh.statistic:+.3f}  p {fh.pvalue:.4f}")
for c in C:
    s = spearmanr(A[c], key.loc[A.index, "fact"])
    out(f"   {c:<6} Spearman {s.statistic:+.3f}  p {s.pvalue:.4f}")
out(f"   lexicon arousal vs factuality on the same items: Spearman {fl.statistic:+.3f}  p {fl.pvalue:.4f}"
    f"  (the sample was stratified by arousal quintile x factuality, which flattens this by design)")
g = pd.DataFrame({"human": A.mean(axis=1), "lexicon": key.loc[A.index, "arousal"],
                  "fact": key.loc[A.index, "fact"]}).groupby("fact").agg(items=("human", "size"),
                  human_mean=("human", "mean"), lexicon_mean=("lexicon", "mean"))
out("   " + g.to_string(float_format=lambda v: f"{v:+.3f}").replace("\n", "\n   "))

open(f"{V}/09d_report.txt", "w").write("\n".join(lines) + "\n")
print("\nRESULT: wrote validation/09d_report.txt and validation/09d_combinations.csv")
