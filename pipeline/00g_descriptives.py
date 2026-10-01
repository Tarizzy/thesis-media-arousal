"""00g_descriptives.py - the numbers the Data chapter still needs: feature means and SDs by factuality
class, how often the sparse features are zero, which correlation coefficient gives -0.741, Ad Fontes
coverage on the 211 analysis-table outlets, and two definition checks. Read-only apart from its outputs.
Writes pipeline/00g_checks.txt and pipeline/00g_descriptives.csv."""
import os
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
FACT5 = {"Very Low": 0, "Low": 1, "Mixed": 2, "Mostly Factual": 3, "High": 4, "Very High": 4}
NAMES = ["Very Low", "Low", "Mixed", "Mostly Factual", "High"]
EXT = {"Least Biased": 0, "Left-Center": 1, "Right-Center": 1, "Left": 2, "Right": 2,
       "Extreme Left": 3, "Extreme Right": 3}
FEATURES = ["arousal", "valence", "dominance", "attribution_density", "hedging", "certainty",
            "caps_ratio", "exclamation_rate", "flesch_kincaid", "mtld", "log_length"]
SPARSE = ["caps_ratio", "certainty", "exclamation_rate", "hedging"]
lines = []


def out(s=""):
    print(s)
    lines.append(str(s))


t = pd.read_parquet(f"{ROOT}/pipeline/analysis_table.parquet")
t["fact_n"] = t.mbfc_factuality.map(FACT5)
t["fact"] = pd.Categorical([NAMES[i] for i in t.fact_n], categories=NAMES, ordered=True)
o = t.groupby("source").first()
o["fact_n"] = o.mbfc_factuality.map(FACT5)
o["bias_ext"] = o.mbfc_bias.map(EXT)
out(f"00g_descriptives - {len(t):,} articles, {len(o)} outlets (Very High merged into High)\n")

out("=" * 92 + "\n1  FEATURES BY FACTUALITY CLASS  (mean above, SD below, article level)")
mean = t.groupby("fact", observed=True)[FEATURES].mean().T
sd = t.groupby("fact", observed=True)[FEATURES].std().T
tab = pd.DataFrame(index=FEATURES)
for c in NAMES:
    if c in mean.columns:
        tab[f"{c} mean"] = mean[c]
        tab[f"{c} SD"] = sd[c]
tab["all mean"] = t[FEATURES].mean()
tab["all SD"] = t[FEATURES].std()
out(tab.round(4).to_string())
tab.round(5).to_csv(f"{ROOT}/pipeline/00g_descriptives.csv")
out("\n   articles and outlets per class:")
out("   " + t.groupby("fact", observed=True).agg(articles=("article_id", "size"),
                                                 outlets=("source", "nunique")).to_string().replace("\n", "\n   "))

out("\n" + "=" * 92 + "\n2  HOW OFTEN THE SPARSE FEATURES ARE EXACTLY ZERO")
for f in SPARSE:
    if f in t:
        z = float((t[f] == 0).mean())
        byc = ", ".join(f"{c} {float((t.loc[t.fact == c, f] == 0).mean()):.0%}" for c in NAMES)
        out(f"   {f:<18}{z:.1%} of all articles   (by class: {byc})")

out("\n" + "=" * 92 + "\n3  FACTUALITY AND BIAS EXTREMITY: WHICH COEFFICIENT IS -0.741?")
ok = o.fact_n.notna() & o.bias_ext.notna()
pe = pearsonr(o.fact_n[ok], o.bias_ext[ok])
sp = spearmanr(o.fact_n[ok], o.bias_ext[ok])
out(f"   over {int(ok.sum())} outlets:  Pearson r = {pe.statistic:.3f}   Spearman rho = {sp.statistic:.3f}")
hit = [n for n, v in [("Pearson", pe.statistic), ("Spearman", sp.statistic)] if abs(abs(v) - 0.741) < 0.0015]
out(f"   -> the reported -0.741 is the {hit[0]} coefficient" if len(hit) == 1
    else "   -> neither matches -0.741 exactly; report the coefficient printed above and say which it is")
out("   cross-tabulation (outlets):")
ct = pd.crosstab(o.mbfc_bias.map(EXT).map({0: "0 least biased", 1: "1 left/right-center",
                                           2: "2 left/right", 3: "3 extreme"}).rename("bias extremity"),
                 o.fact_n.map(dict(enumerate(NAMES))).rename("factuality"))
out("   " + ct.reindex(columns=[c for c in NAMES if c in ct.columns]).to_string().replace("\n", "\n   "))

out("\n" + "=" * 92 + "\n4  AD FONTES COVERAGE, ANALYSIS-TABLE OUTLETS ONLY")
if "adfontes_status" in o.columns:
    out("   status: " + ", ".join(f"{k} {v}" for k, v in o.adfontes_status.value_counts().items())
        + f"  (total {len(o)})")
    cov = pd.DataFrame({"outlets": o.groupby(o.fact_n.map(dict(enumerate(NAMES)))).size(),
                        "with a score": o[o.adfontes_status == "score"]
                       .groupby(o.fact_n.map(dict(enumerate(NAMES)))).size()}) \
        .reindex(NAMES).fillna(0).astype(int).rename_axis("factuality")
    out("   by merged factuality class (Very High counted inside High):")
    out("   " + cov.to_string().replace("\n", "\n   "))
    out(f"   totals: {int(cov['with a score'].sum())} of {int(cov.outlets.sum())} outlets; "
        f"{int(t.adfontes_lean.notna().sum()):,} of {len(t):,} articles "
        f"({t.adfontes_lean.notna().mean():.0%})")
else:
    out("   no adfontes_status column in the analysis table")

out("\n" + "=" * 92 + "\n5  DEFINITION CHECKS")
if "n_tokens" in t.columns:
    e = float((t.log_length - np.log(t.n_tokens)).abs().max())
    ten = float((t.log_length - np.log10(t.n_tokens)).abs().max())
    out(f"   log_length vs natural log of n_tokens: largest difference {e:.2e}")
    out(f"   log_length vs base-10 log of n_tokens: largest difference {ten:.2e}")
    out(f"   -> log_length is the {'NATURAL log (base e)' if e < ten else 'base-10 log'}; "
        f"token counts run {int(t.n_tokens.min()):,} to {int(t.n_tokens.max()):,}")
if "arousal_allwords" in t.columns:
    r = pearsonr(t.arousal, t.arousal_allwords)
    out(f"   arousal vs arousal_allwords: r = {r.statistic:.3f}; means {t.arousal.mean():+.4f} "
        f"vs {t.arousal_allwords.mean():+.4f}")
    out("   arousal_allwords keeps the same name exclusion and drops only the content-word filter "
        "(stopwords, modals, number words and the words that define the other features)")
open(f"{ROOT}/pipeline/00g_checks.txt", "w").write("\n".join(lines) + "\n")
print("\nRESULT: wrote pipeline/00g_checks.txt and pipeline/00g_descriptives.csv")
