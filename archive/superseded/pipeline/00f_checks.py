"""00f_checks.py - resolves the open questions raised during writing, from the final analysis table.
Writes pipeline/00f_checks.txt and pipeline/00f_descriptives.csv. Read-only otherwise."""
import os
import numpy as np, pandas as pd, duckdb
from scipy.stats import spearmanr

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
FACT = ["Very Low", "Low", "Mixed", "Mostly Factual", "High"]
BIAS = {"Extreme Left": -3, "Left": -2, "Left-Center": -1, "Least Biased": 0,
        "Right-Center": 1, "Right": 2, "Extreme Right": 3}
FEATURES = ["arousal", "valence", "dominance", "attribution_density", "hedging", "certainty",
            "caps_ratio", "exclamation_rate", "flesch_kincaid", "mtld", "log_length"]
lines = []


def out(s=""):
    print(s)
    lines.append(str(s))


t = pd.read_parquet(f"{ROOT}/pipeline/analysis_table.parquet")
o = t.groupby("source").first()
out(f"00f_checks - {len(t):,} articles, {len(o)} outlets\n")

out("=" * 90 + "\n1  EXTREME OUTLETS (42 or 47?)")
ext = o[o.mbfc_bias.isin(["Extreme Left", "Extreme Right"])]
out(f"   outlets with an Extreme bias label: {len(ext)}")
out("   " + o.assign(x=o.mbfc_bias.isin(["Extreme Left", "Extreme Right"]))
    .groupby(["x", "mbfc_factuality"]).size().to_string().replace("\n", "\n   "))
bad = ext[~ext.mbfc_factuality.isin(["Very Low", "Low"])]
out(f"   Extreme outlets NOT rated Low or Very Low: {len(bad)} {list(bad.index)}")
out(f"   -> correct sentence: 'all {len(ext)} Extreme outlets are Low or Very Low'"
    if bad.empty else f"   -> the 'all Extreme are Low/Very Low' claim is FALSE as written")

out("\n" + "=" * 90 + "\n2  LEAST BIASED OUTLETS")
lb = o[o.mbfc_bias == "Least Biased"]
out(f"   {len(lb)} outlets; factuality: {dict(lb.mbfc_factuality.value_counts())}")

out("\n" + "=" * 90 + "\n3  AD FONTES COVERAGE (80, 81 or 82?)")
if "adfontes_lean" in o.columns:
    lean = pd.to_numeric(o.adfontes_lean, errors="coerce")
    have = o[lean.notna()]
    out(f"   outlets in the analysis table with an Ad Fontes lean: {len(have)} of {len(o)}")
    out(f"   articles from those outlets: {int(t.source.isin(have.index).sum()):,}")
    r = spearmanr(o.loc[have.index].mbfc_bias.map(BIAS), lean[have.index])
    out(f"   Spearman of MBFC bias with Ad Fontes lean: {r.statistic:.3f} (p {r.pvalue:.1e}), n = {len(have)}")
    out("   coverage by factuality class (with score / total):")
    for f in FACT:
        cls = o[o.mbfc_factuality == f]
        out(f"      {f:<16}{int(lean[cls.index].notna().sum()):>3} of {len(cls):>3}")
    ind = [s for s in o.index if "independ" in s.lower()]
    if ind:
        out(f"   The Independent's stored lean: {dict(lean[ind])} (was mistyped 6,-91 = -6.91)")
else:
    out("   no adfontes_lean column in the analysis table")

out("\n" + "=" * 90 + "\n4  WHY 'VERY HIGH' WAS COLLAPSED INTO 'HIGH'")
try:
    con = duckdb.connect()
    con.execute(f"ATTACH '{ROOT}/data/misinfo-general/metadata.db' AS meta (READ_ONLY)")
    con.register("keep", pd.DataFrame({"source": o.index}))
    raw = con.execute("SELECT s.factuality, count(*) n FROM meta.sources s JOIN keep k "
                      "ON k.source = s.source GROUP BY 1 ORDER BY 2 DESC").df()
    out("   original MBFC labels for the 211 outlets, as shipped in metadata.db:")
    out("   " + raw.to_string(index=False).replace("\n", "\n   "))
    vh = raw[raw.factuality.str.contains("Very High", case=False, na=False)].n.sum()
    out(f"   -> {int(vh)} outlets are rated Very High; too few to estimate a separate class, "
        "which is why Very High is collapsed into High.")
except Exception as e:
    out(f"   could not read metadata.db: {e}")

out("\n" + "=" * 90 + "\n5  IS FLESCH-KINCAID WINSORISED IN THE ANALYSIS TABLE?")
fk = t.flesch_kincaid
lo, hi = fk.quantile([0.01, 0.99])
out(f"   min {fk.min():.2f}, 1st pct {lo:.2f}, 99th pct {hi:.2f}, max {fk.max():.2f}")
out(f"   articles at exactly the min: {int((fk == fk.min()).sum())}; at exactly the max: {int((fk == fk.max()).sum())}")
out(f"   -> {'ALREADY winsorised in the table' if (fk == fk.min()).sum() > 100 else 'NOT winsorised in the table; each script winsorises its own copy (08a logs the bounds it used)'}")

out("\n" + "=" * 90 + "\n6  FEATURE DESCRIPTIVES BY FACTUALITY CLASS (for the Data chapter table)")
t["fact"] = pd.Categorical(t.mbfc_factuality, FACT, ordered=True)
desc = t.groupby("fact", observed=True)[FEATURES].mean().T
desc["overall mean"] = t[FEATURES].mean()
desc["overall SD"] = t[FEATURES].std()
desc.round(4).to_csv(f"{ROOT}/pipeline/00f_descriptives.csv")
out(desc.round(3).to_string())
out("\n   (saved to pipeline/00f_descriptives.csv; outlet counts per class are in 08a_report.txt)")

out("\n" + "=" * 90 + "\n7  ARTICLES PUBLISHED IN 2018 (for the without-2018 specification)")
if "year" in t.columns:
    out("   " + t.year.value_counts().sort_index().to_string().replace("\n", "\n   "))
    out(f"   dropping 2018 would leave {int((t.year != 2018).sum()):,} articles")
open(f"{ROOT}/pipeline/00f_checks.txt", "w").write("\n".join(lines) + "\n")
print("\nRESULT: wrote pipeline/00f_checks.txt and 00f_descriptives.csv")
