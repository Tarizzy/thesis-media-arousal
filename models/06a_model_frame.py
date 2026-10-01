"""06a_model_frame.py - build the modelling frame for the cross-classified model.
Joins the analysis table to the topic measure, fixes the codings, and writes models/model_frame.csv
(read by 06b_mixed.R) plus models/06a_report.txt."""
import os, sys, time
sys.path.insert(0, os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "pipeline"))
import numpy as np, pandas as pd
import preprocessing as pp

OUT = f"{pp.ROOT}/models"
os.makedirs(OUT, exist_ok=True)
FACT = {"Very Low": 0, "Low": 1, "Mixed": 2, "Mostly Factual": 3, "High": 4, "Very High": 4}
SIDE = {"Extreme Left": "left", "Left": "left", "Left-Center": "left", "Least Biased": "centre",
        "Right-Center": "right", "Right": "right", "Extreme Right": "right"}
EXTREMITY = {"Least Biased": 0, "Left-Center": 1, "Right-Center": 1, "Left": 2, "Right": 2,
             "Extreme Left": 3, "Extreme Right": 3}
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


t0 = time.time()
df = pd.read_parquet(f"{pp.ROOT}/pipeline/analysis_table.parquet")
fr = pd.read_csv(f"{pp.ROOT}/partisanship/framing_model.csv")
out(f"06a_model_frame - {time.strftime('%Y-%m-%d %H:%M')}")
out(f"analysis table {len(df):,} articles, {df.source.nunique()} outlets, "
    f"{df.topic_cluster_model.nunique()} clusters | topic measure {len(fr)} rows")

# outcome: standardised so coefficients read in SDs of article arousal
mu, sd = df.arousal.mean(), df.arousal.std()
df["arousal_z"] = (df.arousal - mu) / sd
out(f"arousal mean {mu:.5f}, SD {sd:.5f} (coefficients are in SDs of article arousal)")

# outlet-level predictors
df["fact"] = df.mbfc_factuality.map(FACT)
df["bias_side"] = df.mbfc_bias.map(SIDE)
df["bias_ext"] = df.mbfc_bias.map(EXTREMITY)
for col, name in [("fact", "factuality"), ("bias_side", "bias side"), ("bias_ext", "bias extremity")]:
    if df[col].isna().any():
        bad = df.loc[df[col].isna(), "mbfc_factuality" if col == "fact" else "mbfc_bias"].unique()
        raise SystemExit(f"{name} not codable for: {bad}")
outlets = df.groupby("source")[["fact", "bias_ext"]].first()
df["fact_c"] = df.fact - outlets.fact.mean()          # centred on the mean outlet, not the mean article
out(f"factuality centred at the outlet mean {outlets.fact.mean():.3f} "
    f"(0 = Very Low ... 4 = High incl. Very High)")

# topic-level predictor
col = "framing_z" if "framing_z" in fr.columns else "framing"
fr = fr.rename(columns={col: "part_z"})[["topic_cluster_model", "part_z"]]
if "framing_z" not in fr.columns:
    fr["part_z"] = (fr.part_z - fr.part_z.mean()) / fr.part_z.std()
before = len(df)
df = df.merge(fr, on="topic_cluster_model", how="left", validate="many_to_one")
if len(df) != before or df.part_z.isna().any():
    missing = sorted(df.loc[df.part_z.isna(), "topic_cluster_model"].unique())
    raise SystemExit(f"topic measure missing for clusters {missing}")
out(f"topic partisanship standardised across {fr.topic_cluster_model.nunique()} clusters "
    f"(mean {fr.part_z.mean():.3f}, SD {fr.part_z.std():.3f})")

keep = ["article_id", "source", "topic_cluster_model", "year", "arousal", "arousal_z", "fact", "fact_c",
        "part_z", "bias_side", "bias_ext", "outlet_ok", "country", "adfontes_lean", "n_matched"]
frame = df[[c for c in keep if c in df.columns]].copy()
frame["year"] = frame.year.astype(int)
frame.to_csv(f"{OUT}/model_frame.csv", index=False)

# what the model will lean on
out("\nOUTLETS BY FACTUALITY LEVEL (model coding)")
lev = df.groupby("source")[["fact"]].first().join(df.groupby("source").size().rename("articles"))
out(lev.groupby("fact").agg(outlets=("articles", "size"), articles=("articles", "sum")).to_string())
out("\nFACTUALITY vs BIAS EXTREMITY at outlet level (the confound M5 controls for)")
out(pd.crosstab(outlets.fact, outlets.bias_ext).to_string())
out(f"   Spearman rho {outlets.fact.corr(outlets.bias_ext, method='spearman'):+.3f}")
out(f"\nvariation the interaction rests on: {frame.source.nunique()} outlets x "
    f"{frame.topic_cluster_model.nunique()} clusters, {len(frame):,} articles")
out(f"articles per outlet: median {frame.groupby('source').size().median():.0f}; "
    f"per cluster: median {frame.groupby('topic_cluster_model').size().median():.0f}")
problems = []
if frame[["arousal_z", "fact_c", "part_z"]].isna().any().any():
    problems.append("missing values in the model columns")
if frame.topic_cluster_model.nunique() != 32:
    problems.append(f"{frame.topic_cluster_model.nunique()} clusters, expected 32")
out(f"\nRESULT: {'OK' if not problems else 'CHECK: ' + '; '.join(problems)} - "
    f"wrote model_frame.csv ({len(frame):,} rows) in {time.time() - t0:.0f}s")
with open(f"{OUT}/06a_report.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
