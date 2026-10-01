"""09a_frames.py - data for the Day 10 robustness run.
Writes models/model_frame_plus.csv   (the primary frame plus length, matched words, all-words arousal)
       models/model_frame_unfiltered.csv (the same variables for the no-political-filter sample)
       models/exclude_outliers.csv  (outlets beyond +/-2.5 SD of topic-adjusted arousal)
Read-only apart from those three files."""
import os, sys
import numpy as np, pandas as pd, duckdb

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
METADATA = f"{ROOT}/data/misinfo-general/metadata.db"
MIN_MATCHED = 50
FACT = {"Very Low": 0, "Low": 1, "Mixed": 2, "Mostly Factual": 3, "High": 4, "Very High": 4}
EXT = {"Least Biased": 0, "Left-Center": 1, "Right-Center": 1, "Left": 2, "Right": 2,
       "Extreme Left": 3, "Extreme Right": 3}
SIDE = {"Extreme Left": "left", "Left": "left", "Left-Center": "left", "Least Biased": "centre",
        "Right-Center": "right", "Right": "right", "Extreme Right": "right"}
FOLD = [14, 26]


def stop(msg):
    print(f"STOP: {msg}")
    sys.exit(1)


mf = pd.read_csv(f"{ROOT}/models/model_frame.csv")
need = [c for c in ("article_id", "source", "topic_cluster_model", "arousal_z", "fact_c", "part_z")
        if c not in mf.columns]
if need:
    stop(f"model_frame.csv has no column(s) {need}. It has: {list(mf.columns)}")
tab = pd.read_parquet(f"{ROOT}/pipeline/analysis_table.parquet")
if "arousal" not in tab.columns:
    stop(f"analysis_table.parquet has no raw 'arousal' column. It has: {list(tab.columns)}")
print(f"09a_frames - primary frame {len(mf):,} rows, analysis table {len(tab):,} rows")

extra = [c for c in ("log_length", "n_matched", "arousal_allwords", "year")
         if c in tab.columns and c not in mf.columns]
plus = mf.drop(columns=[c for c in ["arousal"] if c in mf.columns]).merge(tab[["article_id", "arousal"] + extra], on="article_id", how="left")
if plus.arousal.isna().any():
    stop(f"{int(plus.arousal.isna().sum())} rows of the primary frame have no arousal in the analysis table")
s = (plus.arousal.max() - plus.arousal.min()) / (plus.arousal_z.max() - plus.arousal_z.min())
m = plus.arousal.mean() - s * plus.arousal_z.mean()
err = float((plus.arousal_z - (plus.arousal - m) / s).abs().max())
print(f"   arousal z-scoring recovered: mean {m:.4f}, sd {s:.4f} (largest mismatch {err:.1e})")
if err > 0.01:
    stop("arousal_z in model_frame.csv is not a plain z-score of arousal; paste this line")
plus.to_csv(f"{ROOT}/models/model_frame_plus.csv", index=False)
print(f"   wrote model_frame_plus.csv with {', '.join(extra) if extra else 'no extra columns (none found)'}")

uf = pd.read_parquet(f"{ROOT}/pipeline/features_sample_unfiltered.parquet")
if "article_id" not in uf.columns or "arousal" not in uf.columns:
    stop(f"features_sample_unfiltered.parquet has: {list(uf.columns)}")
if "n_matched" in uf.columns:
    uf = uf[uf.n_matched >= MIN_MATCHED]
con = duckdb.connect()
con.execute(f"ATTACH '{METADATA}' AS meta (READ_ONLY)")
con.register("uf", uf[["article_id"]])
meta = con.execute(f"""
    SELECT u.article_id, m.source, m.year, tc.cluster
    FROM uf u JOIN meta.articles m ON m.article_id = u.article_id
    LEFT JOIN '{ROOT}/clustering/topic_clusters.parquet' tc ON tc.topic_id = m.topic_id""").df()
uf = uf.drop(columns=[c for c in ("source", "year", "cluster") if c in uf.columns]) \
       .merge(meta, on="article_id", how="inner")
before = len(uf)
uf = uf[uf.cluster.notna()].copy()
uf["topic_cluster_model"] = np.where(uf.cluster.astype(int).isin(FOLD), -1, uf.cluster.astype(int))
print(f"   unfiltered: {before:,} scorable articles, {len(uf):,} with a topic cluster, {uf.source.nunique()} outlets")

cols = ["mbfc_factuality", "mbfc_bias"] + (["country"] if "country" in tab.columns else [])
rate = tab.groupby("source")[cols].first()
uf = uf[uf.source.isin(rate.index)].copy()
centre = float((plus.fact - plus.fact_c).mean()) if "fact" in plus.columns else 2.332
uf["fact"] = uf.source.map(rate.mbfc_factuality.map(FACT))
uf["fact_c"] = uf.fact - centre
uf["bias_ext"] = uf.source.map(rate.mbfc_bias.map(EXT))
uf["bias_side"] = uf.source.map(rate.mbfc_bias.map(SIDE))
if "country" in rate.columns:
    uf["country"] = uf.source.map(rate.country)
fr = pd.read_csv(f"{ROOT}/partisanship/framing_model.csv").set_index("topic_cluster_model").framing_z
uf = uf[uf.topic_cluster_model.isin(fr.index)].copy()
uf["part_z"] = uf.topic_cluster_model.map(fr)
uf["arousal_z"] = (uf.arousal - m) / s
keep = ["article_id", "source", "year", "topic_cluster_model", "arousal_z", "fact", "fact_c", "part_z",
        "bias_ext", "bias_side"] + [c for c in ("country", "log_length", "n_matched", "arousal_allwords")
                                    if c in uf.columns]
uf[keep].to_csv(f"{ROOT}/models/model_frame_unfiltered.csv", index=False)
same = len(set(uf.article_id) & set(plus.article_id))
print(f"   centring constant {centre:.3f} | overlap with the primary sample: {same:,} articles "
      f"({same / len(uf):.0%} of the unfiltered frame)")
print("   unfiltered outlets per class: " + ", ".join(f"{k}={v}" for k, v in uf.groupby("fact").source.nunique().items()))
if uf.topic_cluster_model.nunique() < 20 or uf.source.nunique() < 150:
    stop("the unfiltered frame lost too many outlets or clusters; paste the lines above")

out = pd.read_csv(f"{ROOT}/models/outlet_level.csv", index_col=0)
bad = list(out.index[out.arousal_adj_z.abs() > 2.5]) if "arousal_adj_z" in out.columns else []
pd.Series(bad, name="source").to_csv(f"{ROOT}/models/exclude_outliers.csv", index=False)
print(f"   outlier outlets (>2.5 SD): {bad}")
print(f"RESULT: wrote model_frame_plus.csv ({len(plus):,} rows), model_frame_unfiltered.csv ({len(uf):,} rows), "
      f"exclude_outliers.csv")
