"""Join the scored articles to outlet ratings and topic clusters, and write the analysis table."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb, numpy as np, pandas as pd
import preprocessing as pp

WB = f"{pp.ROOT}/COPY_final_sample_fr_filtered.xlsx"
OUT = f"{pp.ROOT}/pipeline"
LEAN_COL = "AdFontes - Political Lean 3RD PARTY"
FOLD = [14, 26]            # clusters with fewer than 200 sampled articles
MIN_KEPT_PER_OUTLET = 100  # outlets whose articles are mostly short blurbs
BIAS_ORDER = ["Extreme Left", "Left", "Left-Center", "Least Biased", "Right-Center", "Right", "Extreme Right"]
pd.set_option("display.width", 200)

elig = pd.read_excel(WB, "Eligible")
elig = elig[elig["source"].notna()].copy()
elig["source"] = elig["source"].astype(str).str.strip()
raw = elig[LEAN_COL]
elig["adfontes_lean"] = pd.to_numeric(raw, errors="coerce")
elig["adfontes_status"] = np.select(
    [elig["adfontes_lean"].notna(), raw.isna(), raw.astype(str).str.upper().str.contains("PAYWALL")],
    ["score", "blank", "paywall"], default="other text")

print(f"OUTLETS IN THE WORKBOOK: {len(elig)}")
print(elig["adfontes_status"].value_counts().to_string())
other = elig.loc[elig["adfontes_status"] == "other text", [LEAN_COL]].value_counts()
if len(other):
    print("other text values:", other.to_dict())
print("\nAD FONTES COVERAGE BY FACTUALITY")
print(pd.crosstab(elig["factuality"], elig["adfontes_status"]).to_string())

scored = elig[elig["adfontes_status"] == "score"].copy()
scored["mbfc_rank"] = scored["bias"].map({b: i for i, b in enumerate(BIAS_ORDER)})
ok = scored["mbfc_rank"].notna()
print(f"\nAD FONTES LEAN vs MBFC BIAS: Spearman {scored.loc[ok, 'adfontes_lean'].corr(scored.loc[ok, 'mbfc_rank'], method='spearman'):.3f}"
      f" over {int(ok.sum())} outlets  (negative lean = left, positive = right)")
odd = scored[(scored["bias"].isin(["Extreme Left", "Left", "Left-Center"]) & (scored["adfontes_lean"] > 5)) |
             (scored["bias"].isin(["Extreme Right", "Right", "Right-Center"]) & (scored["adfontes_lean"] < -5))]
print(f"\nOUTLETS WHERE THE TWO RATINGS DISAGREE ON DIRECTION: {len(odd)}")
if len(odd):
    print(odd[["source", "bias", "adfontes_lean", "factuality", "country"]].sort_values("adfontes_lean").to_string(index=False))

con = duckdb.connect()
con.execute(f"ATTACH '{pp.METADATA}' AS meta (READ_ONLY)")
con.register("outlets", elig[["source", "adfontes_lean", "adfontes_status", "bias", "factuality",
                              "credibility", "type", "country"]])
df = con.execute(f"""
    SELECT f.* EXCLUDE (keep), tc.cluster AS topic_cluster,
           o.bias AS mbfc_bias, o.factuality AS mbfc_factuality, o.credibility, o.type, o.country,
           o.adfontes_lean, o.adfontes_status
    FROM '{OUT}/features_sample.parquet' f
    JOIN meta.articles a USING (article_id)
    JOIN '{pp.ROOT}/clustering/topic_clusters.parquet' tc USING (topic_id)
    JOIN outlets o ON o.source = f.source
    WHERE f.keep
""").df()

per_outlet = df.groupby("source").size()
df["outlet_ok"] = df["source"].map(per_outlet >= MIN_KEPT_PER_OUTLET)
df["topic_cluster_model"] = np.where(df["topic_cluster"].isin(FOLD), -1, df["topic_cluster"])
con.register("out_df", df)
con.execute(f"COPY (SELECT * FROM out_df ORDER BY source, year, article_id) TO '{OUT}/analysis_table.parquet' (FORMAT PARQUET)")

print(f"\nANALYSIS TABLE: {len(df):,} articles | {df['source'].nunique()} outlets | "
      f"{df['topic_cluster_model'].nunique()} clusters (two folded into -1)")
print(f"missing values: " + str({c: int(n) for c, n in df.isna().sum().items() if n}))
print(f"outlets below {MIN_KEPT_PER_OUTLET} scorable articles: {int((~df['outlet_ok']).groupby(df['source']).any().sum())} "
      f"({int((~df['outlet_ok']).sum()):,} articles)")
print("\nARTICLES AND OUTLETS BY FACTUALITY")
print(df.groupby("mbfc_factuality").agg(articles=("article_id", "size"), outlets=("source", "nunique"),
                                        with_adfontes=("adfontes_lean", lambda s: s.notna().sum())).to_string())
print("\nIF AD FONTES WERE REQUIRED")
a = df[df["adfontes_lean"].notna()]
print(f"  {len(a):,} articles ({100 * len(a) / len(df):.0f}%) from {a['source'].nunique()} outlets")
print(a.groupby("mbfc_factuality").agg(articles=("article_id", "size"), outlets=("source", "nunique")).to_string())
