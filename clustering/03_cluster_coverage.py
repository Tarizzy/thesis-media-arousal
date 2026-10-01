import os, sys
sys.path.insert(0, os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "pipeline"))
import duckdb, pandas as pd
import preprocessing as pp

LABELS = {0: "Latin America / border", 1: "Cannabis, tobacco, diet", 2: "US pro sports", 3: "Music and pop culture",
          4: "Celebrity, reality TV, soaps", 5: "Russia and Eastern Europe", 6: "Asia-Pacific", 7: "Transport",
          8: "US Congress and campaigns", 9: "Energy and climate", 10: "Film, TV, theatre", 11: "Markets and big tech",
          12: "US local news", 13: "Lifestyle and human interest", 14: "UK football and rugby", 15: "Crime and courts",
          16: "Mixed named figures", 17: "UK domestic affairs", 18: "Economy and welfare", 19: "Defence",
          20: "UK party politics, Brexit", 21: "US investigations", 22: "Partisan media", 23: "Space and nature",
          24: "Abortion and rights", 25: "Election administration", 26: "Olympic sports", 27: "Middle East",
          28: "Disease and COVID", 29: "Race and extremism", 30: "Education", 31: "Technology and cyber",
          32: "European club football"}
NON_POLITICAL = [2, 3, 4, 10, 13, 14, 26, 32]

con = duckdb.connect()
con.execute(f"ATTACH '{pp.METADATA}' AS meta (READ_ONLY)")
df = con.execute(f"""
    WITH a AS (SELECT m.article_id, tc.cluster, {pp.political_sql('m.title')} AS political
               FROM meta.articles m JOIN '{pp.ROOT}/clustering/topic_clusters.parquet' tc USING (topic_id)),
         s AS (SELECT article_id FROM '{pp.ROOT}/pipeline/features_sample.parquet' WHERE keep)
    SELECT a.cluster, COUNT(*) AS corpus_articles,
           ROUND(AVG(CAST(a.political AS INT)), 3) AS political_share,
           COUNT(s.article_id) AS sampled_kept
    FROM a LEFT JOIN s USING (article_id)
    GROUP BY a.cluster ORDER BY sampled_kept
""", {"pattern": pp.political_pattern()}).df()
df["label"] = df["cluster"].map(LABELS)
df["sampled_pct"] = (100 * df["sampled_kept"] / df["sampled_kept"].sum()).round(2)
pd.set_option("display.width", 200)
print(df[["cluster", "label", "corpus_articles", "political_share", "sampled_kept", "sampled_pct"]].to_string(index=False))
print(f"\nsampled articles total: {df['sampled_kept'].sum():,}")
print(f"in the 8 non-political clusters: {df[df['cluster'].isin(NON_POLITICAL)]['sampled_kept'].sum():,} "
      f"({df[df['cluster'].isin(NON_POLITICAL)]['sampled_pct'].sum():.1f}%)")
print(f"clusters with fewer than 200 sampled articles: {int((df['sampled_kept'] < 200).sum())}")
