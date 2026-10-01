import os, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import duckdb, pandas as pd
import preprocessing as pp

con = duckdb.connect()
con.execute(f"ATTACH '{pp.METADATA}' AS meta (READ_ONLY)")
P = {"pattern": pp.political_pattern()}
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 90)

print("1. KEEP FLAG vs n_matched, first run and now")
for f in ["features_sample_v1.parquet", "features_sample.parquet"]:
    print(con.execute(f"""SELECT '{f}' AS file, COUNT(*) AS articles, SUM(CAST(keep AS INT)) AS keep_true,
        SUM(CAST(n_matched >= 50 AS INT)) AS matched_50_plus, ROUND(AVG(n_matched), 1) AS mean_n_matched
        FROM '{f}'""").df().to_string(index=False))
print(con.execute("""SELECT a.keep AS keep_first_run, b.keep AS keep_now, COUNT(*) AS articles
    FROM 'features_sample_v1.parquet' a JOIN 'features_sample.parquet' b USING (article_id)
    GROUP BY ALL ORDER BY ALL""").df().to_string(index=False))

print("\n2. NAME RULE ON 500 SAMPLED ARTICLES (capitalised occurrences skipped as names)")
texts = con.execute("SELECT content FROM 'sample.parquet' ORDER BY hash(article_id) LIMIT 500").df()["content"]
cap, skipped = Counter(), Counter()
for text in texts:
    t, _ = pp.clean(text)
    toks, low, proper = pp.tokens(t)
    for w, lw, p in zip(toks, low, proper):
        if lw in ("trump", "house", "president", "republican", "republicans", "democrats", "united") and w[0].isupper():
            cap[lw] += 1
            skipped[lw] += int(p)
print(pd.DataFrame({"capitalised": cap, "skipped_as_name": skipped}).fillna(0).astype(int).to_string())

print("\n3. SAMPLED ARTICLES vs METADATA")
print(con.execute(f"""
    SELECT COUNT(*) AS sampled, COUNT(m.article_id) AS found_in_metadata,
           SUM(CAST(m.source <> s.source AS INT)) AS source_differs,
           SUM(CAST(coalesce(m.title, '') <> coalesce(s.title, '') AS INT)) AS title_differs,
           SUM(CAST(NOT {pp.political_sql('m.title')} AS INT)) AS not_political_by_metadata_title,
           SUM(CAST(NOT {pp.political_sql('s.title')} AS INT)) AS not_political_by_sample_title
    FROM 'sample.parquet' s LEFT JOIN meta.articles m USING (article_id)
""", P).df().T.to_string())
print(con.execute(f"""
    SELECT s.source, m.title AS metadata_title, s.title AS sample_title
    FROM 'sample.parquet' s JOIN meta.articles m USING (article_id)
    WHERE NOT {pp.political_sql('s.title')} ORDER BY hash(s.article_id) LIMIT 6
""", P).df().to_string(index=False))
