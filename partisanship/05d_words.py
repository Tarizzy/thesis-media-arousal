"""05d_words.py - word counts per outlet and topic cluster, for the framing-divergence measure.
Samples up to CAP political articles per (outlet, cluster), tokenises them in DuckDB, and keeps a fixed
vocabulary. Outlets stay separate so 05e can do split-half and bootstrap over outlets.
Writes partisanship/word_counts.parquet, partisanship/vocab.parquet, partisanship/05d_report.txt.
These counts feed the partisanship measure only - never the article features."""
import os, sys, time
sys.path.insert(0, os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "pipeline"))
import duckdb, pandas as pd
import preprocessing as pp

P = f"{pp.ROOT}/partisanship"
os.makedirs(f"{P}/duckdb_tmp", exist_ok=True)
CAP = 60          # political articles per outlet per cluster
VOCAB = 10_000    # words kept, by total frequency
MIN_OUTLETS = 20  # a word must appear for at least this many outlets
MIN_LEN = 3
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


t0 = time.time()
con = duckdb.connect()
con.execute(f"SET temp_directory='{P}/duckdb_tmp'")
con.execute("SET preserve_insertion_order=false")
con.execute(f"ATTACH '{pp.METADATA}' AS meta (READ_ONLY)")
con.register("outlets", pd.read_parquet(f"{pp.ROOT}/pipeline/analysis_table.parquet",
                                        columns=["source"]).drop_duplicates())
con.register("stop", pd.DataFrame({"w": sorted(pp.STOPWORDS | pp.FUNCTION_WORDS)}))

# 1. sample the articles and clean the text once
con.execute(f"""
CREATE OR REPLACE TABLE samp AS
WITH pol AS (
    SELECT p.article_id, p.source, tc.cluster
    FROM read_parquet('{pp.PARQUET}') p
    JOIN outlets o ON o.source = p.source
    JOIN meta.articles a ON a.article_id = p.article_id
    JOIN '{pp.ROOT}/clustering/topic_clusters.parquet' tc ON tc.topic_id = a.topic_id
    WHERE {pp.political_sql('p.title')}
), ranked AS (
    SELECT *, row_number() OVER (PARTITION BY source, cluster ORDER BY hash(article_id)) AS rn FROM pol
)
SELECT s.source, s.cluster,
       regexp_replace(lower(p.content), '<copyright>|<twitter>|<url>|<?selfref>?', ' ', 'g') AS txt
FROM (SELECT * FROM ranked WHERE rn <= {CAP}) s
JOIN read_parquet('{pp.PARQUET}') p ON p.article_id = s.article_id
""", {"pattern": pp.political_pattern()})
n_art = con.sql("SELECT count(*) FROM samp").fetchone()[0]
out(f"05d_words - {time.strftime('%Y-%m-%d %H:%M')} - cap {CAP} articles per outlet x cluster")
out(f"sampled {n_art:,} political articles in {time.time() - t0:.0f}s")

# 2. vocabulary: frequent words that many outlets use. Tokens are aggregated on the fly in both passes,
#    never materialised, so memory stays bounded by the number of groups rather than the token count.
TOKENS = (f"SELECT source, cluster, w FROM ("
          f"SELECT source, cluster, unnest(regexp_split_to_array(txt, '[^a-z]+')) AS w FROM samp) "
          f"WHERE length(w) >= {MIN_LEN} AND w NOT IN (SELECT w FROM stop)")
con.execute(f"""
CREATE OR REPLACE TABLE vocab AS
SELECT w, sum(n) AS total, count(*) AS outlets FROM (
    SELECT w, source, count(*) AS n FROM ({TOKENS}) GROUP BY w, source
) GROUP BY w HAVING outlets >= {MIN_OUTLETS} ORDER BY total DESC LIMIT {VOCAB}
""")
n_vocab, kept = con.sql("SELECT count(*), sum(total) FROM vocab").fetchone()
out(f"vocabulary {n_vocab:,} words, {kept:,} tokens kept after {time.time() - t0:.0f}s")

# 3. counts per outlet x cluster x word
con.execute(f"""
COPY (
    SELECT t.source, t.cluster, t.w AS word, count(*) AS n
    FROM ({TOKENS}) t JOIN vocab v ON v.w = t.w
    GROUP BY ALL
) TO '{P}/word_counts.parquet' (FORMAT PARQUET)
""")
con.execute(f"COPY (SELECT * FROM vocab) TO '{P}/vocab.parquet' (FORMAT PARQUET)")
wc = con.sql(f"SELECT count(*) AS rows, count(DISTINCT source) AS outlets, count(DISTINCT cluster) AS clusters, "
             f"sum(n) AS tokens FROM '{P}/word_counts.parquet'").df().iloc[0]
out(f"word_counts.parquet: {int(wc['rows']):,} rows | {int(wc.outlets)} outlets | {int(wc.clusters)} clusters "
    f"| {int(wc.tokens):,} tokens")

cells = con.sql("SELECT source, cluster, count(*) AS n FROM samp GROUP BY ALL").df()
out(f"outlet x cluster cells with articles: {len(cells):,} of {int(wc.outlets) * 33:,} | "
    f"median articles per cell {cells.n.median():.0f} | cells at the cap: {(cells.n == CAP).mean():.0%}")
thin = cells.groupby("cluster").n.sum().sort_values().head(5)
out("thinnest clusters (sampled articles): " + ", ".join(f"{int(c)}:{int(v):,}" for c, v in thin.items()))
out("most frequent words: " + ", ".join(con.sql("SELECT w FROM vocab LIMIT 15").df().w))
problems = []
if int(wc.outlets) != con.sql("SELECT count(*) FROM outlets").fetchone()[0]:
    problems.append("some outlets produced no words")
if int(wc.clusters) != 33:
    problems.append("not all 33 clusters are present")
out(f"\nRESULT: {'OK' if not problems else 'CHECK: ' + '; '.join(problems)} - total {time.time() - t0:.0f}s")
with open(f"{P}/05d_report.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
