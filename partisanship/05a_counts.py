"""05a_counts.py - outlet x cluster x year counts of all and political articles, full corpus, 211 outlets.
Same political filter and cluster join as the sampling and 04_analysis_table.py.
Writes partisanship/outlet_cluster_counts.parquet and partisanship/05a_report.txt."""
import os, sys, time
sys.path.insert(0, os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "pipeline"))
import duckdb, pandas as pd
import preprocessing as pp

OUT = f"{pp.ROOT}/partisanship"
os.makedirs(OUT, exist_ok=True)
TABLE = f"{pp.ROOT}/pipeline/analysis_table.parquet"
TOPIC_CLUSTERS = f"{pp.ROOT}/clustering/topic_clusters.parquet"
REFERENCE = f"{pp.ROOT}/pipeline/outlet_counts_v21.csv"
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


t0 = time.time()
con = duckdb.connect()
con.execute(f"ATTACH '{pp.METADATA}' AS meta (READ_ONLY)")
outlets = pd.read_parquet(TABLE, columns=["source"]).drop_duplicates()
con.register("outlets", outlets)

counts = con.execute(f"""
    SELECT p.source,
           tc.cluster,
           TRY_CAST(left(CAST(p.publication_date AS VARCHAR), 4) AS INTEGER) AS year,
           count(*) AS n_all,
           count(*) FILTER (WHERE {pp.political_sql('p.title')}) AS n_political
    FROM read_parquet('{pp.PARQUET}') p
    JOIN outlets o ON o.source = p.source
    LEFT JOIN meta.articles a ON a.article_id = p.article_id
    LEFT JOIN '{TOPIC_CLUSTERS}' tc ON tc.topic_id = a.topic_id
    GROUP BY ALL
""", {"pattern": pp.political_pattern()}).df()

out(f"05a_counts - {time.strftime('%Y-%m-%d %H:%M')} - query took {time.time() - t0:.0f}s")
out(f"articles {int(counts.n_all.sum()):,} | political {int(counts.n_political.sum()):,} "
    f"({counts.n_political.sum() / counts.n_all.sum():.1%}) | outlets {counts.source.nunique()} of {len(outlets)}")

# checks
problems = []
no_cluster = int(counts.loc[counts.cluster.isna(), "n_all"].sum())
out(f"articles with no cluster: {no_cluster:,}")
if no_cluster:
    problems.append("articles without a cluster")
if counts.source.nunique() != len(outlets):
    missing = sorted(set(outlets.source) - set(counts.source))
    problems.append(f"outlets with no articles: {missing}")
clusters = sorted(counts.cluster.dropna().astype(int).unique())
out(f"clusters present: {len(clusters)} ({clusters[0]}-{clusters[-1]})")
if clusters != list(range(33)):
    problems.append("cluster ids are not 0-32")

yr = counts.groupby("year", dropna=False)[["n_all", "n_political"]].sum()
out("\nBY YEAR")
out(yr.to_string())
bad_year = int(counts.loc[~counts.year.between(2017, 2022), "n_all"].sum())
if bad_year:
    out(f"articles with no usable year: {bad_year:,} (year is only needed for optional variants)")

# the political totals must reproduce the eligibility counts exactly
per_outlet = counts.groupby("source")[["n_all", "n_political"]].sum()
if os.path.exists(REFERENCE):
    ref = pd.read_csv(REFERENCE)
    pcol = [c for c in ref.columns if "polit" in c.lower() and "share" not in c.lower()]
    if "source" in ref.columns and pcol:
        cmp = per_outlet.join(ref.set_index("source")[pcol[0]].rename("reference"), how="left")
        diff = cmp[cmp.n_political != cmp.reference]
        out(f"\nRECONCILIATION with {os.path.basename(REFERENCE)} (column '{pcol[0]}'): "
            f"{len(cmp) - len(diff)} of {len(cmp)} outlets match exactly")
        if len(diff):
            out(diff.head(15).to_string())
            problems.append(f"{len(diff)} outlets differ from the reference political counts")
    else:
        out(f"\nreference file columns {list(ref.columns)} - reconciliation skipped")
else:
    out("\nreference file not found - reconciliation skipped")

by_cluster = counts.groupby("cluster")[["n_all", "n_political"]].sum()
by_cluster["political_share"] = by_cluster.n_political / by_cluster.n_all
out("\nBY CLUSTER (all 211 outlets, full corpus)")
out(by_cluster.to_string(formatters={"political_share": "{:.2f}".format}))

counts["cluster"] = counts["cluster"].astype("Int64")
counts.to_parquet(f"{OUT}/outlet_cluster_counts.parquet", index=False)
out(f"\nRESULT: {'OK' if not problems else 'CHECK: ' + '; '.join(problems)}")
out(f"saved outlet_cluster_counts.parquet ({len(counts):,} rows) in {time.time() - t0:.0f}s")
with open(f"{OUT}/05a_report.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
