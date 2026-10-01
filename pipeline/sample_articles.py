"""Draw the analysis samples.
sample.parquet: up to 300 political articles per eligible outlet (keyword list v2.1 on repaired headlines),
allocated across years in proportion to the outlet's political articles per year, drawn at random within year.
sample_unfiltered.parquet: the same design over all articles, for a robustness check."""
import os, sys
from importlib.metadata import version
import duckdb, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocessing as pp

OUT = f"{pp.ROOT}/pipeline"
L = []
def say(*parts):
    line = " ".join(str(p) for p in parts)
    L.append(line)
    print(line, flush=True)

say(f"seed {pp.SEED} | numpy {version('numpy')} | pandas {version('pandas')} | duckdb {version('duckdb')}")
E = pp.eligible_sources()
if len(E) != 212 or len(set(E)) != 212:
    raise SystemExit(f"Eligible sheet has {len(E)} sources ({len(set(E))} unique), expected 212. Stop here.")

con = duckdb.connect()
con.execute(f"ATTACH '{pp.METADATA}' AS meta (READ_ONLY)")

# ---------- 1. flag political articles (one pass over every headline) ----------
arts = con.execute(f"""SELECT article_id, source, CAST(year AS INTEGER) AS year, {pp.political_sql()} AS political
                       FROM meta.articles""", {"pattern": pp.political_pattern()}).df()
src = con.execute("""SELECT source, CAST(bias AS VARCHAR) AS bias, CAST(factuality AS VARCHAR) AS factuality,
                            CAST(label AS VARCHAR) AS label FROM meta.sources""").df()
c = src.merge(arts.groupby("source").agg(n_articles=("article_id", "size"), n_political=("political", "sum")),
              on="source", how="left").fillna({"n_articles": 0, "n_political": 0})
c["political_share"] = np.where(c["n_articles"] > 0, c["n_political"] / c["n_articles"].clip(lower=1), 0.0)

has_bias = c["bias"].notna()
has_fact = has_bias & c["factuality"].notna()
not_type = has_fact & ~c["label"].isin(["Satire", "Pro-Science"])
enough = not_type & (c["n_political"] >= pp.MIN_POLITICAL)
share = enough & (c["political_share"] >= pp.MIN_SHARE)
elig_v21 = set(c.loc[share, "source"])
say("\nFUNNEL (keyword list v2.1)")
for stage, mask in [("All sources", pd.Series(True, index=c.index)), ("Has bias", has_bias),
                    ("Has factuality", has_fact), ("Not satire/pro-science", not_type),
                    (f"Political articles >= {pp.MIN_POLITICAL}", enough),
                    (f"Political share >= {pp.MIN_SHARE:.0%}", share)]:
    say(f"  {stage:<30} {int(mask.sum())}")
not_in_sheet = sorted(elig_v21 - set(E))
say(f"  {'Sampled (Eligible sheet)':<30} {len(E)}")
say(f"  eligible under v2.1 but not sampled (no Ad Fontes score): {len(not_in_sheet)} {not_in_sheet}")
if not set(E) <= elig_v21:
    raise SystemExit(f"Eligible outlets failing v2.1 thresholds: {sorted(set(E) - elig_v21)}. Stop here.")
c[c["source"].isin(E)].sort_values("n_political", ascending=False).to_csv(f"{OUT}/outlet_counts_v21.csv", index=False)

# ---------- 2. stratified draw ----------
def allocate(by_year, n_target):
    exact = by_year / by_year.sum() * n_target
    alloc = np.floor(exact).astype(int)
    order = np.argsort(-(exact - alloc).to_numpy(), kind="stable")
    alloc.iloc[order[: n_target - int(alloc.sum())]] += 1
    return alloc

def draw(pool, rng):
    picks = []
    for source, g in pool.groupby("source", sort=True):
        by_year = g.groupby("year").size().sort_index()
        alloc = allocate(by_year, min(pp.MAX_PER_OUTLET, int(by_year.sum())))
        for year, k in alloc.items():
            if k:
                ids = np.sort(g.loc[g["year"] == year, "article_id"].to_numpy(dtype=object))
                picks.append(pd.DataFrame({"article_id": rng.choice(ids, size=int(k), replace=False),
                                           "source": source, "year": year}))
    return pd.concat(picks, ignore_index=True)

pool = arts[arts["source"].isin(E)]
samples = {"sample": draw(pool[pool["political"]], np.random.default_rng([pp.SEED, 0])),
           "sample_unfiltered": draw(pool, np.random.default_rng([pp.SEED, 1]))}

# ---------- 3. attach text from the parquet files ----------
ids = pd.concat([s[["article_id"]] for s in samples.values()]).drop_duplicates()
con.register("ids", ids)
text = con.execute(f"""SELECT p.article_id, p.title, p.content, p.publication_date, p.domain
                       FROM read_parquet('{pp.PARQUET}') p JOIN ids USING (article_id)""").df()
if text["article_id"].duplicated().any():
    raise SystemExit("Duplicate article_id in the parquet files. Stop here.")

for name, s in samples.items():
    s = s.merge(text, on="article_id", how="left")
    per_outlet = s.groupby("source").size()
    say(f"\n{name.upper()}: {len(s):,} articles from {s['source'].nunique()} outlets")
    say(f"  per outlet: min {per_outlet.min()}, max {per_outlet.max()}, outlets with fewer than 300: {int((per_outlet < 300).sum())}")
    say(f"  duplicate ids: {int(s['article_id'].duplicated().sum())} | missing text: {int(s['content'].isna().sum())} | "
        f"empty text: {int((s['content'].fillna('').str.strip() == '').sum())}")
    say("  articles per year: " + ", ".join(f"{y} {n:,}" for y, n in s.groupby("year").size().items()))
    con.register("s_out", s)
    con.execute(f"COPY (SELECT * FROM s_out ORDER BY source, year, article_id) TO '{OUT}/{name}.parquet' (FORMAT PARQUET)")
    con.unregister("s_out")
    s.sort_values(["source", "year", "article_id"])[["article_id", "source", "year", "publication_date", "title"]] \
        .to_csv(f"{OUT}/{name}_index.csv", index=False)

with open(f"{OUT}/sample_articles.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(L))
