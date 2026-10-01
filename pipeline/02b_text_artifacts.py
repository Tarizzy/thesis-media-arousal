import os
import duckdb, pandas as pd

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
PQ = f"{ROOT}/data/misinfo-general/data/*.parquet"
QMARK = '["\u201c\u201d]'
APOS = r"apos[A-Z]|[a-z]apos(s|t|re|ve|ll|d)\b|Iaposm"
QUOT = r"quot[A-Z]|[a-z.,!?]quot\b"
ENT = r"&(quot|apos|amp|#[0-9]+);"
cols = ["has_quote_mark", "apos_artifact", "quot_artifact", "html_entity"]

con = duckdb.connect()
con.register("elig", pd.read_excel(f"{ROOT}/final_sample.xlsx", "Eligible")[["source"]].astype(str))

df = con.execute(f"""
    SELECT source, COUNT(*) AS n,
           AVG(CAST(regexp_matches(content, $q) AS INT)) AS has_quote_mark,
           AVG(CAST(regexp_matches(content, $a) AS INT)) AS apos_artifact,
           AVG(CAST(regexp_matches(content, $u) AS INT)) AS quot_artifact,
           AVG(CAST(regexp_matches(content, $e) AS INT)) AS html_entity
    FROM read_parquet('{PQ}') JOIN elig USING (source)
    GROUP BY source
""", {"q": QMARK, "a": APOS, "u": QUOT, "e": ENT}).df()

overall = (df[cols].mul(df["n"], axis=0).sum() / df["n"].sum()).round(4)
print(f"ARTICLES SCANNED: {int(df['n'].sum()):,} from {len(df)} outlets")
print("\nSHARE OF ALL ARTICLES CONTAINING EACH PATTERN\n" + overall.to_string())
for c in cols[1:]:
    top = df[df[c] > 0.01].sort_values(c, ascending=False)
    print(f"\nOUTLETS WITH {c} IN MORE THAN 1% OF ARTICLES: {len(top)}")
    if len(top):
        print(top[["source", "n"] + cols].head(20).round(3).to_string(index=False))
print("\n15 OUTLETS WITH THE LOWEST SHARE OF ARTICLES CONTAINING A QUOTATION MARK")
print(df.sort_values("has_quote_mark")[["source", "n"] + cols].head(15).round(3).to_string(index=False))

P = f"{APOS}|{QUOT}|{ENT}"
ex = con.execute(f"""
    SELECT source, regexp_extract(content, $x) AS snippet
    FROM read_parquet('{PQ}') JOIN elig USING (source)
    WHERE regexp_matches(content, $p)
    LIMIT 10
""", {"p": P, "x": ".{0,60}(?:" + P + ").{0,60}"}).df()
print("\nEXAMPLE SNIPPETS")
for s, t in ex.itertuples(index=False):
    print(f"  {s}: {t!r}")
