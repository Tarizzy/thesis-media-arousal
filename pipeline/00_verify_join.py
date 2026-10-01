import os, duckdb, pandas as pd

os.chdir(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")))
db = duckdb.connect("data/misinfo-general/metadata.db", read_only=True)
elig = pd.read_excel("final_sample.xlsx", "Eligible")
elig = elig[elig.source.notna()]

print(f"eligible outlets in workbook: {len(elig)}")
print("\nfirst 5 outlet names:")
print(elig.source.head().tolist())

srcs = tuple(elig.source.tolist())
print("\noutlets found in metadata.articles:")
print(db.execute(f"""
    SELECT COUNT(DISTINCT source) AS outlets_found, COUNT(*) AS articles
    FROM articles WHERE source IN {srcs}
""").df())

print("\noutlets found in parquet data:")
print(db.execute(f"""
    SELECT COUNT(DISTINCT source) AS outlets_found, COUNT(*) AS articles
    FROM read_parquet('data/misinfo-general/data/*.parquet')
    WHERE source IN {srcs}
""").df())

print("\noutlets in workbook but missing from metadata.articles:")
present = set(db.execute(f"SELECT DISTINCT source FROM articles WHERE source IN {srcs}").df().source)
missing = elig[~elig.source.isin(present)]
if len(missing) > 0:
    print(missing.source.tolist())
else:
    print("none — join is clean")
