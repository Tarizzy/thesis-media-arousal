import os
import duckdb, pandas as pd

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
vpath = f"{ROOT}/pipeline/political_validation.xlsx"
if os.path.exists(vpath):
    raise SystemExit("political_validation.xlsx already exists. Not overwriting it.")

con = duckdb.connect()
con.execute(f"ATTACH '{ROOT}/data/misinfo-general/metadata.db' AS meta (READ_ONLY)")
elig = pd.read_excel(f"{ROOT}/final_sample.xlsx", "Eligible")[["source"]].astype(str)
con.register("elig", elig)

ids = con.execute("""
    SELECT a.article_id, a.year
    FROM meta.articles a JOIN elig USING (source)
    WHERE a.title IS NOT NULL AND trim(a.title) <> ''
    ORDER BY a.article_id
""").df()
pick = ids.groupby("year").sample(n=50, random_state=42)

con.register("pick", pick)
val = con.execute("SELECT p.article_id, a.title FROM pick p JOIN meta.articles a USING (article_id)").df()
val = val.sort_values("article_id").sample(frac=1, random_state=42).reset_index(drop=True)
val["political"] = ""
val.to_excel(vpath, index=False)

print(f"{len(val)} headlines written to {vpath}")
print(pick["year"].value_counts().sort_index().to_string())
