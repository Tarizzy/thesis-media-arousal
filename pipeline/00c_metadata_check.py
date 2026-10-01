"""00c_metadata_check.py - read-only. Do the MBFC ratings the models use (from the workbook, via the analysis
table) agree with the MBFC ratings shipped in the dataset's own metadata.db? Changes nothing.
Writes pipeline/00c_metadata_check.txt."""
import os, sys, re
sys.path.insert(0, os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "pipeline"))
import duckdb, pandas as pd
import preprocessing as pp

OUT = f"{pp.ROOT}/pipeline/00c_metadata_check.txt"
ORDER = {"verylow": 0, "low": 1, "mixed": 2, "mostlyfactual": 3, "high": 4, "veryhigh": 4}
KEYS = ["source", "publisher", "outlet", "source_name", "name", "domain", "site"]
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


def norm(v):
    return re.sub(r"[^a-z]", "", str(v).lower()) if pd.notna(v) and str(v).strip() else None


t = pd.read_parquet(f"{pp.ROOT}/pipeline/analysis_table.parquet")
wb = t.groupby("source").agg(bias=("mbfc_bias", "first"), fact=("mbfc_factuality", "first"),
                             articles=("source", "size"))
out(f"00c_metadata_check - workbook side: {len(wb)} outlets from analysis_table.parquet")

con = duckdb.connect(pp.METADATA, read_only=True)
tables = [r[0] for r in con.sql("SHOW TABLES").fetchall()]
schema = {tb: [r[0] for r in con.sql(f'DESCRIBE "{tb}"').fetchall()] for tb in tables}
out("\nmetadata.db tables:")
for tb, cols in schema.items():
    out(f"   {tb}: {cols}")

cands = []
for tb, cols in schema.items():
    low = {c.lower(): c for c in cols}
    key = next((low[k] for k in KEYS if k in low), None)
    fcol = next((c for c in cols if "factual" in c.lower() or "reliab" in c.lower()), None)
    bcol = next((c for c in cols if "bias" in c.lower() or c.lower() in ("lean", "political_lean")), None)
    ccol = next((c for c in cols if "country" in c.lower()), None)
    if key and (fcol or bcol):
        cands.append((int(bool(fcol)) + int(bool(bcol)) + int("source" in tb.lower() or "publish" in tb.lower()),
                      tb, key, fcol, bcol, ccol))
if not cands:
    out("\nRESULT: no table with an outlet key and a factuality or bias column. Paste this report to Claude.")
    open(OUT, "w").write("\n".join(lines) + "\n")
    sys.exit(0)
_, tb, key, fcol, bcol, ccol = sorted(cands, reverse=True)[0]
out(f"\nusing table '{tb}': key '{key}', factuality '{fcol}', bias '{bcol}'")
meta = con.sql(f'SELECT * FROM "{tb}"').df()
meta["_k"] = meta[key].astype(str).str.strip().str.lower()
wkeys = pd.Series(wb.index.astype(str).str.strip().str.lower(), index=wb.index)
if wkeys.isin(meta._k).mean() < 0.5:           # fall back to letters-only matching
    meta["_k"] = meta[key].map(norm)
    wkeys = pd.Series([norm(s) for s in wb.index], index=wb.index)
out(f"outlets found in metadata.db: {int(wkeys.isin(meta._k).sum())} of {len(wb)}")

rows = []
for src, r in wb.iterrows():
    m = meta[meta._k == wkeys[src]]
    rec = {"source": src, "articles": r.articles, "workbook factuality": r.fact, "workbook bias": r.bias}
    if m.empty:
        rec["status"] = "not in metadata.db"
        rows.append(rec)
        continue
    mf = sorted({str(v) for v in m[fcol].dropna()}) if fcol else []
    mb = sorted({str(v) for v in m[bcol].dropna()}) if bcol else []
    rec["metadata factuality"], rec["metadata bias"] = " | ".join(mf), " | ".join(mb)
    wf, wbias = norm(r.fact), norm(r.bias)
    f_exact = wf in {norm(v) for v in mf}
    f_merged = ORDER.get(wf) in {ORDER.get(norm(v)) for v in mf}
    steps = [abs(ORDER[wf] - ORDER[norm(v)]) for v in mf if wf in ORDER and norm(v) in ORDER]
    rec["factuality"] = ("agree" if f_exact else "agree once Very High = High" if f_merged
                         else f"DISAGREE by {min(steps)} step(s)" if steps else "DISAGREE (unmapped label)") if mf else "no value"
    rec["bias"] = ("agree" if wbias in {norm(v) for v in mb} else "DISAGREE") if mb else "no value"
    rec["several ratings in metadata"] = len(mf) > 1 or len(mb) > 1
    rows.append(rec)
res = pd.DataFrame(rows)
res.to_csv(OUT.replace(".txt", ".csv"), index=False)

out("\nFACTUALITY (the model's outcome)")
out(res.factuality.fillna("-").value_counts().to_string())
out("\nBIAS")
out(res.bias.fillna("-").value_counts().to_string())
cols = ["source", "articles", "workbook factuality", "metadata factuality", "workbook bias", "metadata bias"]
bad_f = res[res.factuality.fillna("").str.startswith("DISAGREE")]
bad_b = res[res.bias.fillna("").str.startswith("DISAGREE")]
several = res[res.get("several ratings in metadata", pd.Series(False, index=res.index)).fillna(False).astype(bool)]
missing = res[res.status.eq("not in metadata.db")] if "status" in res else res.iloc[0:0]
for title, d in [("FACTUALITY DISAGREEMENTS", bad_f), ("BIAS DISAGREEMENTS", bad_b),
                 ("OUTLETS WITH SEVERAL RATINGS IN metadata.db", several), ("NOT FOUND IN metadata.db", missing)]:
    out(f"\n{title}: {len(d)} outlets, {int(d.articles.sum()) if len(d) else 0:,} articles")
    if len(d):
        out(d[[c for c in cols if c in d.columns] + (["factuality"] if "factuality" in d else [])]
            .to_string(index=False))
if ccol:
    wc = t.groupby("source").country.first()
    nocountry = wc[wc.isna()].index
    out(f"\nCOUNTRY in metadata.db for the {len(nocountry)} outlets with no country in the workbook")
    for src in nocountry:
        vals = meta.loc[meta._k == wkeys[src], ccol].dropna().astype(str).unique()
        out(f"   {src:<24}{', '.join(vals) if len(vals) else 'not listed'}")
out(f"\nRESULT: {'no factuality disagreements' if bad_f.empty else f'{len(bad_f)} factuality disagreements to review'}"
    f" - full table in 00c_metadata_check.csv. Nothing was changed.")
open(OUT, "w").write("\n".join(lines) + "\n")
