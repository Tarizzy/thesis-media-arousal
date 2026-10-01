import os, re, glob
import pandas as pd

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
WB = f"{ROOT}/final_sample.xlsx"
elig = pd.read_excel(WB, "Eligible")
E = set(elig["source"].dropna().astype(str))

print("FUNNEL SHEET")
print(pd.read_excel(WB, "Funnel").to_string())

print("\nCUT REASONS (All_With_Reasons)")
print(pd.read_excel(WB, "All_With_Reasons")["cut_reason"].value_counts(dropna=False).to_string())

print("\nELIGIBLE ROWS WHOSE match_notes MENTION A CUT")
hit = elig["match_notes"].fillna("").astype(str).str.contains(
    r"exclu|drop|remov|duplic|cut|clean|aggregat|republish|satire", case=False, regex=True)
print(elig.loc[hit, ["source", "match_notes"]].to_string() if hit.any() else "  none")

def compare(label, df):
    if "source" not in df.columns:
        print(f"  {label}: {df.shape}, columns {list(df.columns)[:10]}")
        return
    S = set(df["source"].dropna().astype(str))
    a, b = sorted(E - S), sorted(S - E)
    print(f"  {label}: {len(S)} sources | only in Eligible ({len(a)}): {a if len(a) <= 20 else '...'}"
          f" | only here ({len(b)}): {b if len(b) <= 20 else '...'}")

print("\nOTHER OUTLET LISTS vs Eligible (212)")
for f in ["Thesis Sample.xlsx", "final_sample+NELdata.xlsx"]:
    xl = pd.ExcelFile(f"{ROOT}/{f}")
    for s in xl.sheet_names:
        compare(f"{f} [{s}]", pd.read_excel(xl, s))
for f in ["source_manifest.csv", "source_manifest_eligible.csv"]:
    compare(f, pd.read_csv(f"{ROOT}/{f}"))

pat = open(f"{ROOT}/political_pattern.txt", encoding="utf-8").read().strip()
re.compile(pat)
print(f"\npolitical_pattern.txt: {len(pat):,} chars, ~{pat.count('|') + 1} alternatives")
print(f"  start: {pat[:120]!r}")

print("\n===== build_final_sample.py =====")
print(open(f"{ROOT}/build_final_sample.py", encoding="utf-8").read()[:20000])

LEX = f"{ROOT}/NRC-VAD-Lexicon-v2.1"
files = sorted(f for f in glob.glob(f"{LEX}/**/*", recursive=True) if os.path.isfile(f))
lang = [f for f in files if re.search(r"language|translat", f, re.I)]
main = [f for f in files if f not in lang]
print(f"\nLEXICON FILES ({len(files)} total, {len(lang)} in language/translation folders)")
for f in main:
    print(f"  {os.path.relpath(f, LEX)} ({os.path.getsize(f):,} bytes)")
for f in [f for f in main if f.lower().endswith((".txt", ".tsv", ".csv")) and "readme" not in f.lower()][:6]:
    print(f"\n== {os.path.relpath(f, LEX)}")
    lines = open(f, encoding="utf-8").read().splitlines()
    for l in lines[:4]:
        print("  " + l)
    terms = {re.split(r"[\t,]", l)[0].strip().lower() for l in lines}
    print(f"  lines {len(lines):,} | multi-word terms {sum(' ' in t for t in terms):,}")
    print("  function words present:",
          [w for w in ["the", "and", "of", "to", "a", "is", "in", "that", "it", "said"] if w in terms])
    try:
        print(pd.read_csv(f, sep="\t").describe().round(3).to_string())
    except Exception as e:
        print("  not tab-separated:", e)
