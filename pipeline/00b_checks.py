import os, glob, pandas as pd

H = os.path.expanduser("~")
print("PATHS")
for p in ["thesis", "thesis/data", "thesis/final_sample.xlsx", "Desktop/thesis"]:
    f = os.path.join(H, p)
    print(f"  ~/{p}: exists={os.path.exists(f)} symlink={os.path.islink(f)} -> {os.path.realpath(f)}")

print("\nCONTENTS OF ~/Desktop/thesis")
for f in sorted(os.listdir(os.path.join(H, "Desktop/thesis"))):
    print("  ", f)

print("\nPROJECT FILES")
found = {}
for name in ["political_pattern.txt", "EXCLUSIONS.md", "build_final_sample.py"]:
    found[name] = sorted({os.path.realpath(h) for root in ["thesis", "Desktop/thesis"]
                          for h in glob.glob(os.path.join(H, root, "**", name), recursive=True)})
    print(f"  {name}: {found[name] or 'NOT FOUND'}")

wb = os.path.join(H, "thesis/final_sample.xlsx")
xl = pd.ExcelFile(wb)
elig = pd.read_excel(xl, "Eligible")
src = elig["source"].dropna().astype(str)

print("\nWORKBOOK SHEETS")
for s in xl.sheet_names:
    df = pd.read_excel(xl, s)
    print(f"  {s}: {df.shape[0]} rows, columns {list(df.columns)}")
    if s != "Eligible" and "source" in df.columns:
        ov = sorted(set(df["source"].dropna().astype(str)) & set(src))
        print(f"    sources also in Eligible: {len(ov)}" + (f" {ov}" if len(ov) <= 20 else ""))

print("\nELIGIBLE SHEET CHECKS")
print(f"  rows {len(elig)}, non-null source {len(src)}, unique {src.nunique()}, "
      f"unique after strip/lower {src.str.strip().str.lower().nunique()}")
if "n_political" in elig.columns:
    print("  n_political < 200:", elig.loc[elig["n_political"] < 200, "source"].tolist())
if "political_share" in elig.columns:
    print("  political_share < 0.10:", elig.loc[elig["political_share"] < 0.10, "source"].tolist())
for c in elig.columns:
    if c != "source" and elig[c].dtype == object and elig[c].nunique(dropna=False) <= 12:
        print(f"  {c}: {elig[c].value_counts(dropna=False).to_dict()}")

for f in found["EXCLUSIONS.md"]:
    print(f"\nEXCLUSIONS.md ({f})")
    print(open(f).read()[:8000])
