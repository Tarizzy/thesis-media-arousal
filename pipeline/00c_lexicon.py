import os, re, glob
import pandas as pd

base = os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "data")
files = sorted(f for f in glob.glob(f"{base}/**/*", recursive=True)
               if "vad" in f.lower() and os.path.isfile(f))
print("FILES")
for f in files:
    print(f"  {f}  ({os.path.getsize(f):,} bytes)")

for f in files:
    if not f.lower().endswith((".txt", ".tsv", ".csv")):
        continue
    print(f"\n== {f}")
    lines = open(f, encoding="utf-8").read().splitlines()
    for l in lines[:5]:
        print("  " + l)
    terms = {re.split(r"[\t,]", l)[0].strip().lower() for l in lines}
    print(f"  lines {len(lines):,} | terms containing a space {sum(' ' in t for t in terms):,}")
    print("  function words present:",
          [w for w in ["the", "and", "of", "to", "a", "is", "in", "that", "it", "said"] if w in terms])
    try:
        print(pd.read_csv(f, sep=None, engine="python").describe().round(3).to_string())
    except Exception as e:
        print("  not a table:", e)
