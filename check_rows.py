import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 30)
WB = "COPY_final_sample_fr_filtered.xlsx"
FACT = ["Very High", "High", "Mostly Factual", "Mixed", "Low", "Very Low"]
BIAS = ["Extreme Left", "Left", "Left-Center", "Least Biased", "Right-Center", "Right", "Extreme Right"]
LEAN = "AdFontes - Political Lean 3RD PARTY"
COLS = ["source", LEAN, "label", "bias", "factuality", "credibility", "type", "country"]

df = pd.read_excel(WB, "Eligible")
df = df[df["source"].notna()]
df["excel_row"] = df.index + 2
bad_fact = ~df["factuality"].isin(FACT)
bad_bias = ~df["bias"].isin(BIAS)
lean_num = pd.to_numeric(df[LEAN], errors="coerce")
bad_lean = df[LEAN].notna() & lean_num.isna() & ~df[LEAN].astype(str).str.upper().str.replace(" ", "").eq("PAYWALL")

print(f"rows: {len(df)} | bad factuality: {bad_fact.sum()} | bad bias: {bad_bias.sum()} | bad lean text: {bad_lean.sum()}")
print("\nROWS TO FIX")
print(df.loc[bad_fact | bad_bias | bad_lean, ["excel_row"] + COLS].to_string(index=False))
print("\nA CORRECT ROW FOR COMPARISON")
print(df.loc[~(bad_fact | bad_bias), ["excel_row"] + COLS].head(2).to_string(index=False))
print(f"\nusable Ad Fontes scores: {int(lean_num.notna().sum())} | paywalled: {int(df[LEAN].astype(str).str.upper().str.replace(' ', '').eq('PAYWALL').sum())}")
