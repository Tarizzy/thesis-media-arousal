"""07b_outliers.py - which outlets sit at the extremes of topic-adjusted arousal (07's figure). Read-only."""
import os, pandas as pd
T = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
o = pd.read_csv(f"{T}/models/outlet_level.csv", index_col=0)
o["factuality"] = o.fact.map({0: "Very Low", 1: "Low", 2: "Mixed", 3: "Mostly Factual", 4: "High"})
cols = ["factuality", "bias_side", "bias_ext", "articles", "outlet_ok", "arousal_raw_z", "arousal_adj_z"]
f = lambda v: f"{v:+.2f}"
print("MOST AROUSED OUTLETS (topic-adjusted, SD across outlets)")
print(o.sort_values("arousal_adj_z", ascending=False)[cols].head(8).to_string(float_format=f))
print("\nLEAST AROUSED OUTLETS")
print(o.sort_values("arousal_adj_z")[cols].head(8).to_string(float_format=f))
print(f"\noutlets beyond +/-2.5 SD: {int((o.arousal_adj_z.abs() > 2.5).sum())}")
