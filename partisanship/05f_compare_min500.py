"""05f_compare_min500.py - framing divergence with 500 instead of 2,000 tokens per outlet-cluster cell
(the sensitivity declared in 05_expectations.md). Read-only."""
import os
import pandas as pd
from scipy.stats import spearmanr
os.chdir(os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "partisanship"))
p = pd.read_csv("cluster_framing.csv", index_col=0)
m = pd.read_csv("min500_cluster_framing.csv", index_col=0)
both = p.index.intersection(m.index)
rho = spearmanr(p.loc[both, "framing"], m.loc[both, "framing"]).statistic
print(f"framing divergence, 500 vs 2,000 tokens: Spearman {rho:.3f} over {len(both)} clusters")
fm = pd.read_csv("framing_model.csv").set_index("topic_cluster_model").framing
f5 = pd.read_csv("min500_framing_model.csv").set_index("topic_cluster_model").framing
print(f"model-ready values (32 levels incl. Other): Spearman {spearmanr(fm, f5.loc[fm.index]).statistic:.3f}")
top = lambda d: ", ".join(f"{c} {str(d.loc[c, 'label'])[:22]}" for c in d.framing.sort_values(ascending=False).index[:6])
print("top 6 at 2,000:", top(p))
print("top 6 at   500:", top(m))
