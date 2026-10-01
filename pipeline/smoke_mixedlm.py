import time
import duckdb, numpy as np, pandas as pd, statsmodels.formula.api as smf

df = duckdb.sql("SELECT arousal, source, topic_cluster_model, year FROM 'analysis_table.parquet'").df()
df["topic_cluster_model"] = df["topic_cluster_model"].astype(str)
df["grp"] = 1
for n in (5000, 20000):
    d = df.sample(n=n, random_state=42)
    t = time.time()
    m = smf.mixedlm("arousal ~ C(year)", d, groups="grp",
                    vc_formula={"outlet": "0 + C(source)", "cluster": "0 + C(topic_cluster_model)"}).fit()
    print(f"n={n:,}: {time.time() - t:.0f}s | outlet var {m.vcomp[0]:.5f} | cluster var {m.vcomp[1]:.5f} | converged {m.converged}", flush=True)
