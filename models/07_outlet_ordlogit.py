"""07_outlet_ordlogit.py - confirmatory arm: outlet-level ordered logit of MBFC factuality on the outlet's
topic-adjusted mean arousal. 211 outlets, one row each. Direction is reversed relative to the mixed model
on purpose; neither arm identifies causality (see the Discussion).
Reads models/model_frame.csv. Writes models/outlet_level.csv, models/07_report.txt, models/fig_outlet.png"""
import os, sys, time
sys.path.insert(0, os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "pipeline"))
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import preprocessing as pp

OUT = f"{pp.ROOT}/models"
LEVELS = {0: "Very Low", 1: "Low", 2: "Mixed", 3: "Mostly Factual", 4: "High"}
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


t0 = time.time()
d = pd.read_csv(f"{OUT}/model_frame.csv")
out(f"07_outlet_ordlogit - {time.strftime('%Y-%m-%d %H:%M')}")
out(f"{len(d):,} articles, {d.source.nunique()} outlets")

# topic-adjusted arousal: strip cluster and year, then average within outlet, so an outlet is not
# rewarded for covering intrinsically emotive topics
fit = smf.ols("arousal_z ~ C(topic_cluster_model) + C(year)", data=d).fit()
d["resid"] = fit.resid
o = (d.groupby("source")
       .agg(fact=("fact", "first"), bias_ext=("bias_ext", "first"), bias_side=("bias_side", "first"),
            outlet_ok=("outlet_ok", "first"), articles=("arousal_z", "size"),
            arousal_raw=("arousal_z", "mean"), arousal_adj=("resid", "mean")))
for c in ["arousal_raw", "arousal_adj"]:
    o[c + "_z"] = (o[c] - o[c].mean()) / o[c].std()
o.to_csv(f"{OUT}/outlet_level.csv")
out(f"cluster and year fixed effects removed {fit.rsquared:.1%} of article arousal variance before averaging")

out("\nOUTLET MEAN AROUSAL BY FACTUALITY (z-scored across outlets)")
tab = o.groupby("fact").agg(outlets=("articles", "size"), articles=("articles", "sum"),
                            raw=("arousal_raw_z", "mean"), adjusted=("arousal_adj_z", "mean"),
                            sd=("arousal_adj_z", "std"))
tab.index = [LEVELS[i] for i in tab.index]
out(tab.to_string(float_format=lambda v: f"{v:+.3f}"))
rho = spearmanr(o.fact, o.arousal_adj_z)
out(f"Spearman rho between factuality and adjusted arousal: {rho.statistic:+.3f} (p = {rho.pvalue:.4f})")


def ordlogit(label, cols, data):
    X = data[cols]
    m = OrderedModel(data["fact"], X, distr="logit").fit(method="bfgs", disp=False)
    ci = m.conf_int()
    out(f"\n=== {label}   (n = {len(data)}, McFadden pseudo-R2 = {m.prsquared:.3f}, "
        f"converged = {m.mle_retvals['converged']})")
    res = pd.DataFrame({"coef": m.params[cols], "se": m.bse[cols], "z": m.tvalues[cols],
                        "p": m.pvalues[cols], "OR": np.exp(m.params[cols]),
                        "OR 2.5%": np.exp(ci.loc[cols, 0]), "OR 97.5%": np.exp(ci.loc[cols, 1])})
    out(res.to_string(float_format=lambda v: f"{v:.4f}"))
    return m


d1 = ordlogit("S1 raw mean arousal", ["arousal_raw_z"], o)
d2 = ordlogit("S2 topic-adjusted arousal", ["arousal_adj_z"], o)
o2 = o.join(pd.get_dummies(o.bias_side, prefix="side", drop_first=True).astype(float))
side_cols = [c for c in o2.columns if c.startswith("side_")]
d3 = ordlogit("S3 + bias controls", ["arousal_adj_z", "bias_ext"] + side_cols, o2)
ok = o[o.outlet_ok]
d4 = ordlogit("S4 excluding flagged outlets", ["arousal_adj_z"], ok)

# proportional odds: does the arousal coefficient hold across the four cut points?
out("\nPROPORTIONAL ODDS CHECK (separate logits for factuality > k; the ordered logit assumes one slope)")
rows = []
for k in range(4):
    y = (o.fact > k).astype(int)
    b = sm.Logit(y, sm.add_constant(o[["arousal_adj_z"]])).fit(disp=False)
    rows.append({"cut": f"> {LEVELS[k]}", "n_above": int(y.sum()), "coef": b.params["arousal_adj_z"],
                 "se": b.bse["arousal_adj_z"]})
pod = pd.DataFrame(rows)
out(pod.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
out(f"   spread of the four slopes: {pod.coef.max() - pod.coef.min():.3f}; "
    f"ordered-logit slope {d2.params['arousal_adj_z']:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))
for lv, g in o.groupby("fact"):
    ax.scatter(np.full(len(g), lv) + np.random.default_rng(pp.SEED).normal(0, 0.06, len(g)),
               g.arousal_adj_z, s=14, color="black", alpha=0.55)
means = o.groupby("fact").arousal_adj_z.mean()
ax.plot(means.index, means.values, color="black", lw=1.5)
ax.set_xticks(list(LEVELS), [LEVELS[i] for i in LEVELS], rotation=20)
ax.set_xlabel("MBFC factuality"); ax.set_ylabel("topic-adjusted mean arousal (SD across outlets)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_outlet.png", dpi=200); plt.close(fig)

out(f"\nRESULT: fitted S1-S4, wrote outlet_level.csv and fig_outlet.png in {time.time() - t0:.0f}s")
with open(f"{OUT}/07_report.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
