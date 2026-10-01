"""05b_partisanship.py - coverage asymmetry delta_c per topic cluster: Monroe, Colaresi & Quinn (2008)
log-odds with an informative Dirichlet prior, outlets weighted equally. Outlet bootstrap, split-half
reliability, variants, and the pre-registered checks in 05_expectations.md.
Reads  partisanship/outlet_cluster_counts.parquet, pipeline/analysis_table.parquet, clustering/cluster_labels.csv
Writes partisanship/cluster_partisanship.csv, delta_model.csv, delta_boot.npy, 05b_report.txt,
       fig_delta_forest.png, fig_share_vs_absdelta.png"""
import os, sys, time
sys.path.insert(0, os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "pipeline"))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import preprocessing as pp

P = f"{pp.ROOT}/partisanship"
K = 33
ALPHA0 = 1_000
B_BOOT, N_SPLITS = 1_000, 200
SIDE = {"Extreme Left": "left", "Left": "left", "Left-Center": "left", "Least Biased": "centre",
        "Right-Center": "right", "Right": "right", "Extreme Right": "right"}
MIXED_PLUS = {"Mixed", "Mostly Factual", "High", "Very High"}
CONTESTED = [24, 25, 29, 28, 0]
NONPOL = [2, 3, 4, 10, 13, 14, 26, 32]
FOLD = [14, 26]
rng = np.random.default_rng(pp.SEED)
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


def delta(M, side, alpha0=ALPHA0):
    """M: outlets x clusters article counts; side: 'left' / 'right' / 'centre' per outlet.
    Returns delta (right minus left log-odds), descriptive z, and a flag for clusters one side never covers."""
    tot = M.sum(1, keepdims=True)
    S = M / tot                                   # each outlet's shares sum to 1
    L, R = side == "left", side == "right"
    n_l, n_r = tot[L].sum(), tot[R].sum()
    y_l, y_r = S[L].mean(0) * n_l, S[R].mean(0) * n_r   # outlet-weighted pseudo-counts
    a = alpha0 * S.mean(0)                        # prior: equal-weight mean share over all outlets
    d = np.log((y_r + a) / (n_r + alpha0 - y_r - a)) - np.log((y_l + a) / (n_l + alpha0 - y_l - a))
    z = d / np.sqrt(1 / (y_r + a) + 1 / (y_l + a))
    return d, z, (y_l == 0) | (y_r == 0)


def resample(strata):
    return np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata])


def merge_other(M):
    keep = [c for c in range(K) if c not in FOLD]
    return np.column_stack([M[:, keep], M[:, FOLD].sum(1)]), keep


def rho(a, b):
    return spearmanr(a, b).statistic


t0 = time.time()
counts = pd.read_parquet(f"{P}/outlet_cluster_counts.parquet")
counts["cluster"] = counts["cluster"].astype(int)
tab = pd.read_parquet(f"{pp.ROOT}/pipeline/analysis_table.parquet",
                      columns=["source", "mbfc_bias", "mbfc_factuality", "country"])
outl = tab.groupby("source").first()
outl["side"] = outl["mbfc_bias"].map(SIDE)
if outl["side"].isna().any():
    raise SystemExit(f"bias labels without a side: {outl.loc[outl.side.isna(), 'mbfc_bias'].unique()}")
labels = pd.read_csv(f"{pp.ROOT}/clustering/cluster_labels.csv").set_index("cluster")


def matrix(col):
    m = counts.pivot_table(index="source", columns="cluster", values=col, aggfunc="sum", fill_value=0)
    return m.reindex(index=outl.index, columns=range(K), fill_value=0).to_numpy(float)


M, M_all = matrix("n_political"), matrix("n_all")
side = outl["side"].to_numpy()
is_us = (outl["country"] == "USA").fillna(False).to_numpy(bool)
mixed_plus = outl["mbfc_factuality"].isin(MIXED_PLUS).to_numpy()
if (M.sum(1) == 0).any():
    raise SystemExit("an outlet has no political articles")

out(f"05b_partisanship - {time.strftime('%Y-%m-%d %H:%M')} - alpha_0 {ALPHA0:,}, {B_BOOT:,} bootstraps, {N_SPLITS} splits")
grp = pd.DataFrame({"side": side, "us": is_us, "mixed_plus": mixed_plus, "political": M.sum(1)})
summary = grp.groupby("side").agg(outlets=("side", "size"), political_articles=("political", "sum"),
                                  us_outlets=("us", "sum"), mixed_plus_outlets=("mixed_plus", "sum"))
out(summary.to_string())
out(f"outlets with unknown country (left out of the US-only variant): {int(outl['country'].isna().sum())}")

# primary estimate
d, z, _ = delta(M, side)
d_m, _, _ = delta(merge_other(M)[0], side)
keep = merge_other(M)[1]
assert np.allclose(d_m[:-1], d[keep]), "merging 14 and 26 should not move other clusters"

# bootstrap: outlets resampled within side (and within centre for the prior)
strata = [np.where(side == s)[0] for s in ("left", "right", "centre")]
boot, zero = np.empty((B_BOOT, K)), np.zeros(K)
for b in range(B_BOOT):
    take = resample(strata)
    boot[b], _, zr = delta(M[take], side[take])
    zero += zr
zero /= B_BOOT
strata_c = [np.where((side == s) & (is_us == u))[0] for s in ("left", "right", "centre") for u in (True, False)]
strata_c = [ix for ix in strata_c if len(ix)]
boot_c = np.empty((B_BOOT, K))
for b in range(B_BOOT):
    take = resample(strata_c)
    boot_c[b] = delta(M[take], side[take])[0]
np.save(f"{P}/delta_boot.npy", boot)

# split-half reliability
r_signed, r_abs = [], []
for _ in range(N_SPLITS):
    A, Bh = [], []
    for ix in strata:
        perm = rng.permutation(ix)
        A.append(perm[: len(perm) // 2]); Bh.append(perm[len(perm) // 2:])
    A, Bh = np.concatenate(A), np.concatenate(Bh)
    dA, dB = delta(M[A], side[A])[0], delta(M[Bh], side[Bh])[0]
    r_signed.append(rho(dA, dB)); r_abs.append(rho(np.abs(dA), np.abs(dB)))
sb = lambda r: 2 * np.asarray(r) / (1 + np.asarray(r))

# variants
us = is_us
variants = {
    "us_only": (M[us], side[us]),
    "mixed_plus": (M[mixed_plus], side[mixed_plus]),
    "all_articles": (M_all, side),
    "alpha_100": None, "alpha_10000": None,
}
vd, vd_other = {}, {}
for name, spec in variants.items():
    if name.startswith("alpha"):
        a0 = int(name.split("_")[1])
        vd[name] = delta(M, side, a0)[0]
        vd_other[name] = delta(merge_other(M)[0], side, a0)[0][-1]
    else:
        Mv, sv = spec
        vd[name] = delta(Mv, sv)[0]
        vd_other[name] = delta(merge_other(Mv)[0], sv)[0][-1]

# assemble
res = labels[["label", "political", "cap_code"]].copy()
res["n_political"] = M.sum(0).astype(int)
res["political_share"] = M.sum(0) / M_all.sum(0)
res["delta"], res["abs_delta"], res["z"] = d, np.abs(d), z
res["ci_lo"], res["ci_hi"] = np.percentile(boot, [2.5, 97.5], axis=0)
res["abs_ci_lo"], res["abs_ci_hi"] = np.percentile(np.abs(boot), [2.5, 97.5], axis=0)
res["ci_country_lo"], res["ci_country_hi"] = np.percentile(boot_c, [2.5, 97.5], axis=0)
res["zero_share"] = zero
res["rank_abs"] = (-res["abs_delta"]).rank(method="first").astype(int)
for name in vd:
    res[f"delta_{name}"] = vd[name]
res.to_csv(f"{P}/cluster_partisanship.csv", float_format="%.5f")

model = pd.DataFrame({"topic_cluster_model": keep + [-1], "delta": d_m})
for name in ("us_only", "mixed_plus", "all_articles"):
    model[f"delta_{name}"] = list(vd[name][keep]) + [vd_other[name]]
model["abs_delta"] = model["delta"].abs()
model.to_csv(f"{P}/delta_model.csv", index=False, float_format="%.5f")

# report
out("\nCLUSTERS BY |delta| (delta > 0: right over-covers; CI = 95% outlet bootstrap)")
show = res.sort_values("rank_abs")
tbl = pd.DataFrame({
    "rank": show["rank_abs"], "label": show["label"].str.slice(0, 30),
    "delta": show["delta"].map("{:+.3f}".format),
    "95% CI": [f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(show.ci_lo, show.ci_hi)],
    "z": show["z"].map("{:+.1f}".format), "pol_share": show["political_share"].map("{:.2f}".format),
    "n_pol": show["n_political"].map("{:,}".format),
    "zero": show["zero_share"].map(lambda v: f"{v:.1%}" if v else ""),
    "np": show["political"].map(lambda p: "" if p else "non-pol")})
out(tbl.to_string())
out(f"Other (14 + 26 merged, for the model): delta {d_m[-1]:+.3f}")
unstable = res.index[res.zero_share > 0.05].tolist()
if unstable:
    out(f"unstable: a side had zero articles in >5% of resamples for clusters {unstable}")

out("\nPRE-REGISTERED CHECKS (05_expectations.md)")
top = res["rank_abs"] <= 16
n_top = int(top[CONTESTED].sum())
m_con, m_np = res.loc[CONTESTED, "abs_delta"].mean(), res.loc[NONPOL, "abs_delta"].mean()
rule2 = n_top >= 3 and m_con > m_np
out(f"Rule 2 sanity: {n_top} of 5 contested clusters in the top half (need 3); "
    f"mean |delta| contested {m_con:.3f} vs non-political {m_np:.3f} -> {'PASS' if rule2 else 'FAIL'}")
med_abs, med_signed = np.median(sb(r_abs)), np.median(sb(r_signed))
rule3 = med_abs >= 0.7
out(f"Rule 3 split-half, |delta|: rho median {np.median(r_abs):.2f} "
    f"[{np.percentile(r_abs, 2.5):.2f}, {np.percentile(r_abs, 97.5):.2f}], Spearman-Brown {med_abs:.2f} "
    f"-> {'PASS' if rule3 else 'FAIL'}")
out(f"         signed delta: rho median {np.median(r_signed):.2f}, Spearman-Brown {med_signed:.2f}")
r_us = rho(d, vd["us_only"])
out(f"Rule 1 US-only vs all 211: Spearman rho {r_us:.3f} (|delta| {rho(np.abs(d), np.abs(vd['us_only'])):.3f}) -> "
    f"{'all 211 stay primary' if r_us >= 0.9 else 'US-only becomes primary'}")
out("Rule 4: CES validation still to run")

out("\nVARIANTS vs PRIMARY (Spearman rho, signed / |delta|; max |difference|)")
for name in vd:
    out(f"   {name:<13}{rho(d, vd[name]):6.3f} / {rho(np.abs(d), np.abs(vd[name])):.3f}   "
        f"{np.abs(d - vd[name]).max():.3f}")
w, wc = res.ci_hi - res.ci_lo, res.ci_country_hi - res.ci_country_lo
out(f"country-stratified bootstrap: median CI width {wc.median():.3f} vs {w.median():.3f} side-only")

out("\nEXPECTATIONS vs RESULT (sign, and whether the 95% CI excludes zero)")
def sign_line(c, expected):
    r = res.loc[c]
    obs = "right" if r.delta > 0 else "left"
    sure = "CI excludes 0" if (r.ci_lo > 0 or r.ci_hi < 0) else "CI includes 0"
    return f"   {c:>2} {r.label[:30]:<31} expected {expected:<6} observed {obs:<6} {sure}"
for c in [25, 24, 28, 29, 22, 0]:
    out(sign_line(c, "right"))
for c in [9, 18]:
    out(sign_line(c, "left"))
for c in [7, 23, 31, 11, 8]:
    out(f"   {c:>2} {res.loc[c, 'label'][:30]:<31} expected small |delta|  rank {res.loc[c, 'rank_abs']} of 33")
out(sign_line(2, "right") + "  (exploratory)")

# figures
fig, ax = plt.subplots(figsize=(8, 10))
f = res.sort_values("delta")
ypos = np.arange(len(f))
ax.errorbar(f.delta, ypos, xerr=[f.delta - f.ci_lo, f.ci_hi - f.delta], fmt="none", ecolor="grey", lw=1)
ax.scatter(f.delta[f.political], ypos[f.political.to_numpy()], color="black", s=18, label="political cluster")
ax.scatter(f.delta[~f.political], ypos[~f.political.to_numpy()], facecolors="white", edgecolors="black", s=18,
           label="non-political cluster")
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(ypos, [f"{c}  {l}" for c, l in zip(f.index, f.label)], fontsize=8)
ax.set_xlabel("delta (log-odds): < 0 left outlets over-cover, > 0 right outlets over-cover")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{P}/fig_delta_forest.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
pmask = res.political.to_numpy()
ax.scatter(res.political_share[pmask], res.abs_delta[pmask], color="black", s=22, label="political cluster")
ax.scatter(res.political_share[~pmask], res.abs_delta[~pmask], facecolors="white", edgecolors="black", s=22,
           label="non-political cluster")
for c, r in res.iterrows():
    ax.annotate(f"{c} {r.label[:18]}", (r.political_share, r.abs_delta), fontsize=6, xytext=(3, 2),
                textcoords="offset points")
ax.set_xlabel("political share of the cluster's articles")
ax.set_ylabel("|delta| (coverage asymmetry)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{P}/fig_share_vs_absdelta.png", dpi=200); plt.close(fig)

out(f"\nsaved cluster_partisanship.csv, delta_model.csv, delta_boot.npy and two figures in {time.time() - t0:.0f}s")
with open(f"{P}/05b_report.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
