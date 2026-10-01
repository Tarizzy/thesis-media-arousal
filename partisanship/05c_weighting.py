"""05c_weighting.py - diagnose why the coverage-asymmetry measure failed rule 3 and test a fixed set of
weighting schemes by split-half reliability. The outcome variable (arousal) plays no part here.
Writes partisanship/05c_report.txt and partisanship/delta_by_scheme.csv."""
import os, sys, time
sys.path.insert(0, os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "pipeline"))
import numpy as np, pandas as pd
from scipy.stats import spearmanr, trim_mean
import preprocessing as pp

P = f"{pp.ROOT}/partisanship"
K, ALPHA0, ALPHA_OUTLET, N_SPLITS = 33, 1_000, 20, 200
SIDE = {"Extreme Left": "left", "Left": "left", "Left-Center": "left", "Least Biased": "centre",
        "Right-Center": "right", "Right": "right", "Extreme Right": "right"}
BIG = 1_000              # "large outlets only" scheme: at least this many political articles
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


counts = pd.read_parquet(f"{P}/outlet_cluster_counts.parquet")
counts["cluster"] = counts["cluster"].astype(int)
tab = pd.read_parquet(f"{pp.ROOT}/pipeline/analysis_table.parquet", columns=["source", "mbfc_bias"])
outl = tab.groupby("source").first()
outl["side"] = outl["mbfc_bias"].map(SIDE)
M = (counts.pivot_table(index="source", columns="cluster", values="n_political", aggfunc="sum", fill_value=0)
     .reindex(index=outl.index, columns=range(K), fill_value=0).to_numpy(float))
side = outl["side"].to_numpy()
n_o = M.sum(1)
S = M / n_o[:, None]
labels = pd.read_csv(f"{pp.ROOT}/clustering/cluster_labels.csv").set_index("cluster")["label"]


def wmean(X, w):
    w = w / w.sum()
    return (X * w[:, None]).sum(0)


def delta_from_shares(p_l, p_r, pi, n_l, n_r, alpha0=ALPHA0):
    a = alpha0 * pi
    y_l, y_r = p_l * n_l, p_r * n_r
    return (np.log((y_r + a) / (n_r + alpha0 - y_r - a))
            - np.log((y_l + a) / (n_l + alpha0 - y_l - a)))


def scheme_delta(name, idx):
    """idx: outlet positions to use. Returns delta for one weighting scheme."""
    s, sd, n = S[idx], side[idx], n_o[idx]
    L, R = sd == "left", sd == "right"
    if not (L.any() and R.any()):
        return np.full(K, np.nan)
    if name == "equal":
        w = np.ones(len(idx))
    elif name == "sqrt_n":
        w = np.sqrt(n)
    elif name == "pooled":
        w = n.copy()
    elif name == "capped":
        w = np.minimum(n, np.quantile(n, 0.9))
    elif name == "trim10":
        p_l, p_r = trim_mean(s[L], 0.1, axis=0), trim_mean(s[R], 0.1, axis=0)
        pi = s.mean(0)
        return delta_from_shares(p_l / p_l.sum(), p_r / p_r.sum(), pi / pi.sum(), n[L].sum(), n[R].sum())
    elif name == "big_only":
        keep = n >= BIG
        if not (L & keep).any() or not (R & keep).any():
            return np.full(K, np.nan)
        s, L, R, n = s[keep], L[keep], R[keep], n[keep]
        w = np.ones(len(n))
    elif name == "mean_logodds":
        pi = s.mean(0)
        sm = (s * n[:, None] + ALPHA_OUTLET * pi) / (n[:, None] + ALPHA_OUTLET)
        lo = np.log(sm / (1 - sm))
        return lo[R].mean(0) - lo[L].mean(0)
    else:
        raise ValueError(name)
    p_l, p_r, pi = wmean(s[L], w[L]), wmean(s[R], w[R]), wmean(s, w)
    return delta_from_shares(p_l, p_r, pi, n[L].sum(), n[R].sum())


SCHEMES = ["equal", "sqrt_n", "capped", "pooled", "trim10", "big_only", "mean_logodds"]
rng = np.random.default_rng(pp.SEED)
allpos = np.arange(len(outl))
strata = [np.where(side == s)[0] for s in ("left", "right", "centre")]
splits = []
for _ in range(N_SPLITS):
    A, B = [], []
    for ix in strata:
        perm = rng.permutation(ix)
        A.append(perm[: len(perm) // 2]); B.append(perm[len(perm) // 2:])
    splits.append((np.concatenate(A), np.concatenate(B)))

out(f"05c_weighting - {time.strftime('%Y-%m-%d %H:%M')} - {N_SPLITS} split-halves, alpha_0 {ALPHA0:,}")
out(f"outlets {len(outl)} | political articles per outlet: median {np.median(n_o):,.0f}, "
    f"min {n_o.min():,.0f}, max {n_o.max():,.0f} | at least {BIG:,}: {(n_o >= BIG).sum()}")

rows, deltas = [], {}
for name in SCHEMES:
    d = scheme_delta(name, allpos)
    deltas[name] = d
    ra, rs = [], []
    for A, B in splits:
        dA, dB = scheme_delta(name, A), scheme_delta(name, B)
        if np.isnan(dA).any() or np.isnan(dB).any():
            continue
        ra.append(spearmanr(np.abs(dA), np.abs(dB)).statistic)
        rs.append(spearmanr(dA, dB).statistic)
    sb = lambda r: float(np.median(2 * np.asarray(r) / (1 + np.asarray(r))))
    rows.append({"scheme": name, "SB_abs": sb(ra), "SB_signed": sb(rs),
                 "rho_abs": float(np.median(ra)), "rho_signed": float(np.median(rs)),
                 "sd_delta": float(np.std(d)), "top3": ", ".join(
                     f"{c}:{labels[c][:18]}" for c in np.argsort(-np.abs(d))[:3])})
rel = pd.DataFrame(rows).set_index("scheme")
out("\nSPLIT-HALF RELIABILITY BY WEIGHTING SCHEME (rule 3 threshold: SB_abs >= 0.70)")
out(rel.to_string(formatters={c: "{:.2f}".format for c in ["SB_abs", "SB_signed", "rho_abs", "rho_signed", "sd_delta"]}))

# how much of the between-outlet variation in topic attention is side, rather than outlet idiosyncrasy?
pi = S.mean(0)
sm = (M + ALPHA_OUTLET * pi) / (n_o[:, None] + ALPHA_OUTLET)
lo = np.log(sm / (1 - sm))
L, R = side == "left", side == "right"
r2 = []
for c in range(K):
    y = lo[L | R, c]
    g = side[L | R]
    grand = y.mean()
    between = sum(len(y[g == s]) * (y[g == s].mean() - grand) ** 2 for s in ("left", "right"))
    r2.append(between / ((y - grand) ** 2).sum())
r2 = np.array(r2)
out(f"\nSIDE AS A SHARE OF BETWEEN-OUTLET VARIANCE in log-odds attention: median {np.median(r2):.3f}, "
    f"max {r2.max():.3f} (cluster {int(np.argmax(r2))}), clusters above 0.10: {(r2 > 0.1).sum()} of {K}")
top = np.argsort(-r2)[:8]
out("   strongest: " + ", ".join(f"{c} {labels[c][:20]} {r2[c]:.2f}" for c in top))

df = pd.DataFrame(deltas, index=[f"{c} {labels[c]}" for c in range(K)])
df["side_R2"] = r2
df.to_csv(f"{P}/delta_by_scheme.csv", float_format="%.4f")
best = rel["SB_abs"].idxmax()
out(f"\nRESULT: most reliable scheme '{best}' SB_abs {rel.loc[best, 'SB_abs']:.2f} -> "
    f"{'meets the 0.70 threshold' if rel.loc[best, 'SB_abs'] >= 0.7 else 'below 0.70: the fallback measure applies'}")
out("cross-scheme agreement (Spearman on |delta|):")
out(pd.DataFrame({a: {b: spearmanr(np.abs(deltas[a]), np.abs(deltas[b])).statistic for b in SCHEMES}
                  for a in SCHEMES}).to_string(float_format="{:.2f}".format))
with open(f"{P}/05c_report.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
