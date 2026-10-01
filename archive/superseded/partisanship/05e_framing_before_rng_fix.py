"""05e_framing.py - framing divergence per topic cluster: how differently left and right outlets write about
the same topic. Monroe, Colaresi & Quinn (2008) log-odds with an informative Dirichlet prior, applied to
words inside each cluster, outlets weighted equally, with a permutation baseline so clusters of different
size are comparable. Split-half reliability, outlet bootstrap and variants follow 05_expectations.md.
Reads partisanship/word_counts.parquet, vocab.parquet, outlet_cluster_counts.parquet,
      pipeline/analysis_table.parquet, clustering/cluster_labels.csv
Writes partisanship/cluster_framing.csv, framing_model.csv, 05e_report.txt, fig_framing.png"""
import os, sys, time
sys.path.insert(0, os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "pipeline"))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import preprocessing as pp

P = f"{pp.ROOT}/partisanship"
K, FOLD = 33, [14, 26]
ALPHA0 = 1_000          # prior strength, in tokens
MIN_TOK = int(os.environ.get('MIN_TOK', 2000))   # tokens an outlet needs in a cluster to contribute
MIN_WORD = 20           # a word needs this many tokens in the cluster to count
N_PERM, N_SPLITS, N_BOOT = 40, 100, 200
CHUNK = 100
SIDE = {"Extreme Left": "left", "Left": "left", "Left-Center": "left", "Least Biased": "centre",
        "Right-Center": "right", "Right": "right", "Extreme Right": "right"}
MIXED_PLUS = {"Mixed", "Mostly Factual", "High", "Very High"}
CONTESTED = [24, 25, 29, 28, 0]
NONPOL = [2, 3, 4, 10, 13, 14, 26, 32]
rng = np.random.default_rng(pp.SEED)
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


t0 = time.time()
wc = pd.read_parquet(f"{P}/word_counts.parquet")
tab = pd.read_parquet(f"{pp.ROOT}/pipeline/analysis_table.parquet",
                      columns=["source", "mbfc_bias", "mbfc_factuality", "country"])
outl = tab.groupby("source").first()
outl["side"] = outl["mbfc_bias"].map(SIDE)
labels = pd.read_csv(f"{pp.ROOT}/clustering/cluster_labels.csv").set_index("cluster")
counts = pd.read_parquet(f"{P}/outlet_cluster_counts.parquet")
pol_share = (counts.groupby("cluster")[["n_all", "n_political"]].sum()
             .assign(share=lambda d: d.n_political / d.n_all))

sources = list(outl.index)
s_idx = {s: i for i, s in enumerate(sources)}
words = sorted(wc.word.unique())
w_idx = {w: i for i, w in enumerate(words)}
side = outl["side"].to_numpy()
is_us = (outl["country"] == "USA").fillna(False).to_numpy(bool)
mixed_plus = outl["mbfc_factuality"].isin(MIXED_PLUS).to_numpy()
vad = pp.load_vad()
emotion = np.array([w in vad for w in words])
out(f"05e_framing - {time.strftime('%Y-%m-%d %H:%M')} - {len(sources)} outlets, {len(words):,} words, "
    f"{int(emotion.sum()):,} of them in the NRC-VAD lexicon")

wc["si"] = wc.source.map(s_idx)
wc["wi"] = wc.word.map(w_idx)
cluster_mats = {}
for c, g in wc.groupby("cluster"):
    X = np.zeros((len(sources), len(words)), dtype=np.float32)
    X[g.si.to_numpy(), g.wi.to_numpy()] = g.n.to_numpy()
    cluster_mats[int(c)] = X


def prep(X, rows):
    sub = X[rows]
    tot = sub.sum(1)
    keep = sub.sum(0) >= MIN_WORD
    Xn = (sub[:, keep] / np.maximum(tot, 1)[:, None]).astype(np.float64)
    pi_bg = Xn.mean(0)                       # background: fixed for the cluster, always positive
    pi_bg = pi_bg / pi_bg.sum()
    return Xn, tot, keep, pi_bg


def scores(prepped, reps_l, reps_r, alpha0=ALPHA0):
    """Framing divergence for each replicate: rows of reps_* are outlet weights over the prepped outlets."""
    Xn, tot, _, pi = prepped
    al = alpha0 * pi
    out_scores = np.empty(len(reps_l))
    for a in range(0, len(reps_l), CHUNK):
        wl, wr = reps_l[a:a + CHUNK], reps_r[a:a + CHUNK]
        norm = lambda W: W / W.sum(1, keepdims=True)
        p_l, p_r = norm(wl) @ Xn, norm(wr) @ Xn
        n_l, n_r = (wl * tot).sum(1, keepdims=True), (wr * tot).sum(1, keepdims=True)
        y_l, y_r = p_l * n_l, p_r * n_r
        d = (np.log((y_r + al) / (n_r + alpha0 - y_r - al))
             - np.log((y_l + al) / (n_l + alpha0 - y_l - al)))
        out_scores[a:a + CHUNK] = (pi * np.abs(d)).sum(1)
    return out_scores


def one_score(prepped, left, right, n_perm=N_PERM, alpha0=ALPHA0):
    """Observed divergence minus the mean divergence under shuffled side labels (same group sizes)."""
    m = len(prepped[1])
    L = np.zeros(m); L[left] = 1
    R = np.zeros(m); R[right] = 1
    reps_l, reps_r = [L], [R]
    lab = np.concatenate([np.ones(int(L.sum())), np.zeros(int(R.sum()))])
    pos = np.where(L + R > 0)[0]
    for _ in range(n_perm):
        pl = rng.permutation(lab)
        a = np.zeros(m); b = np.zeros(m)
        a[pos[pl == 1]] = 1; b[pos[pl == 0]] = 1
        reps_l.append(a); reps_r.append(b)
    s = scores(prepped, np.array(reps_l), np.array(reps_r), alpha0)
    return s[0] - s[1:].mean(), s[0]


def eligible(c, mask=None):
    X = cluster_mats[c]
    tot = X.sum(1)
    ok = tot >= MIN_TOK
    if mask is not None:
        ok &= mask
    rows = np.where(ok)[0]
    sd = side[rows]
    return X, rows, np.where(sd == "left")[0], np.where(sd == "right")[0]


res, tops = [], {}
for c in range(K):
    X, rows, li, ri = eligible(c)
    if len(li) < 5 or len(ri) < 5:
        res.append({"cluster": c, "framing": np.nan, "raw": np.nan, "outlets": len(rows),
                    "left": len(li), "right": len(ri)})
        continue
    pre = prep(X, rows)
    adj, raw = one_score(pre, li, ri)
    # bootstrap and split-half over outlets
    boots = []
    for _ in range(N_BOOT):
        bl = rng.choice(li, len(li), replace=True)
        br = rng.choice(ri, len(ri), replace=True)
        L = np.bincount(bl, minlength=len(rows)).astype(float)
        R = np.bincount(br, minlength=len(rows)).astype(float)
        boots.append(scores(pre, L[None, :], R[None, :])[0])
    halves = []
    for _ in range(N_SPLITS):
        pl, pr = rng.permutation(li), rng.permutation(ri)
        a = one_score(pre, pl[: len(pl) // 2], pr[: len(pr) // 2], n_perm=20)[0]
        b = one_score(pre, pl[len(pl) // 2:], pr[len(pr) // 2:], n_perm=20)[0]
        halves.append((a, b))
    res.append({"cluster": c, "framing": adj, "raw": raw, "outlets": len(rows), "left": len(li), "right": len(ri),
                "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
                "half_a": float(np.mean([h[0] for h in halves])), "half_b": float(np.mean([h[1] for h in halves])),
                "halves": halves})
    # most distinctive words, for face validity
    Xn, tot, keep, pi = pre
    p_l, p_r = Xn[li].mean(0), Xn[ri].mean(0)
    n_l, n_r = tot[li].sum(), tot[ri].sum()
    al = ALPHA0 * pi
    y_l, y_r = p_l * n_l, p_r * n_r
    d = (np.log((y_r + al) / (n_r + ALPHA0 - y_r - al)) - np.log((y_l + al) / (n_l + ALPHA0 - y_l - al)))
    z = d / np.sqrt(1 / (y_r + al) + 1 / (y_l + al))
    kw = np.array(words)[keep]
    tops[c] = (", ".join(kw[np.argsort(-z)[:8]]), ", ".join(kw[np.argsort(z)[:8]]))

R_ = pd.DataFrame(res).set_index("cluster")
R_["label"] = labels["label"]
R_["political"] = labels["political"]
R_["political_share"] = pol_share["share"]
usable = R_["framing"].notna()
out(f"clusters scored: {int(usable.sum())} of {K} (an outlet needs {MIN_TOK:,} tokens in a cluster; "
    f"a cluster needs 5 outlets a side)")

if not usable.any():
    out("no cluster had enough outlets on both sides - lower MIN_TOK or raise the cap in 05d")
    with open(f"{P}/05e_report.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    raise SystemExit(1)

out("\nCLUSTERS BY FRAMING DIVERGENCE (permutation-adjusted; CI = 95% outlet bootstrap of the raw score)")
show = R_[usable].sort_values("framing", ascending=False)
out(pd.DataFrame({"label": show.label.str.slice(0, 30),
                  "framing": show.framing.map("{:.4f}".format),
                  "raw": show.raw.map("{:.4f}".format),
                  "95% CI": [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(show.ci_lo, show.ci_hi)],
                  "outlets": show.outlets, "L/R": [f"{l}/{r}" for l, r in zip(show.left, show.right)],
                  "pol_share": show.political_share.map("{:.2f}".format),
                  "np": show.political.map(lambda p: "" if p else "non-pol")}).to_string())

out("\nMOST DISTINCTIVE WORDS (top 5 clusters by divergence: right-leaning words, then left-leaning)")
for c in show.index[:5]:
    out(f"   {c} {R_.loc[c, 'label']}")
    out(f"      right: {tops[c][0]}")
    out(f"      left:  {tops[c][1]}")

allh = [h for c in R_.index[usable] for h in R_.loc[c, "halves"]]
A = np.array([[h[0] for h in R_.loc[c, "halves"]] for c in R_.index[usable]])
B = np.array([[h[1] for h in R_.loc[c, "halves"]] for c in R_.index[usable]])
rhos = [spearmanr(A[:, i], B[:, i]).statistic for i in range(A.shape[1])]
pears = [float(np.corrcoef(A[:, i], B[:, i])[0, 1]) for i in range(A.shape[1])]
sb_rank = float(np.median(2 * np.asarray(rhos) / (1 + np.asarray(rhos))))
sb_val = float(np.median(2 * np.asarray(pears) / (1 + np.asarray(pears))))
rule3 = sb_val >= 0.7
out(f"\nPRE-REGISTERED CHECKS")
out(f"Rule 3 split-half, values (the model uses the value): r median {np.median(pears):.2f} "
    f"[{np.percentile(pears, 2.5):.2f}, {np.percentile(pears, 97.5):.2f}], Spearman-Brown {sb_val:.2f} "
    f"-> {'PASS' if rule3 else 'FAIL'}")
out(f"           ranks: rho median {np.median(rhos):.2f}, Spearman-Brown {sb_rank:.2f} "
    f"(harsh when most clusters sit near zero)")
rank = R_.loc[usable, "framing"].rank(ascending=False)
n_top = int(sum(rank.get(c, 99) <= len(rank) / 2 for c in CONTESTED))
m_con = R_.loc[[c for c in CONTESTED if usable[c]], "framing"].mean()
m_np = R_.loc[[c for c in NONPOL if usable.get(c, False)], "framing"].mean()
out(f"Rule 2 sanity: {n_top} of 5 contested clusters in the top half (need 3); mean contested {m_con:.4f} "
    f"vs non-political {m_np:.4f} -> {'PASS' if (n_top >= 3 and m_con > m_np) else 'FAIL'}")

# variants
var = {}
for name, mask in [("us_only", is_us), ("mixed_plus", mixed_plus), ("no_emotion_words", None)]:
    vals = {}
    for c in R_.index[usable]:
        if name == "no_emotion_words":
            X = cluster_mats[c].copy(); X[:, emotion] = 0
            _, rows, li, ri = eligible(c)
        else:
            X, rows, li, ri = eligible(c, mask)
            if len(li) < 5 or len(ri) < 5:
                continue
        vals[c] = one_score(prep(X, rows), li, ri, n_perm=20)[0]
    var[name] = pd.Series(vals)
    R_[f"framing_{name}"] = var[name]
out("\nVARIANTS vs PRIMARY (Spearman rho over the clusters both score)")
for name, v in var.items():
    both = v.index.intersection(R_.index[usable])
    out(f"   {name:<18}{spearmanr(R_.loc[both, 'framing'], v[both]).statistic:.3f}  ({len(both)} clusters)")

R_.drop(columns=["halves"]).to_csv(f"{P}/cluster_framing.csv", float_format="%.5f")
keep = [c for c in range(K) if c not in FOLD]
model = pd.DataFrame({"topic_cluster_model": keep, "framing": R_.loc[keep, "framing"].to_numpy()})
Xo = sum(cluster_mats[c] for c in FOLD)
_, rows, li, ri = eligible(FOLD[0])
tot = Xo.sum(1)
rows = np.where(tot >= MIN_TOK)[0]
li, ri = np.where(side[rows] == "left")[0], np.where(side[rows] == "right")[0]
other = one_score(prep(Xo, rows), li, ri)[0] if (len(li) >= 5 and len(ri) >= 5) else np.nan
model = pd.concat([model, pd.DataFrame({"topic_cluster_model": [-1], "framing": [other]})])
model["framing_z"] = (model.framing - model.framing.mean()) / model.framing.std()
model.to_csv(f"{P}/framing_model.csv", index=False, float_format="%.5f")
out(f"Other (14 + 26 merged): {other:.4f}")

fig, ax = plt.subplots(figsize=(8, 10))
f = R_[usable].sort_values("framing")
y = np.arange(len(f))
ax.scatter(f.framing[f.political], y[f.political.to_numpy()], color="black", s=18, label="political cluster")
ax.scatter(f.framing[~f.political], y[~f.political.to_numpy()], facecolors="white", edgecolors="black", s=18,
           label="non-political cluster")
ax.set_yticks(y, [f"{c}  {l}" for c, l in zip(f.index, f.label)], fontsize=8)
ax.set_xlabel("framing divergence (permutation-adjusted)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{P}/fig_framing.png", dpi=200); plt.close(fig)

out(f"\nRESULT: {'reliable enough to use' if rule3 else 'FAILS rule 3 - fall back to hand-coded divisiveness'}"
    f" - total {time.time() - t0:.0f}s")
with open(f"{P}/05e_report.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
