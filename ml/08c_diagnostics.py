"""08c_diagnostics.py - the questions 08a/08b raised, answered with outlet-level uncertainty.
  A  feature ceilings: how much of each feature's variance sits between outlets, and how it tracks factuality
  B  leakage at the outlet level: naive k-fold, RF out-of-bag and outlet-grouped, scored per article and per outlet
  C  feature-group ablation: emotion / epistemic / style alone and each group removed (grouped CV)
  D  does text add anything once MBFC bias is known, on QWK, accuracy and macro F1
  E  same-rater check on the Ad Fontes subset: MBFC bias vs Ad Fontes lean as the benchmark, same outlets
Every comparison uses the same folds for both sides (paired). Confidence intervals come from resampling outlets
(2,000 draws) and average over the repeats, so they describe uncertainty about new outlets, not new articles.
Hyperparameters are fixed at the values 08b's nested search chose most often.
Reads pipeline/analysis_table.parquet. Writes ml/08c_*.csv, ml/08c_outlet_rows.parquet, two figures, 08c_report.txt."""
import os, sys, time, warnings
sys.path.insert(0, os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "pipeline"))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
from statsmodels.miscmodels.ordinal_model import OrderedModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import preprocessing as pp

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = f"{pp.ROOT}/ml"
os.makedirs(OUT, exist_ok=True)
QUICK = os.environ.get("QUICK") == "1"
REPEATS = 1 if QUICK else int(os.environ.get("REPEATS", 5))
BOOT = 200 if QUICK else 2000
FOLDS, NC = 5, 5
N_TREES, N_ITER = (20, 20) if QUICK else (200, 150)
RF_PARAMS = {"min_samples_leaf": 20}                           # 08b: chosen in 25 of 30 outer folds
HGB_PARAMS = {"learning_rate": 0.05, "max_leaf_nodes": 15}     # 08b: chosen in 29-30 of 30
MODELS = ["ordinal logit", "random forest", "boosting"]
FACT = {"Very Low": 0, "Low": 1, "Mixed": 2, "Mostly Factual": 3, "High": 4, "Very High": 4}
LEVELS = ["Very Low", "Low", "Mixed", "Mostly Factual", "High"]
EXTREMITY = {"Least Biased": 0, "Left-Center": 1, "Right-Center": 1, "Left": 2, "Right": 2,
             "Extreme Left": 3, "Extreme Right": 3}
SIDE = {"Extreme Left": "left", "Left": "left", "Left-Center": "left", "Least Biased": "centre",
        "Right-Center": "right", "Right": "right", "Extreme Right": "right"}
GROUPS = {"emotion": ["arousal", "valence", "dominance"],
          "epistemic": ["attribution_density", "hedging", "certainty"],
          "style": ["caps_ratio", "exclamation_rate", "flesch_kincaid", "mtld", "log_length"]}
if sorted(sum(GROUPS.values(), [])) != sorted(pp.FEATURES):
    raise SystemExit(f"feature groups do not match pp.FEATURES: {pp.FEATURES}")
BIAS = ["bias_ext", "side_left", "side_right"]
AF = ["af_lean", "af_abs"]
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


# ---------- metrics from confusion counts, vectorised so the bootstrap is cheap ----------
W2 = (np.subtract.outer(np.arange(NC), np.arange(NC)) ** 2) / (NC - 1) ** 2


def metrics(C):
    """C: (..., 25) confusion counts, rows = true class. Returns dict of arrays: QWK, accuracy, macro F1."""
    C = np.asarray(C, float).reshape(*np.shape(C)[:-1], NC, NC)
    n = C.sum((-1, -2))
    rows, cols = C.sum(-1), C.sum(-2)
    E = rows[..., :, None] * cols[..., None, :] / n[..., None, None]
    qwk = 1 - (W2 * C).sum((-1, -2)) / (W2 * E).sum((-1, -2))
    tp = np.diagonal(C, axis1=-2, axis2=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(cols > 0, tp / cols, 0.0)
        rec = np.where(rows > 0, tp / rows, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = (rows + cols) > 0
    return {"QWK": qwk, "accuracy": tp.sum(-1) / n, "macro F1": (f1 * present).sum(-1) / present.sum(-1)}


def outlet_rows(P, y, codes, n_out):
    """Per-outlet confusion rows: article-level counts, and the outlet's own (true, predicted) pair."""
    pa = P.argmax(1)
    art = np.zeros((n_out, NC * NC))
    np.add.at(art, (codes, y * NC + pa), 1)
    mean_p = np.zeros((n_out, NC))
    np.add.at(mean_p, codes, P)
    mean_p /= np.bincount(codes, minlength=n_out)[:, None]
    oy = np.zeros(n_out, int)
    oy[codes] = y
    op = mean_p.argmax(1)
    outl = np.zeros((n_out, NC * NC))
    outl[np.arange(n_out), oy * NC + op] = 1
    return art, outl, mean_p


def proba(model, Xtr, ytr, Xte, seed):
    """Class probabilities placed in columns 0-4, whichever classes the training fold happens to contain."""
    P = np.zeros((len(Xte), NC))
    if model == "ordinal logit":
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        res = OrderedModel(ytr, (Xtr - mu) / sd, distr="logit").fit(method="bfgs", maxiter=300, disp=False)
        P[:, np.unique(ytr)] = np.asarray(res.predict((Xte - mu) / sd))
        return P
    if model == "random forest":
        m = RandomForestClassifier(n_estimators=N_TREES, max_features="sqrt", class_weight="balanced",
                                   n_jobs=-1, random_state=seed, **RF_PARAMS)
    else:
        m = HistGradientBoostingClassifier(max_iter=N_ITER, early_stopping=False, min_samples_leaf=50,
                                           l2_regularization=1.0, class_weight="balanced",
                                           random_state=seed, **HGB_PARAMS)
    m.fit(Xtr, ytr)
    P[:, m.classes_] = m.predict_proba(Xte)
    return P


STORE = {}      # (block, feature set, model, scheme) -> list over repeats of (article rows, outlet rows)
RUNS = []


def record(block, fset, model, scheme, rep, P, y, codes, n_out, sources):
    art, outl, mean_p = outlet_rows(P, y, codes, n_out)
    STORE.setdefault((block, fset, model, scheme), []).append((art, outl))
    ma, mo = metrics(art.sum(0)), metrics(outl.sum(0))
    RUNS.append({"block": block, "features": fset, "model": model, "scheme": scheme, "repeat": rep,
                 **{f"article {k}": float(v) for k, v in ma.items()},
                 **{f"outlet {k}": float(v) for k, v in mo.items()}})
    if rep == 0:
        pa = P.argmax(1)
        assert abs(ma["QWK"] - cohen_kappa_score(y, pa, weights="quadratic")) < 1e-9
        assert abs(ma["macro F1"] - f1_score(y, pa, average="macro")) < 1e-9
        assert abs(ma["accuracy"] - accuracy_score(y, pa)) < 1e-9
    return ma, mo


def boot_weights(n_out, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_out, np.full(n_out, 1 / n_out), size=BOOT)


def estimate(key, level, Wb, metric="QWK"):
    """Point estimate and bootstrap draws, both averaged over repeats."""
    idx = 0 if level == "article" else 1
    runs = STORE[key]
    point = np.mean([metrics(r[idx].sum(0))[metric] for r in runs])
    draws = np.mean([metrics(Wb @ r[idx])[metric] for r in runs], axis=0)
    return point, draws


def ci(d):
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return f"[{lo:+.3f}, {hi:+.3f}]"


EST = []


def report_one(block, label, key, level, Wb, metric="QWK"):
    p, d = estimate(key, level, Wb, metric)
    EST.append({"block": block, "comparison": label, "level": level, "metric": metric, "estimate": p,
                "ci_lo": np.nanpercentile(d, 2.5), "ci_hi": np.nanpercentile(d, 97.5)})
    return p, d


def report_diff(block, label, key_a, key_b, level, Wb, metric="QWK"):
    pa, da = estimate(key_a, level, Wb, metric)
    pb, db = estimate(key_b, level, Wb, metric)
    d = da - db
    EST.append({"block": block, "comparison": label, "level": level, "metric": metric, "estimate": pa - pb,
                "ci_lo": np.nanpercentile(d, 2.5), "ci_hi": np.nanpercentile(d, 97.5),
                "share_draws_above_0": float((d > 0).mean())})
    return pa - pb, d


# ---------- data ----------
t0 = time.time()
df = pd.read_parquet(f"{pp.ROOT}/pipeline/analysis_table.parquet")
if QUICK:
    df = df.sample(frac=1, random_state=1).groupby("source").head(40)
df = df.reset_index(drop=True)
df["y"] = df.mbfc_factuality.map(FACT)
df["bias_ext"] = df.mbfc_bias.map(EXTREMITY)
side = df.mbfc_bias.map(SIDE)
df["side_left"], df["side_right"] = (side == "left").astype(float), (side == "right").astype(float)
lo, hi = df.flesch_kincaid.quantile([0.01, 0.99])
df["flesch_kincaid"] = df.flesch_kincaid.clip(lo, hi)
df["af_lean"] = pd.to_numeric(df.adfontes_lean, errors="coerce")
df["af_abs"] = df.af_lean.abs()
codes, sources = pd.factorize(df.source)
n_out = len(sources)
y = df.y.to_numpy()
out(f"08c_diagnostics - {time.strftime('%Y-%m-%d %H:%M')} - {len(df):,} articles, {n_out} outlets"
    f"{' | QUICK SMOKE TEST' if QUICK else ''}")
out(f"{FOLDS}-fold x {REPEATS} repeats | {BOOT:,} outlet-bootstrap draws | fixed parameters: "
    f"forest {RF_PARAMS}, boosting {HGB_PARAMS}")

# ---------- A: feature ceilings ----------
out("\n" + "=" * 100 + "\nA  FEATURE CEILINGS")


def icc1(x, c):
    k, N = c.max() + 1, len(x)
    n_i = np.bincount(c, minlength=k)
    m = np.bincount(c, x, minlength=k) / n_i
    msb = (n_i * (m - x.mean()) ** 2).sum() / (k - 1)
    msw = ((x - m[c]) ** 2).sum() / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


D = pd.get_dummies(df[["topic_cluster_model", "year"]].astype(str), drop_first=True).to_numpy(float)
D = np.column_stack([np.ones(len(df)), D])
fc = []
for f in pp.FEATURES:
    x = df[f].to_numpy(float)
    resid = x - D @ np.linalg.lstsq(D, x, rcond=None)[0]
    cls_means = df.groupby("y")[f].transform("mean").to_numpy()
    om = df.groupby("source").agg(m=(f, "mean"), y=("y", "first"))
    fc.append({"feature": f, "group": next(gname for gname, fs in GROUPS.items() if f in fs),
               "ICC outlet": icc1(x, codes), "ICC outlet, topic+year removed": icc1(resid, codes),
               "eta2 factuality (articles)": ((cls_means - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum(),
               "rho outlet mean vs factuality": spearmanr(om.m, om.y).statistic})
fc = pd.DataFrame(fc).set_index("feature")
fc.to_csv(f"{OUT}/08c_feature_ceiling.csv", float_format="%.4f")
out("ICC = share of a feature's variance that sits between outlets (the most an outlet-level signal can use);")
out("eta2 = share explained by the five factuality classes; rho = Spearman of outlet means with factuality")
out(fc.to_string(float_format=lambda v: f"{v:+.3f}"))

# ---------- B: leakage at article and outlet level ----------
out("\n" + "=" * 100 + "\nB  LEAKAGE: naive k-fold and out-of-bag vs outlet-grouped (all 11 text features)")
X = df[pp.FEATURES].to_numpy(float)
fold_sets = []
for rep in range(REPEATS):
    fold_sets.append({
        "grouped": list(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=pp.SEED + rep).split(X, y, codes)),
        "naive": list(StratifiedKFold(FOLDS, shuffle=True, random_state=pp.SEED + rep).split(X, y))})
for rep in range(REPEATS):
    for model in MODELS:
        for scheme in ("naive", "grouped"):
            t = time.time()
            P = np.zeros((len(df), NC))
            for tr, te in fold_sets[rep][scheme]:
                P[te] = proba(model, X[tr], y[tr], X[te], pp.SEED + rep)
            ma, mo = record("B", "text", model, scheme, rep, P, y, codes, n_out, sources)
            out(f"   rep {rep} {model:<15}{scheme:<9}article QWK {ma['QWK']:.3f}  outlet QWK {mo['QWK']:.3f}"
                f"  ({time.time() - t:.0f}s)")
    rf = RandomForestClassifier(n_estimators=N_TREES, max_features="sqrt", class_weight="balanced", oob_score=True,
                                n_jobs=-1, random_state=pp.SEED + rep, **RF_PARAMS).fit(X, y)
    P = np.nan_to_num(rf.oob_decision_function_)
    ma, mo = record("B", "text", "random forest", "OOB", rep, P, y, codes, n_out, sources)
    out(f"   rep {rep} {'random forest':<15}{'OOB':<9}article QWK {ma['QWK']:.3f}  outlet QWK {mo['QWK']:.3f}")

Wb = boot_weights(n_out, pp.SEED)
out("\n   QWK (mean over repeats) with 95% outlet-bootstrap CI")
out(f"   {'model':<15}{'scheme':<9}{'article':>9}  {'CI':<18}{'outlet':>8}  CI")
for model in MODELS:
    for scheme in ("naive", "OOB", "grouped"):
        key = ("B", "text", model, scheme)
        if key not in STORE:
            continue
        pa_, da = report_one("B", f"{model} {scheme}", key, "article", Wb)
        po_, do = report_one("B", f"{model} {scheme}", key, "outlet", Wb)
        out(f"   {model:<15}{scheme:<9}{pa_:9.3f}  {ci(da):<18}{po_:8.3f}  {ci(do)}")
out("\n   INFLATION: naive minus grouped (paired by outlet)")
for model in MODELS:
    for level in ("article", "outlet"):
        d, dd = report_diff("B", f"{model}: naive - grouped", ("B", "text", model, "naive"),
                            ("B", "text", model, "grouped"), level, Wb)
        g = estimate(("B", "text", model, "grouped"), level, Wb)[0]
        out(f"   {model:<15}{level:<8} {d:+.3f} {ci(dd)}  = {d / g:+.0%} of the grouped score")
d, dd = report_diff("B", "random forest: OOB - grouped", ("B", "text", "random forest", "OOB"),
                    ("B", "text", "random forest", "grouped"), "outlet", Wb)
out(f"   {'random forest':<15}{'OOB, outlet':<8} {d:+.3f} {ci(dd)}")

# ---------- C + D: feature sets under outlet-grouped CV ----------
out("\n" + "=" * 100 + "\nC  FEATURE-GROUP ABLATION and D  TEXT BEYOND BIAS (outlet-grouped, same folds throughout)")
SETS = {"emotion only": GROUPS["emotion"], "epistemic only": GROUPS["epistemic"], "style only": GROUPS["style"],
        "all text": pp.FEATURES,
        "text without emotion": [f for f in pp.FEATURES if f not in GROUPS["emotion"]],
        "text without epistemic": [f for f in pp.FEATURES if f not in GROUPS["epistemic"]],
        "text without style": [f for f in pp.FEATURES if f not in GROUPS["style"]],
        "bias only": BIAS, "text + bias": pp.FEATURES + BIAS}
for rep in range(REPEATS):
    for fset, feats in SETS.items():
        Xs = df[feats].to_numpy(float)
        for model in MODELS:
            if fset == "all text":        # identical to block B's grouped run: reuse it
                STORE.setdefault(("C", fset, model, "grouped"), []).append(STORE[("B", "text", model, "grouped")][rep])
                continue
            t = time.time()
            P = np.zeros((len(df), NC))
            for tr, te in fold_sets[rep]["grouped"]:
                P[te] = proba(model, Xs[tr], y[tr], Xs[te], pp.SEED + rep)
            ma, mo = record("C", fset, model, "grouped", rep, P, y, codes, n_out, sources)
            out(f"   rep {rep} {fset:<23}{model:<15}article QWK {ma['QWK']:.3f}  outlet QWK {mo['QWK']:.3f}"
                f"  ({time.time() - t:.0f}s)")

out("\n   OUTLET-LEVEL QWK by feature set, 95% CI (article-level QWK listed underneath)")
tab = []
for fset in SETS:
    row = {"features": fset}
    for model in MODELS:
        p, d = report_one("C", f"{fset} | {model}", ("C", fset, model, "grouped"), "outlet", Wb)
        pa_, _ = report_one("C", f"{fset} | {model}", ("C", fset, model, "grouped"), "article", Wb)
        row[model] = f"{p:.3f} {ci(d)}"
        row[f"{model} article"] = pa_
    tab.append(row)
tab = pd.DataFrame(tab).set_index("features")
out(tab[MODELS].to_string())
out("   article QWK: " + "; ".join(f"{i}: " + "/".join(f"{tab.loc[i, f'{m} article']:.3f}" for m in MODELS)
                                   for i in tab.index))

out("\n   UNIQUE CONTRIBUTION of each group: all text minus text without that group (outlet QWK, paired)")
for gname in GROUPS:
    for model in MODELS:
        d, dd = report_diff("C", f"unique {gname} | {model}", ("C", "all text", model, "grouped"),
                            ("C", f"text without {gname}", model, "grouped"), "outlet", Wb)
        out(f"   {gname:<10}{model:<15}{d:+.3f} {ci(dd)}  P(draw > 0) {(dd > 0).mean():.2f}")

out("\n   D  TEXT + BIAS minus BIAS ONLY (outlet level, paired)")
for model in MODELS:
    for metric in ("QWK", "accuracy", "macro F1"):
        d, dd = report_diff("D", f"text+bias - bias | {model}", ("C", "text + bias", model, "grouped"),
                            ("C", "bias only", model, "grouped"), "outlet", Wb, metric)
        out(f"   {model:<15}{metric:<10}{d:+.3f} {ci(dd)}  P(draw > 0) {(dd > 0).mean():.2f}")

# ---------- E: same-rater check on the Ad Fontes subset ----------
out("\n" + "=" * 100 + "\nE  SAME-RATER CHECK: MBFC bias vs Ad Fontes lean as the benchmark, on the same outlets")
sub = df[df.af_lean.notna()].reset_index(drop=True)
scodes, ssources = pd.factorize(sub.source)
sn = len(ssources)
sy = sub.y.to_numpy()
out(f"   {len(sub):,} articles, {sn} outlets | outlets by class: "
    + ", ".join(f"{LEVELS[k]} {v}" for k, v in sub.groupby('source').y.first().value_counts().sort_index().items()))
ASETS = {"MBFC bias": BIAS, "Ad Fontes lean": AF, "text only": pp.FEATURES, "text + Ad Fontes": pp.FEATURES + AF}
for rep in range(REPEATS):
    folds = list(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=pp.SEED + rep).split(sub, sy, scodes))
    for fset, feats in ASETS.items():
        Xs = sub[feats].to_numpy(float)
        for model in MODELS:
            P = np.zeros((len(sub), NC))
            for tr, te in folds:
                P[te] = proba(model, Xs[tr], sy[tr], Xs[te], pp.SEED + rep)
            ma, mo = record("E", fset, model, "grouped", rep, P, sy, scodes, sn, ssources)
            out(f"   rep {rep} {fset:<18}{model:<15}outlet QWK {mo['QWK']:.3f}")
Wa = boot_weights(sn, pp.SEED + 1)
out("\n   OUTLET-LEVEL QWK on the Ad Fontes subset, 95% CI")
for fset in ASETS:
    out(f"   {fset:<18}" + "  ".join(
        f"{m}: {report_one('E', f'{fset} | {m}', ('E', fset, m, 'grouped'), 'outlet', Wa)[0]:.3f} "
        f"{ci(estimate(('E', fset, m, 'grouped'), 'outlet', Wa)[1])}" for m in MODELS))
out("\n   paired differences (outlet QWK)")
for model in MODELS:
    d, dd = report_diff("E", f"MBFC - Ad Fontes | {model}", ("E", "MBFC bias", model, "grouped"),
                        ("E", "Ad Fontes lean", model, "grouped"), "outlet", Wa)
    d2, dd2 = report_diff("E", f"text+AF - AF | {model}", ("E", "text + Ad Fontes", model, "grouped"),
                          ("E", "Ad Fontes lean", model, "grouped"), "outlet", Wa)
    out(f"   {model:<15}MBFC minus Ad Fontes {d:+.3f} {ci(dd)}   text+AF minus AF {d2:+.3f} {ci(dd2)}")

# ---------- save ----------
pd.DataFrame(RUNS).to_csv(f"{OUT}/08c_runs.csv", index=False)
pd.DataFrame(EST).to_csv(f"{OUT}/08c_estimates.csv", index=False)
rows = []
for (block, fset, model, scheme), runs in STORE.items():
    names = sources if block != "E" else ssources
    for rep, (art, outl) in enumerate(runs):
        r = pd.DataFrame(np.column_stack([art, outl]),
                         columns=[f"a{i}{j}" for i in range(NC) for j in range(NC)]
                         + [f"o{i}{j}" for i in range(NC) for j in range(NC)])
        r.insert(0, "source", names)
        r.insert(0, "repeat", rep)
        for c, v in (("scheme", scheme), ("model", model), ("features", fset), ("block", block)):
            r.insert(0, c, v)
        rows.append(r)
pd.concat(rows).to_parquet(f"{OUT}/08c_outlet_rows.parquet", index=False)

# ---------- figures ----------
e = pd.DataFrame(EST)
fig, ax = plt.subplots(figsize=(8, 6))
order = list(SETS)
for j, model in enumerate(MODELS):
    sel = e[(e.block == "C") & (e.level == "outlet") & (e.metric == "QWK")
            & e.comparison.str.endswith(f"| {model}") & ~e.comparison.str.startswith("unique")]
    sel = sel.assign(fset=sel.comparison.str.split(" | ", regex=False).str[0]).set_index("fset").loc[order]
    ypos = np.arange(len(order)) + (j - 1) * 0.22
    ax.errorbar(sel.estimate, ypos, xerr=[sel.estimate - sel.ci_lo, sel.ci_hi - sel.estimate], fmt="o",
                ms=4, capsize=2, label=model, color=["0.6", "0.35", "0.0"][j])
ax.set_yticks(np.arange(len(order)), order)
ax.invert_yaxis()
ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("outlet-level QWK, unseen outlets (95% outlet-bootstrap CI)")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_08c_ablation.png", dpi=200); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, level in zip(axes, ("article", "outlet")):
    for j, model in enumerate(MODELS):
        for k, scheme in enumerate(("naive", "OOB", "grouped")):
            sel = e[(e.block == "B") & (e.level == level) & (e.comparison == f"{model} {scheme}")]
            if len(sel):
                r = sel.iloc[0]
                ax.errorbar(j + (k - 1) * 0.22, r.estimate, yerr=[[r.estimate - r.ci_lo], [r.ci_hi - r.estimate]],
                            fmt=["s", "^", "o"][k], ms=5, capsize=2, color=["0.65", "0.35", "0.0"][k],
                            label=scheme if j == 0 else None)
    ax.set_xticks(range(len(MODELS)), MODELS, fontsize=8)
    ax.set_title(f"{level} level", fontsize=10)
axes[0].set_ylabel("QWK (95% outlet-bootstrap CI)")
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_08c_leakage.png", dpi=200); plt.close(fig)

out(f"\nRESULT: wrote 08c_feature_ceiling.csv, 08c_runs.csv, 08c_estimates.csv, 08c_outlet_rows.parquet, "
    f"fig_08c_ablation.png, fig_08c_leakage.png in {time.time() - t0:.0f}s")
with open(f"{OUT}/08c_report.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
