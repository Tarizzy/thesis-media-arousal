"""08b2_nested_full.py - the final predictive numbers, run to the Day 8 specification.
Outer StratifiedGroupKFold(5) x 5 repeats, inner StratifiedGroupKFold(4); every split grouped by outlet.
  forest   tuned on max_features x min_samples_leaf (grid extended below 20, where 08b's optimum sat)
  boosting tuned on learning_rate x max_leaf_nodes (grid extended below 0.05 and 15); the number of boosting
           rounds is set by early stopping on an outlet-grouped validation split, then the model is refit
           on the whole outer training set with that many rounds
  ordinal  unpenalised proportional-odds logit (11 predictors on ~45k articles: a penalty would be ~0)
Feature sets: text only, bias only, text + bias. Scores articles and whole unseen outlets, with 2,000-draw
outlet-bootstrap CIs, paired comparisons, and breakdowns by bias extremity, bias side and country.
08b stays as the first pass; this file supersedes its numbers.
Reads pipeline/analysis_table.parquet. Writes ml/08b2_*.csv, ml/08b2_outlet_rows.parquet, ml/08b2_report.txt."""
import os, sys, time, itertools, warnings
sys.path.insert(0, os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "pipeline"))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
from statsmodels.miscmodels.ordinal_model import OrderedModel
import preprocessing as pp

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = f"{pp.ROOT}/ml"
os.makedirs(OUT, exist_ok=True)
QUICK = os.environ.get("QUICK") == "1"
REPEATS = 1 if QUICK else int(os.environ.get("REPEATS", 5))
OUTER, INNER, NC = 5, (3 if QUICK else 4), 5
BOOT = 200 if QUICK else 2000
N_TREES = 20 if QUICK else 300
MAX_ITER, PATIENCE = (60, 5) if QUICK else (2000, 30)
GRIDS = {
    "random forest": [{"max_features": mf, "min_samples_leaf": leaf}
                      for mf, leaf in itertools.product(("sqrt", 0.5), (5, 10, 20, 50, 100))],
    "boosting": [{"learning_rate": lr, "max_leaf_nodes": ml}
                 for lr, ml in itertools.product((0.02, 0.05, 0.1), (7, 15, 31))],
}
if QUICK:
    GRIDS = {m: g[:2] for m, g in GRIDS.items()}
MODELS = ["ordinal logit", "random forest", "boosting"]
FACT = {"Very Low": 0, "Low": 1, "Mixed": 2, "Mostly Factual": 3, "High": 4, "Very High": 4}
LEVELS = ["Very Low", "Low", "Mixed", "Mostly Factual", "High"]
EXTREMITY = {"Least Biased": 0, "Left-Center": 1, "Right-Center": 1, "Left": 2, "Right": 2,
             "Extreme Left": 3, "Extreme Right": 3}
EXT_NAME = {0: "0 least biased", 1: "1 left/right-center", 2: "2 left/right", 3: "3 extreme left/right"}
SIDE = {"Extreme Left": "left", "Left": "left", "Left-Center": "left", "Least Biased": "centre",
        "Right-Center": "right", "Right": "right", "Extreme Right": "right"}
BIAS = ["bias_ext", "side_left", "side_right"]
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


# ---------- metrics from confusion counts (vectorised; checked against scikit-learn below) ----------
W2 = (np.subtract.outer(np.arange(NC), np.arange(NC)) ** 2) / (NC - 1) ** 2


def metrics(C):
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
    pa = P.argmax(1)
    art = np.zeros((n_out, NC * NC))
    np.add.at(art, (codes, y * NC + pa), 1)
    mean_p = np.zeros((n_out, NC))
    np.add.at(mean_p, codes, P)
    mean_p /= np.bincount(codes, minlength=n_out)[:, None]
    oy = np.zeros(n_out, int)
    oy[codes] = y
    outl = np.zeros((n_out, NC * NC))
    outl[np.arange(n_out), oy * NC + mean_p.argmax(1)] = 1
    return art, outl


# ---------- models ----------
def grouped_split(y, g, seed, k=5):
    """One outlet-grouped (train, validation) split of a training set, for early stopping."""
    return next(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g))


def place(P_small, classes):
    P = np.zeros((len(P_small), NC))
    P[:, classes] = P_small
    return P


def fit_proba(model, params, Xtr, ytr, gtr, Xte, seed, refit=True):
    """Class probabilities for Xte (columns 0-4). Returns (P, n_iter or None)."""
    if model == "ordinal logit":
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        res = OrderedModel(ytr, (Xtr - mu) / sd, distr="logit").fit(method="bfgs", maxiter=500, disp=False)
        return place(np.asarray(res.predict((Xte - mu) / sd)), np.unique(ytr)), None
    if model == "random forest":
        m = RandomForestClassifier(n_estimators=N_TREES, class_weight="balanced", n_jobs=-1, random_state=seed,
                                   **params).fit(Xtr, ytr)
        return place(m.predict_proba(Xte), m.classes_), None
    fi, vi = grouped_split(ytr, gtr, seed)
    base = dict(min_samples_leaf=50, l2_regularization=1.0, class_weight="balanced", random_state=seed, **params)
    es = HistGradientBoostingClassifier(max_iter=MAX_ITER, early_stopping=True, n_iter_no_change=PATIENCE,
                                        **base).fit(Xtr[fi], ytr[fi], X_val=Xtr[vi], y_val=ytr[vi])
    if not refit:
        return place(es.predict_proba(Xte), es.classes_), es.n_iter_
    m = HistGradientBoostingClassifier(max_iter=es.n_iter_, early_stopping=False, **base).fit(Xtr, ytr)
    return place(m.predict_proba(Xte), m.classes_), es.n_iter_


def tune(model, X, y, g, seed):
    if model == "ordinal logit":
        return {}
    inner = list(StratifiedGroupKFold(INNER, shuffle=True, random_state=seed + 100).split(X, y, g))
    scores = []
    for params in GRIDS[model]:
        s = [cohen_kappa_score(y[te], fit_proba(model, params, X[tr], y[tr], g[tr], X[te], seed,
                                                refit=False)[0].argmax(1), weights="quadratic")
             for tr, te in inner]
        scores.append(np.mean(s))
    return GRIDS[model][int(np.argmax(scores))]


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
if df[["y", "bias_ext"]].isna().any().any():
    raise SystemExit("factuality or bias not codable for some rows")
lo, hi = df.flesch_kincaid.quantile([0.01, 0.99])
df["flesch_kincaid"] = df.flesch_kincaid.clip(lo, hi)
codes, sources = pd.factorize(df.source)
n_out, y, g = len(sources), df.y.to_numpy(), df.source.to_numpy()
FEATURE_SETS = {"text only": pp.FEATURES, "bias only": BIAS, "text + bias": pp.FEATURES + BIAS}
out(f"08b2_nested_full - {time.strftime('%Y-%m-%d %H:%M')} - {len(df):,} articles, {n_out} outlets"
    f"{' | QUICK SMOKE TEST' if QUICK else ''}")
out(f"outer {OUTER}-fold x {REPEATS} repeats, inner {INNER}-fold, all grouped by outlet | {BOOT:,} bootstrap draws")
for m, gr in GRIDS.items():
    out(f"grid {m} ({len(gr)} settings): {gr}")
out(f"boosting: up to {MAX_ITER} rounds, early stopping after {PATIENCE} rounds without improvement on an "
    f"outlet-grouped validation split")

STORE, RUNS, PARAMS = {}, [], []
for rep in range(REPEATS):
    folds = list(StratifiedGroupKFold(OUTER, shuffle=True, random_state=pp.SEED + rep).split(df, y, codes))
    for fname, feats in FEATURE_SETS.items():
        X = df[feats].to_numpy(float)
        for model in MODELS:
            t = time.time()
            P = np.zeros((len(df), NC))
            for k, (tr, te) in enumerate(folds):
                seed = pp.SEED + 10 * rep + k
                best = tune(model, X[tr], y[tr], g[tr], seed)
                P[te], n_iter = fit_proba(model, best, X[tr], y[tr], g[tr], X[te], seed)
                PARAMS.append({"repeat": rep, "fold": k, "features": fname, "model": model, **best,
                               "n_iter": n_iter})
            art, outl = outlet_rows(P, y, codes, n_out)
            STORE.setdefault((fname, model), []).append((art, outl))
            ma, mo = metrics(art.sum(0)), metrics(outl.sum(0))
            if rep == 0:
                pa = P.argmax(1)
                assert abs(ma["QWK"] - cohen_kappa_score(y, pa, weights="quadratic")) < 1e-9
                assert abs(ma["macro F1"] - f1_score(y, pa, average="macro")) < 1e-9
                assert abs(ma["accuracy"] - accuracy_score(y, pa)) < 1e-9
            RUNS.append({"repeat": rep, "features": fname, "model": model,
                         **{f"article {k2}": float(v) for k2, v in ma.items()},
                         **{f"outlet {k2}": float(v) for k2, v in mo.items()}, "seconds": time.time() - t})
            out(f"   rep {rep}  {fname:<12}{model:<15}article QWK {ma['QWK']:.3f}  outlet QWK {mo['QWK']:.3f}"
                f"  ({time.time() - t:.0f}s)")
            pd.DataFrame(RUNS).to_csv(f"{OUT}/08b2_runs.csv", index=False)
            pd.DataFrame(PARAMS).to_csv(f"{OUT}/08b2_params.csv", index=False)

# ---------- estimates with outlet-bootstrap CIs ----------
rng = np.random.default_rng(pp.SEED)
Wb = rng.multinomial(n_out, np.full(n_out, 1 / n_out), size=BOOT)


def estimate(key, level, metric):
    idx = 0 if level == "article" else 1
    runs = STORE[key]
    return (np.mean([metrics(r[idx].sum(0))[metric] for r in runs]),
            np.mean([metrics(Wb @ r[idx])[metric] for r in runs], axis=0))


def ci(d):
    lo_, hi_ = np.nanpercentile(d, [2.5, 97.5])
    return f"[{lo_:+.3f}, {hi_:+.3f}]"


EST = []
out("\n" + "=" * 100 + "\nFINAL PERFORMANCE ON UNSEEN OUTLETS (mean over repeats, 95% outlet-bootstrap CI)")
n_high_a, n_high_o = int((y == 4).sum()), int((df.groupby("source").y.first() == 4).sum())
out(f"   majority-class baseline (always 'High'): QWK 0 | accuracy {n_high_a / len(y):.3f} articles, "
    f"{n_high_o / n_out:.3f} outlets | macro F1 {2 * (n_high_a / len(y)) / (1 + n_high_a / len(y)) / NC:.3f}, "
    f"{2 * (n_high_o / n_out) / (1 + n_high_o / n_out) / NC:.3f}")
for level in ("article", "outlet"):
    out(f"\n   {level.upper()} LEVEL")
    out(f"   {'features':<12}{'model':<15}{'QWK':>7} {'CI':<18}{'accuracy':>9} {'CI':<18}{'macro F1':>9} CI")
    for fname in FEATURE_SETS:
        for model in MODELS:
            cells = []
            for metric in ("QWK", "accuracy", "macro F1"):
                p, d = estimate((fname, model), level, metric)
                EST.append({"comparison": f"{fname} | {model}", "level": level, "metric": metric, "estimate": p,
                            "ci_lo": np.nanpercentile(d, 2.5), "ci_hi": np.nanpercentile(d, 97.5)})
                cells.append(f"{p:7.3f} {ci(d):<18}")
            out(f"   {fname:<12}{model:<15}" + "  ".join(cells))

out("\n   PAIRED DIFFERENCES (outlet level)")
for model in MODELS:
    for metric in ("QWK", "accuracy", "macro F1"):
        pa_, da = estimate(("text + bias", model), "outlet", metric)
        pb_, db = estimate(("bias only", model), "outlet", metric)
        d = da - db
        EST.append({"comparison": f"text+bias - bias | {model}", "level": "outlet", "metric": metric,
                    "estimate": pa_ - pb_, "ci_lo": np.nanpercentile(d, 2.5), "ci_hi": np.nanpercentile(d, 97.5)})
        out(f"   text+bias minus bias   {model:<15}{metric:<9}{pa_ - pb_:+.3f} {ci(d)}  P(>0) {(d > 0).mean():.2f}")
for model in ("random forest", "boosting"):
    pa_, da = estimate(("text only", model), "outlet", "QWK")
    pb_, db = estimate(("text only", "ordinal logit"), "outlet", "QWK")
    d = da - db
    EST.append({"comparison": f"text: {model} - ordinal logit", "level": "outlet", "metric": "QWK",
                "estimate": pa_ - pb_, "ci_lo": np.nanpercentile(d, 2.5), "ci_hi": np.nanpercentile(d, 97.5)})
    out(f"   text only: {model} minus ordinal logit   QWK {pa_ - pb_:+.3f} {ci(d)}  P(>0) {(d > 0).mean():.2f}")
pd.DataFrame(EST).to_csv(f"{OUT}/08b2_estimates.csv", index=False)

# ---------- breakdowns: where do the text models get outlets right or wrong? ----------
o_info = df.groupby("source").agg(y=("y", "first"), ext=("bias_ext", "first"),
                                  side=("mbfc_bias", lambda s: SIDE[s.iloc[0]]),
                                  country=("country", "first")).loc[sources]
o_info["country"] = o_info.country.fillna("unknown").map(
    lambda c: c if c in ("USA", "United Kingdom", "unknown") else "other")
o_info["ext"] = o_info.ext.map(EXT_NAME)
BD = []
out("\n" + "=" * 100 + "\nBREAKDOWN BY OUTLET GROUP (outlet level, mean over repeats; 95% CI from resampling outlets")
out("within the group). error = predicted minus true class, in factuality steps: < 0 means the model rates the")
out("group's outlets LESS factual than MBFC does, > 0 MORE factual.")
for fname, model in [("text only", "boosting"), ("text only", "random forest"), ("text + bias", "boosting")]:
    preds = np.array([r[1].argmax(1) % NC for r in STORE[(fname, model)]])      # repeats x outlets
    err = preds - o_info.y.to_numpy()[None, :]
    out(f"\n   {fname} | {model}")
    out(f"   {'group':<28}{'outlets':>8}{'accuracy':>10}{'mean |error|':>14}{'mean error':>12}  {'95% CI of mean error'}")
    for gcol in ("ext", "side", "country"):
        for gval, idx in o_info.groupby(gcol).indices.items():
            e = err[:, idx]
            acc, mae, bias_ = (e == 0).mean(), np.abs(e).mean(), e.mean()
            bs = np.random.default_rng(pp.SEED).integers(0, len(idx), size=(BOOT, len(idx)))
            dist = e.mean(0)[bs].mean(1)
            BD.append({"model": f"{fname} | {model}", "grouping": gcol, "group": gval, "outlets": len(idx),
                       "accuracy": acc, "mean_abs_error": mae, "mean_error": bias_,
                       "err_ci_lo": np.percentile(dist, 2.5), "err_ci_hi": np.percentile(dist, 97.5),
                       "true_classes": ", ".join(LEVELS[c] for c in sorted(set(o_info.y.to_numpy()[idx])))})
            out(f"   {gcol + ': ' + str(gval):<28}{len(idx):>8}{acc:>10.2f}{mae:>14.2f}{bias_:>+12.2f}  "
                f"[{np.percentile(dist, 2.5):+.2f}, {np.percentile(dist, 97.5):+.2f}]")
pd.DataFrame(BD).to_csv(f"{OUT}/08b2_breakdown.csv", index=False)

# ---------- save per-outlet rows and chosen parameters ----------
rows = []
for (fname, model), runs in STORE.items():
    for rep, (art, outl) in enumerate(runs):
        r = pd.DataFrame(np.column_stack([art, outl]), columns=[f"a{i}{j}" for i in range(NC) for j in range(NC)]
                         + [f"o{i}{j}" for i in range(NC) for j in range(NC)])
        r.insert(0, "source", sources)
        r.insert(0, "repeat", rep)
        r.insert(0, "model", model)
        r.insert(0, "features", fname)
        rows.append(r)
pd.concat(rows).to_parquet(f"{OUT}/08b2_outlet_rows.parquet", index=False)
pl = pd.DataFrame(PARAMS)


def canon(v):
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


out("\nCHOSEN SETTINGS (how often each value won an outer fold, over all feature sets and repeats)")
for m in ("random forest", "boosting"):
    sub = pl[pl.model == m]
    keys = [k for k in GRIDS[m][0]]
    out(f"   {m}: " + " | ".join(f"{k}: " + ", ".join(f"{v}: {n}" for v, n in sub[k].map(canon).value_counts().items())
                                 for k in keys))
    edge = [k for k in keys if sub[k].map(canon).value_counts().index[0] in
            (canon(GRIDS[m][0][k]), canon(GRIDS[m][-1][k]))]
    out(f"      most frequent value on the edge of the grid for: {edge if edge else 'none'}")
bo = pl[(pl.model == "boosting") & pl.n_iter.notna()]
out(f"   boosting rounds chosen by early stopping: median {bo.n_iter.median():.0f}, range {bo.n_iter.min():.0f}-"
    f"{bo.n_iter.max():.0f}; hit the {MAX_ITER}-round cap in {(bo.n_iter >= MAX_ITER).sum()} of {len(bo)} fits")
out(f"\nRESULT: wrote 08b2_runs.csv, 08b2_estimates.csv, 08b2_breakdown.csv, 08b2_params.csv, "
    f"08b2_outlet_rows.parquet in {(time.time() - t0) / 60:.0f} min")
with open(f"{OUT}/08b2_report.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
