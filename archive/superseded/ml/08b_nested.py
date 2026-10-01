"""08b_nested.py - honest performance: outlet-grouped nested CV with tuning inside each outer fold.
Three feature sets: text only (the thesis question), bias only (MBFC benchmark), text + bias (does text add
anything once bias is known?). Scores articles and, by averaging each held-out outlet's predicted
probabilities, whole unseen outlets.
Reads pipeline/analysis_table.parquet. Writes ml/08b_results.csv, ml/08b_params.csv, ml/08b_oof.parquet,
ml/08b_report.txt."""
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
FACT = {"Very Low": 0, "Low": 1, "Mixed": 2, "Mostly Factual": 3, "High": 4, "Very High": 4}
EXTREMITY = {"Least Biased": 0, "Left-Center": 1, "Right-Center": 1, "Left": 2, "Right": 2,
             "Extreme Left": 3, "Extreme Right": 3}
SIDE = {"Extreme Left": "left", "Left": "left", "Left-Center": "left", "Least Biased": "centre",
        "Right-Center": "right", "Right": "right", "Extreme Right": "right"}
QUICK = os.environ.get("QUICK") == "1"
REPEATS = 1 if QUICK else int(os.environ.get("REPEATS", 2))
OUTER, INNER = 5, 3
N_TREES, N_ITER = (20, 20) if QUICK else (200, 150)
GRIDS = {
    "random forest": [{"min_samples_leaf": v} for v in ((20, 100) if QUICK else (20, 50, 100))],
    "boosting": [{"learning_rate": lr, "max_leaf_nodes": ml}
                 for lr, ml in itertools.product((0.05, 0.1), (15, 31))][: 2 if QUICK else 4],
}
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


def qwk(y, p):
    return cohen_kappa_score(y, p, weights="quadratic")


def make(model, params):
    if model == "random forest":
        return RandomForestClassifier(n_estimators=N_TREES, max_features="sqrt", class_weight="balanced",
                                      n_jobs=-1, random_state=pp.SEED, **params)
    return HistGradientBoostingClassifier(max_iter=N_ITER, early_stopping=False, min_samples_leaf=50,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=pp.SEED, **params)


def fit_predict(model, params, Xtr, ytr, Xte):
    """Returns class probabilities (n_test x 5)."""
    if model == "ordinal logit":
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        res = OrderedModel(ytr, (Xtr - mu) / sd, distr="logit").fit(method="bfgs", maxiter=300, disp=False)
        return np.asarray(res.predict((Xte - mu) / sd))
    return make(model, params).fit(Xtr, ytr).predict_proba(Xte)


def tune(model, X, y, g, seed):
    """Inner outlet-grouped CV on the outer training set; returns the best parameter set."""
    if model == "ordinal logit":
        return {}
    inner = StratifiedGroupKFold(INNER, shuffle=True, random_state=seed + 100)
    scores = []
    for params in GRIDS[model]:
        s = [qwk(y[te], fit_predict(model, params, X[tr], y[tr], X[te]).argmax(1))
             for tr, te in inner.split(X, y, g)]
        scores.append(np.mean(s))
    return GRIDS[model][int(np.argmax(scores))]


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
y, g = df.y.to_numpy(), df.source.to_numpy()
BIAS = ["bias_ext", "side_left", "side_right"]
FEATURE_SETS = {"text only": pp.FEATURES, "bias only": BIAS, "text + bias": pp.FEATURES + BIAS}
MODELS = ["ordinal logit", "random forest", "boosting"]

out(f"08b_nested - {time.strftime('%Y-%m-%d %H:%M')} - {len(df):,} articles, {df.source.nunique()} outlets"
    f"{' | QUICK SMOKE TEST' if QUICK else ''}")
out(f"outer {OUTER}-fold x {REPEATS} repeats, inner {INNER}-fold, all splits grouped by outlet")
out("grids: " + "; ".join(f"{m}: {GRIDS[m]}" for m in GRIDS))

rows, params_log, oof = [], [], []
for rep in range(REPEATS):
    outer = StratifiedGroupKFold(OUTER, shuffle=True, random_state=pp.SEED + rep)
    folds = list(outer.split(df, y, g))
    for fname, feats in FEATURE_SETS.items():
        X = df[feats].to_numpy(float)
        for model in MODELS:
            t = time.time()
            P = np.zeros((len(df), 5))
            for k, (tr, te) in enumerate(folds):
                best = tune(model, X[tr], y[tr], g[tr], seed=pp.SEED + 10 * rep + k)
                P[te] = fit_predict(model, best, X[tr], y[tr], X[te])
                params_log.append({"repeat": rep, "fold": k, "features": fname, "model": model, **best})
            pred = P.argmax(1)
            # outlet level: average the probabilities over each held-out outlet's articles
            op = pd.DataFrame(P).groupby(g).mean()
            oy = df.groupby("source").y.first().loc[op.index].to_numpy()
            opred = op.to_numpy().argmax(1)
            rows.append({"repeat": rep, "features": fname, "model": model,
                         "article QWK": qwk(y, pred), "article macro F1": f1_score(y, pred, average="macro"),
                         "article accuracy": accuracy_score(y, pred),
                         "outlet QWK": qwk(oy, opred), "outlet macro F1": f1_score(oy, opred, average="macro"),
                         "outlet accuracy": accuracy_score(oy, opred), "seconds": time.time() - t})
            if rep == 0:
                oof.append(pd.DataFrame({"article_id": df.article_id if "article_id" in df else df.index,
                                         "source": g, "y": y, "features": fname, "model": model,
                                         **{f"p{c}": P[:, c] for c in range(5)}}))
            r = rows[-1]
            out(f"   rep {rep}  {fname:<12}{model:<15}article QWK {r['article QWK']:.3f}  "
                f"outlet QWK {r['outlet QWK']:.3f}  ({r['seconds']:.0f}s)")
            pd.DataFrame(rows).to_csv(f"{OUT}/08b_results.csv", index=False)

res = pd.DataFrame(rows)
pd.DataFrame(params_log).to_csv(f"{OUT}/08b_params.csv", index=False)
pd.concat(oof).to_parquet(f"{OUT}/08b_oof.parquet", index=False)

metrics = ["article QWK", "article macro F1", "outlet QWK", "outlet macro F1", "outlet accuracy"]
summ = res.groupby(["features", "model"])[metrics].mean()
out("\nNESTED, OUTLET-GROUPED PERFORMANCE (mean over repeats)")
out(summ.to_string(float_format=lambda v: f"{v:.3f}"))

out("\nDOES TEXT ADD ANYTHING ONCE BIAS IS KNOWN? (outlet-level QWK, text + bias minus bias only)")
for m in MODELS:
    a, b = summ.loc[("text + bias", m), "outlet QWK"], summ.loc[("bias only", m), "outlet QWK"]
    out(f"   {m:<15}{a:.3f} - {b:.3f} = {a - b:+.3f}")
out("\nCHOSEN PARAMETERS (how often each value won an outer fold)")
pl = pd.DataFrame(params_log)
for m in ["random forest", "boosting"]:
    cols = [c for c in pl.columns if c not in ("repeat", "fold", "features", "model")]
    sub = pl[pl.model == m].dropna(axis=1, how="all")
    keys = [c for c in cols if c in sub.columns]
    out(f"   {m}: " + ", ".join(
        f"{k} " + " / ".join(f"{v:g}: {int(n)}" for v, n in sub[k].value_counts().items()) for k in keys))
out(f"\nRESULT: wrote 08b_results.csv, 08b_params.csv, 08b_oof.parquet in {time.time() - t0:.0f}s")
with open(f"{OUT}/08b_report.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
