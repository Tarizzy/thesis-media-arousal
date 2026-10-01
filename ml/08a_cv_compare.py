"""08a_cv_compare.py - the headline methodological result: the same models scored three ways.
Naive article-level k-fold and random-forest OOB both let a model see the outlets it is tested on;
outlet-grouped k-fold does not. Also runs the leakage check (dropping the two features closest to an
outlet fingerprint) and two benchmarks. No tuning here - that is 08b.
Reads pipeline/analysis_table.parquet. Writes ml/08a_results.csv and ml/08a_report.txt."""
import os, sys, time
sys.path.insert(0, os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "pipeline"))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
import preprocessing as pp

OUT = f"{pp.ROOT}/ml"
os.makedirs(OUT, exist_ok=True)
FACT = {"Very Low": 0, "Low": 1, "Mixed": 2, "Mostly Factual": 3, "High": 4, "Very High": 4}
EXTREMITY = {"Least Biased": 0, "Left-Center": 1, "Right-Center": 1, "Left": 2, "Right": 2,
             "Extreme Left": 3, "Extreme Right": 3}
SIDE = {"Extreme Left": -1, "Left": -1, "Left-Center": -1, "Least Biased": 0,
        "Right-Center": 1, "Right": 1, "Extreme Right": 1}
FINGERPRINT = ["log_length", "caps_ratio"]      # closest to outlet house style
QUICK = os.environ.get("QUICK") == "1"          # smoke-test mode: 1 repeat, tiny forests
REPEATS, FOLDS = (1, 5) if QUICK else (3, 5)
N_TREES, N_ITER = (20, 20) if QUICK else (200, 150)
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


def qwk(y, p):
    return cohen_kappa_score(y, p, weights="quadratic")


t0 = time.time()
df = pd.read_parquet(f"{pp.ROOT}/pipeline/analysis_table.parquet")
df["y"] = df.mbfc_factuality.map(FACT)
df["bias_ext"] = df.mbfc_bias.map(EXTREMITY)
df["bias_side"] = df.mbfc_bias.map(SIDE)
if df[["y", "bias_ext"]].isna().any().any():
    raise SystemExit("factuality or bias not codable for some rows")
lo, hi = df.flesch_kincaid.quantile([0.01, 0.99])
df["flesch_kincaid"] = df.flesch_kincaid.clip(lo, hi)
groups = df.source.to_numpy()
y = df.y.to_numpy()

out(f"08a_cv_compare - {time.strftime('%Y-%m-%d %H:%M')}")
out(f"{len(df):,} articles | {df.source.nunique()} outlets | {FOLDS}-fold, {REPEATS} repeats"
    f"{' | QUICK SMOKE TEST' if QUICK else ''}")
out(f"flesch_kincaid winsorised to [{lo:.1f}, {hi:.1f}]")
cls = (df.groupby("y").agg(articles=("source", "size"), outlets=("source", "nunique")))
cls.index = ["Very Low", "Low", "Mixed", "Mostly Factual", "High (incl. Very High)"]
out("\nCLASSES (the effective sample size for grouped CV is the outlet count)")
out(cls.to_string())

FEATURE_SETS = {
    "all 11 features": pp.FEATURES,
    "without fingerprints": [f for f in pp.FEATURES if f not in FINGERPRINT],
    "bias only (benchmark)": ["bias_ext", "bias_side"],
}
MODELS = {
    "random forest": lambda: RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=20,
                                                    class_weight="balanced", n_jobs=-1, random_state=pp.SEED),
    "boosting": lambda: HistGradientBoostingClassifier(max_iter=N_ITER, early_stopping=False,
                                                       class_weight="balanced", random_state=pp.SEED),
}


def run_cv(make, X, splitter, seeds):
    """Out-of-fold predictions for each repeat; returns one metric row per repeat."""
    rows = []
    for s in seeds:
        sp = splitter(s)
        pred = np.empty(len(y), dtype=int)
        g = groups if isinstance(sp, StratifiedGroupKFold) else None
        for tr, te in sp.split(X, y, g):
            m = make().fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        rows.append({"QWK": qwk(y, pred), "macro F1": f1_score(y, pred, average="macro"),
                     "accuracy": accuracy_score(y, pred)})
    return pd.DataFrame(rows)


results = []
for fname, feats in FEATURE_SETS.items():
    X = df[feats].to_numpy(float)
    for mname, make in MODELS.items():
        for scheme, splitter in [
            ("naive k-fold", lambda s: StratifiedKFold(FOLDS, shuffle=True, random_state=s)),
            ("outlet-grouped", lambda s: StratifiedGroupKFold(FOLDS, shuffle=True, random_state=s)),
        ]:
            t = time.time()
            r = run_cv(make, X, splitter, range(REPEATS))
            results.append({"features": fname, "model": mname, "scheme": scheme,
                            **{f"{k} mean": v for k, v in r.mean().items()},
                            **{f"{k} sd": v for k, v in r.std().items()},
                            "seconds": time.time() - t})
            out(f"   {fname:<22}{mname:<15}{scheme:<16}QWK {r['QWK'].mean():.3f} "
                f"(sd {r['QWK'].std():.3f})  {time.time() - t:.0f}s")
            pd.DataFrame(results).to_csv(f"{OUT}/08a_results.csv", index=False)
        # random-forest OOB: bootstrap over articles, so an outlet is in-bag and out-of-bag at once
        if mname == "random forest":
            m = RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=20, class_weight="balanced",
                                       oob_score=True, bootstrap=True, n_jobs=-1, random_state=pp.SEED).fit(X, y)
            oob = m.classes_[np.argmax(m.oob_decision_function_, axis=1)]
            results.append({"features": fname, "model": mname, "scheme": "OOB (article bootstrap)",
                            "QWK mean": qwk(y, oob), "macro F1 mean": f1_score(y, oob, average="macro"),
                            "accuracy mean": accuracy_score(y, oob)})
            out(f"   {fname:<22}{mname:<15}{'OOB':<16}QWK {qwk(y, oob):.3f}")

res = pd.DataFrame(results)
res.to_csv(f"{OUT}/08a_results.csv", index=False)

out("\nQWK BY SCHEME (higher is better; 0 = no better than predicting one class)")
piv = res.pivot_table(index=["features", "model"], columns="scheme", values="QWK mean")
order = [c for c in ["naive k-fold", "OOB (article bootstrap)", "outlet-grouped"] if c in piv.columns]
out(piv[order].to_string(float_format=lambda v: f"{v:.3f}"))
out("\nMACRO F1 BY SCHEME")
out(res.pivot_table(index=["features", "model"], columns="scheme", values="macro F1 mean")[order]
    .to_string(float_format=lambda v: f"{v:.3f}"))

main = piv.loc["all 11 features"]
for m in main.index:
    gap = main.loc[m, "naive k-fold"] - main.loc[m, "outlet-grouped"]
    out(f"\n{m}: naive {main.loc[m, 'naive k-fold']:.3f} vs grouped {main.loc[m, 'outlet-grouped']:.3f} "
        f"-> naive overstates QWK by {gap:.3f}")
out(f"\nRESULT: wrote 08a_results.csv in {time.time() - t0:.0f}s")
with open(f"{OUT}/08a_report.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
