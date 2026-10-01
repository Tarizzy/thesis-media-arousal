"""08e_tfidf.py - does word-level text beat the eleven engineered features?
Identical outlet-grouped folds, identical outlet aggregation, identical metric, so the comparison is fair.
QUICK=1 runs a 1-minute version on a subsample. Writes ml/08e_report.txt and ml/08e_results.csv."""
import os, re, html, time
import numpy as np, pandas as pd, duckdb
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
PARQUET = f"{ROOT}/data/misinfo-general/data/*.parquet"
OUT = f"{ROOT}/ml"
SEED, FOLDS = 42, 5
QUICK = bool(os.environ.get("QUICK"))
REPEATS = 1 if QUICK else 2
MAXF = 5_000 if QUICK else 50_000
FACT = {"Very Low": 0, "Low": 1, "Mixed": 2, "Mostly Factual": 3, "High": 4, "Very High": 4}
FEATURES = ["arousal", "valence", "dominance", "attribution_density", "hedging", "certainty",
            "caps_ratio", "exclamation_rate", "flesch_kincaid", "mtld", "log_length"]
MASK_RE = re.compile(r"<copyright>|<twitter>|<url>|(?<!\w)<?selfref>?(?!\w)", re.IGNORECASE)
t0, lines = time.time(), []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


df = pd.read_parquet(f"{ROOT}/pipeline/analysis_table.parquet")
df["y"] = df.mbfc_factuality.map(FACT)
if QUICK:
    df = df[df.groupby("source").cumcount() < 20]
df = df.reset_index(drop=True)

con = duckdb.connect()
df = df.drop(columns=[c for c in ("content", "title") if c in df.columns])
con.register("want", df[["article_id"]])
txt = con.execute(f"""SELECT p.article_id, p.content FROM read_parquet('{PARQUET}') p
                      JOIN want w ON w.article_id = p.article_id""").df()
df = df.merge(txt, on="article_id", how="inner").reset_index(drop=True)
text = [re.sub(r"\s+", " ", MASK_RE.sub(" ", html.unescape(str(c)))) for c in df.content]
y = df.y.to_numpy()
g = df.source.to_numpy()
fk = df.flesch_kincaid.clip(*df.flesch_kincaid.quantile([0.01, 0.99]))
X = df[FEATURES].assign(flesch_kincaid=fk).to_numpy(float)
out(f"08e_tfidf - {time.strftime('%Y-%m-%d %H:%M')} - {len(df):,} articles, {df.source.nunique()} outlets"
    f"{' | QUICK TEST' if QUICK else ''}")
out(f"{FOLDS}-fold x {REPEATS} repeats, all splits grouped by outlet; TF-IDF fitted inside each fold, "
    f"max {MAXF:,} terms\n")

qwk = lambda a, b: cohen_kappa_score(a, b, weights="quadratic")
rows = []


def score(P, name, rep):
    art = P.argmax(1)
    op = pd.DataFrame(P).groupby(g).mean()
    oy = df.groupby("source").y.first().loc[op.index].to_numpy()
    opred = op.to_numpy().argmax(1)
    r = dict(repeat=rep, model=name, article_QWK=qwk(y, art), article_accuracy=accuracy_score(y, art),
             article_macroF1=f1_score(y, art, average="macro"), outlet_QWK=qwk(oy, opred),
             outlet_accuracy=accuracy_score(oy, opred), outlet_macroF1=f1_score(oy, opred, average="macro"),
             outlet_mean_abs_error=np.abs(oy - opred).mean())
    rows.append(r)
    out(f"   rep {rep}  {name:<26}article QWK {r['article_QWK']:.3f}   outlet QWK {r['outlet_QWK']:.3f}   "
        f"outlet acc {r['outlet_accuracy']:.3f}")
    return r


for rep in range(REPEATS):
    folds = list(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED + rep).split(df, y, g))
    P = {k: np.zeros((len(df), 5)) for k in ["TF-IDF words", "11 features, logistic",
                                             "11 features, boosting", "TF-IDF + 11 features"]}
    for k, (tr, te) in enumerate(folds):
        vec = TfidfVectorizer(strip_accents="unicode", min_df=5, max_features=MAXF, sublinear_tf=True)
        Ttr, Tte = vec.fit_transform([text[i] for i in tr]), vec.transform([text[i] for i in te])
        lr = lambda: OneVsRestClassifier(LogisticRegression(
            solver="liblinear", C=1.0, class_weight="balanced"))
        P["TF-IDF words"][te] = lr().fit(Ttr, y[tr]).predict_proba(Tte)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = np.ascontiguousarray(sc.transform(X[tr])), np.ascontiguousarray(sc.transform(X[te]))
        P["11 features, logistic"][te] = lr().fit(Xtr, y[tr]).predict_proba(Xte)
        P["11 features, boosting"][te] = HistGradientBoostingClassifier(
            learning_rate=0.02, max_leaf_nodes=7, max_iter=500, early_stopping=False,
            random_state=SEED).fit(X[tr], y[tr]).predict_proba(X[te])
        P["TF-IDF + 11 features"][te] = lr().fit(
            hstack([Ttr, csr_matrix(Xtr)]).tocsr(), y[tr]).predict_proba(
            hstack([Tte, csr_matrix(Xte)]).tocsr())
        out(f"   rep {rep} fold {k}: {Ttr.shape[1]:,} terms, {len(tr):,} train / {len(te):,} held-out "
            f"articles ({time.time() - t0:.0f}s)")
    for name in P:
        score(P[name], name, rep)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/08e_results.csv", index=False)
m = res.groupby("model").mean(numeric_only=True).drop(columns="repeat")
out("\n" + "=" * 96 + "\nMEAN OVER REPEATS (held-out outlets)")
out(m.round(3).to_string())
a, b = m.loc["TF-IDF words", "outlet_QWK"], m.loc["11 features, boosting", "outlet_QWK"]
c = m.loc["TF-IDF + 11 features", "outlet_QWK"]
out(f"\n   TF-IDF minus the best engineered-feature model: {a - b:+.3f} outlet QWK")
out(f"   TF-IDF + features minus TF-IDF alone:           {c - a:+.3f} outlet QWK")
out("   (the 08b2 tuned text-only outlet QWK is 0.563 forest / 0.564 boosting, on 5 repeats)")

out("\n" + "=" * 96 + "\nWHICH WORDS THE TF-IDF MODEL USES (fitted on all articles: in-sample, for illustration only)")
vec = TfidfVectorizer(strip_accents="unicode", min_df=5, max_features=MAXF, sublinear_tf=True)
T = vec.fit_transform(text)
clf = OneVsRestClassifier(LogisticRegression(
    solver="liblinear", C=1.0, class_weight="balanced")).fit(T, y)
names = np.array(vec.get_feature_names_out())
for cls, label in [(0, "Very Low"), (4, "High")]:
    w = clf.estimators_[cls].coef_[0]
    top = names[np.argsort(w)[-20:]][::-1]
    out(f"   towards {label:<9}: {', '.join(top)}")
open(f"{OUT}/08e_report.txt", "w").write("\n".join(lines) + "\n")
print(f"\nRESULT: wrote ml/08e_report.txt and 08e_results.csv in {(time.time() - t0) / 60:.0f} min")
