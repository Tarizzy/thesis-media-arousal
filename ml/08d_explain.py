"""08d_explain.py - what the text model uses, and in which direction.
  1  SHAP: for each outer fold of an outlet-grouped 5-fold split, a boosting model is trained and explained
     on its held-out outlets only, so every article's explanation comes from a model that never saw its outlet.
  2  Permutation importance on the same held-out outlets, for single features and for the three feature groups
     (emotion / epistemic / style), measured as the drop in article- and outlet-level QWK.
  3  Ordinal-logit coefficients (all articles, standardised features) with 95% CIs from resampling outlets:
     the plain-language direction of each feature, holding the other ten constant.
Boosting settings come from 08b2 (most frequent choice) if it has run, otherwise from 08b.
Reads pipeline/analysis_table.parquet. Writes ml/08d_*.csv, ml/08d_shap.parquet, figures, ml/08d_report.txt."""
import os, sys, time, warnings
sys.path.insert(0, os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "pipeline"))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score
from statsmodels.miscmodels.ordinal_model import OrderedModel
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import preprocessing as pp

warnings.filterwarnings("ignore")
OUT = f"{pp.ROOT}/ml"
os.makedirs(OUT, exist_ok=True)
QUICK = os.environ.get("QUICK") == "1"
FOLDS, NC = 5, 5
PERM_REPEATS = 1 if QUICK else 3          # repeats of the whole fold split for permutation importance
N_PERM = 3 if QUICK else 10               # shuffles per feature per fold
B_ORD = 20 if QUICK else 500              # outlet bootstrap draws for the ordinal coefficients
MAX_ITER, PATIENCE = (60, 5) if QUICK else (2000, 30)
FACT = {"Very Low": 0, "Low": 1, "Mixed": 2, "Mostly Factual": 3, "High": 4, "Very High": 4}
LEVELS = ["Very Low", "Low", "Mixed", "Mostly Factual", "High"]
GROUPS = {"emotion": ["arousal", "valence", "dominance"],
          "epistemic": ["attribution_density", "hedging", "certainty"],
          "style": ["caps_ratio", "exclamation_rate", "flesch_kincaid", "mtld", "log_length"]}
F = pp.FEATURES
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


def qwk(a, b):
    return cohen_kappa_score(a, b, weights="quadratic")


# ---------- settings ----------
params, source = {"learning_rate": 0.05, "max_leaf_nodes": 15}, "08b (most frequent choice)"
pfile = f"{OUT}/08b2_params.csv"
if os.path.exists(pfile):
    pl = pd.read_csv(pfile)
    pl = pl[(pl.model == "boosting") & (pl.features == "text only")]
    if len(pl):
        params = {"learning_rate": float(pl.learning_rate.mode()[0]), "max_leaf_nodes": int(pl.max_leaf_nodes.mode()[0])}
        source = "08b2 (most frequent choice for text only)"
BASE = dict(min_samples_leaf=50, l2_regularization=1.0, class_weight="balanced", **params)


def fit_boost(Xtr, ytr, gtr, seed):
    """Early stopping on an outlet-grouped validation split, then refit on all training outlets."""
    fi, vi = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xtr, ytr, gtr))
    es = HistGradientBoostingClassifier(max_iter=MAX_ITER, early_stopping=True, n_iter_no_change=PATIENCE,
                                        random_state=seed, **BASE).fit(Xtr[fi], ytr[fi], X_val=Xtr[vi], y_val=ytr[vi])
    m = HistGradientBoostingClassifier(max_iter=es.n_iter_, early_stopping=False, random_state=seed,
                                       **BASE).fit(Xtr, ytr)
    return m


def proba(m, X):
    P = np.zeros((len(X), NC))
    P[:, m.classes_] = m.predict_proba(X)
    return P


def scores(P, y, codes):
    """Article-level and outlet-level QWK from a full out-of-fold probability matrix."""
    op = pd.DataFrame(P).groupby(codes).mean()
    oy = pd.Series(y).groupby(codes).first().loc[op.index].to_numpy()
    return qwk(y, P.argmax(1)), qwk(oy, op.to_numpy().argmax(1))


# ---------- data ----------
t0 = time.time()
df = pd.read_parquet(f"{pp.ROOT}/pipeline/analysis_table.parquet")
if QUICK:
    df = df.sample(frac=1, random_state=1).groupby("source").head(40)
df = df.reset_index(drop=True)
df["y"] = df.mbfc_factuality.map(FACT)
lo, hi = df.flesch_kincaid.quantile([0.01, 0.99])
df["flesch_kincaid"] = df.flesch_kincaid.clip(lo, hi)
codes, sources = pd.factorize(df.source)
X, y, g = df[F].to_numpy(float), df.y.to_numpy(), df.source.to_numpy()
out(f"08d_explain - {time.strftime('%Y-%m-%d %H:%M')} - {len(df):,} articles, {len(sources)} outlets"
    f"{' | QUICK SMOKE TEST' if QUICK else ''}")
out(f"boosting settings {params} from {source}; rounds by outlet-grouped early stopping in every fold")

# ---------- 1: SHAP on held-out outlets ----------
out("\n" + "=" * 100 + "\n1  SHAP (each article explained by a model that never saw its outlet)")
folds = list(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=pp.SEED).split(X, y, codes))
SV = np.zeros((len(df), len(F), NC))
P0 = np.zeros((len(df), NC))
models0 = []
for k, (tr, te) in enumerate(folds):
    m = fit_boost(X[tr], y[tr], g[tr], pp.SEED + k)
    models0.append(m)
    P0[te] = proba(m, X[te])
    sv = shap.TreeExplainer(m)(X[te]).values                   # (n_te, features, classes present)
    SV[np.ix_(te, np.arange(len(F)), m.classes_)] = sv
    out(f"   fold {k}: {len(te):,} held-out articles, {m.n_iter_} rounds")
a0, o0 = scores(P0, y, codes)
out(f"   out-of-fold QWK of the explained models: article {a0:.3f}, outlet {o0:.3f}")
shap_df = pd.DataFrame(SV.reshape(len(df), -1).astype("float32"),
                       columns=[f"{f}|{LEVELS[c]}" for f in F for c in range(NC)])
shap_df.insert(0, "y", y)
shap_df.insert(0, "source", g)
if "article_id" in df:
    shap_df.insert(0, "article_id", df.article_id.to_numpy())
shap_df.to_parquet(f"{OUT}/08d_shap.parquet", index=False)

mean_abs = pd.DataFrame(np.abs(SV).mean(0), index=F, columns=LEVELS)
mean_abs["all classes"] = mean_abs.sum(1)
direction = pd.DataFrame({f"rho | {LEVELS[c]}": [spearmanr(X[:, j], SV[:, j, c]).statistic for j in range(len(F))]
                          for c in (0, 4)}, index=F)
summ = mean_abs.join(direction).sort_values("all classes", ascending=False)
summ.insert(0, "group", [next(gn for gn, fs in GROUPS.items() if f in fs) for f in summ.index])
summ.to_csv(f"{OUT}/08d_shap_summary.csv", float_format="%.4f")
out("\n   mean |SHAP| per class (log-odds units; bigger = the feature moves that class's score more), and the")
out("   direction: Spearman rho between a feature's value and its SHAP value. rho > 0 for Very Low means higher")
out("   values push an article TOWARDS Very Low; rho > 0 for High means higher values push it towards High.")
out(summ.to_string(float_format=lambda v: f"{v:+.3f}"))
grp = pd.DataFrame({gn: mean_abs.loc[fs, "all classes"].sum() for gn, fs in GROUPS.items()}, index=["mean |SHAP|"]).T
grp["share"] = grp["mean |SHAP|"] / grp["mean |SHAP|"].sum()
out("\n   by feature group (sum over the group's features and all classes)")
out(grp.to_string(float_format=lambda v: f"{v:.3f}"))

# figures: beeswarms for the two ends of the scale, bar chart, dependence panels
rs = np.random.default_rng(pp.SEED)
samp = rs.choice(len(df), size=min(len(df), 10_000), replace=False)
for c in (0, 4):
    ex = shap.Explanation(values=SV[samp, :, c], data=X[samp], feature_names=F)
    plt.figure()
    shap.plots.beeswarm(ex, max_display=len(F), show=False)
    plt.title(f"SHAP for the '{LEVELS[c]}' class (held-out outlets, 10,000-article sample)", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_08d_beeswarm_{LEVELS[c].replace(' ', '_').lower()}.png", dpi=200)
    plt.close("all")
fig, ax = plt.subplots(figsize=(8, 5))
order = mean_abs.sort_values("all classes").index
left = np.zeros(len(order))
for c in range(NC):
    ax.barh(order, mean_abs.loc[order, LEVELS[c]], left=left, label=LEVELS[c], color=str(0.15 + 0.17 * c))
    left += mean_abs.loc[order, LEVELS[c]].to_numpy()
ax.set_xlabel("mean |SHAP| summed over classes (held-out outlets)")
ax.legend(fontsize=8, title="class", title_fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_08d_shap_bar.png", dpi=200); plt.close(fig)
dep = ["arousal", "valence", "dominance", "exclamation_rate", "caps_ratio", "certainty", "attribution_density",
       "log_length"]
for c in (0, 4):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, f in zip(axes.ravel(), dep):
        j = F.index(f)
        x, s = X[:, j], SV[:, j, c]
        lo_, hi_ = np.nanpercentile(x, [1, 99])
        keep = (x >= lo_) & (x <= hi_)
        ax.scatter(x[samp][keep[samp]], s[samp][keep[samp]], s=2, alpha=0.15, color="grey")
        bins = np.unique(np.nanpercentile(x[keep], np.linspace(0, 100, 21)))
        if len(bins) > 2:
            idx = np.clip(np.digitize(x[keep], bins) - 1, 0, len(bins) - 2)
            mids = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
            means = [s[keep][idx == i].mean() if (idx == i).any() else np.nan for i in range(len(bins) - 1)]
            ax.plot(mids, means, color="black", lw=1.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f, fontsize=9)
    fig.suptitle(f"SHAP dependence for the '{LEVELS[c]}' class: line = binned mean (above 0 pushes towards "
                 f"{LEVELS[c]})", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_08d_dependence_{LEVELS[c].replace(' ', '_').lower()}.png", dpi=200)
    plt.close(fig)

# ---------- 2: permutation importance on held-out outlets ----------
out("\n" + "=" * 100 + "\n2  PERMUTATION IMPORTANCE (held-out outlets; drop in QWK when a feature's values are shuffled)")
targets = {f: [F.index(f)] for f in F}
targets.update({f"GROUP {gn}": [F.index(f) for f in fs] for gn, fs in GROUPS.items()})
drops = {t: [] for t in targets}
for r in range(PERM_REPEATS):
    fr = folds if r == 0 else list(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=pp.SEED + r)
                                   .split(X, y, codes))
    ms = models0 if r == 0 else [fit_boost(X[tr], y[tr], g[tr], pp.SEED + 10 * r + k) for k, (tr, _) in enumerate(fr)]
    P = np.zeros((len(df), NC))
    for (tr, te), m in zip(fr, ms):
        P[te] = proba(m, X[te])
    base_a, base_o = scores(P, y, codes)
    rng = np.random.default_rng(pp.SEED + r)
    for name, cols in targets.items():
        for b in range(N_PERM):
            Pp = np.zeros_like(P)
            for (tr, te), m in zip(fr, ms):
                Xp = X[te].copy()
                Xp[:, cols] = Xp[rng.permutation(len(te))][:, cols]    # one shuffle for the whole group
                Pp[te] = proba(m, Xp)
            a, o = scores(Pp, y, codes)
            drops[name].append((base_a - a, base_o - o))
    out(f"   repeat {r}: baseline QWK article {base_a:.3f}, outlet {base_o:.3f}")
perm = pd.DataFrame({t: {"article drop": np.mean([d[0] for d in v]),
                         "article 2.5%": np.percentile([d[0] for d in v], 2.5),
                         "article 97.5%": np.percentile([d[0] for d in v], 97.5),
                         "outlet drop": np.mean([d[1] for d in v]),
                         "outlet 2.5%": np.percentile([d[1] for d in v], 2.5),
                         "outlet 97.5%": np.percentile([d[1] for d in v], 97.5)} for t, v in drops.items()}).T
perm = perm.sort_values("outlet drop", ascending=False)
perm.to_csv(f"{OUT}/08d_permutation.csv", float_format="%.4f")
out(f"   ({PERM_REPEATS} fold splits x {N_PERM} shuffles; interval = spread of the drops across shuffles and splits)")
out(perm.to_string(float_format=lambda v: f"{v:+.3f}"))
fig, ax = plt.subplots(figsize=(7, 6))
pp_ = perm.sort_values("outlet drop")
ax.errorbar(pp_["outlet drop"], range(len(pp_)), xerr=[pp_["outlet drop"] - pp_["outlet 2.5%"],
                                                       pp_["outlet 97.5%"] - pp_["outlet drop"]], fmt="o", color="black",
            ms=4, capsize=2)
ax.set_yticks(range(len(pp_)), pp_.index)
ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("drop in outlet-level QWK when shuffled (held-out outlets)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_08d_permutation.png", dpi=200); plt.close(fig)

# ---------- 3: ordinal logit coefficients with outlet-bootstrap CIs ----------
out("\n" + "=" * 100 + "\n3  ORDINAL LOGIT ON ALL ARTICLES (standardised features, other ten held constant)")
mu, sd = X.mean(0), X.std(0)
Z = (X - mu) / sd
fit = OrderedModel(y, Z, distr="logit").fit(method="bfgs", maxiter=500, disp=False)
coef = np.asarray(fit.params[:len(F)])
idx_by_outlet = [np.where(codes == i)[0] for i in range(len(sources))]
rng = np.random.default_rng(pp.SEED)
boot = []
for b in range(B_ORD):
    pick = rng.integers(0, len(sources), len(sources))
    rows = np.concatenate([idx_by_outlet[i] for i in pick])
    try:
        rb = OrderedModel(y[rows], Z[rows], distr="logit").fit(method="bfgs", maxiter=500, disp=False)
        boot.append(np.asarray(rb.params[:len(F)]))
    except Exception:
        continue
boot = np.array(boot)
ords = pd.DataFrame({"group": [next(gn for gn, fs in GROUPS.items() if f in fs) for f in F],
                     "coef per SD": coef, "odds ratio": np.exp(coef),
                     "OR 2.5%": np.exp(np.percentile(boot, 2.5, axis=0)),
                     "OR 97.5%": np.exp(np.percentile(boot, 97.5, axis=0))}, index=F)
ords["reading"] = [("higher -> MORE factual outlet" if lo_ > 1 else "higher -> LESS factual outlet" if hi_ < 1
                    else "no clear direction") for lo_, hi_ in zip(ords["OR 2.5%"], ords["OR 97.5%"])]
ords = ords.sort_values("coef per SD")
ords.to_csv(f"{OUT}/08d_ordinal_coefs.csv", float_format="%.4f")
out(f"   odds ratio = change in the odds of a HIGHER factuality class per 1 SD of the feature;")
out(f"   95% CI from {len(boot)} outlet-bootstrap refits (articles resampled with their outlet)")
out(ords.to_string(float_format=lambda v: f"{v:.3f}"))

out(f"\nRESULT: wrote 08d_shap.parquet, 08d_shap_summary.csv, 08d_permutation.csv, 08d_ordinal_coefs.csv and "
    f"7 figures in {(time.time() - t0) / 60:.0f} min")
with open(f"{OUT}/08d_report.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
