"""K sweep, final clustering, validity checks, cluster profiles and article assignment.
Reads topic_emb.npy and topics.parquet from 01_embed.py."""
import os
from collections import Counter
import duckdb, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_mutual_info_score as ami, adjusted_rand_score as ari,
                             calinski_harabasz_score, davies_bouldin_score, pairwise_distances, silhouette_score)

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
OUT = f"{ROOT}/clustering"
SEED = 42
SWEEP_N = int(os.environ.get("SWEEP_N", 15000))   # a full distance matrix at n=41,705 needs ~7 GB per copy
K_MIN, K_MAX = 5, int(os.environ.get("K_MAX", 60))
PICK_LOW, PICK_HIGH = 20, 35                      # K range fixed before seeing the sweep
MAX_CLUSTER_SHARE = 0.15                          # a solution where one cluster swallows more is rejected

L = []
def say(*parts):
    line = " ".join(str(p) for p in parts)
    L.append(line)
    print(line, flush=True)

topics = duckdb.sql(f"SELECT * FROM '{OUT}/topics.parquet' ORDER BY topic_id").df().reset_index(drop=True)
X = np.load(f"{OUT}/topic_emb.npy")
say(f"{len(topics):,} topics | embeddings {X.shape}")

# ---------- 1. K sweep on a year-stratified sample ----------
n = min(SWEEP_N, len(topics))
idx = topics.groupby("year").sample(frac=n / len(topics), random_state=SEED).index.to_numpy()
np.save(f"{OUT}/sample_idx.npy", idx)
Xs = X[idx]
D = pairwise_distances(Xs).astype(np.float32)   # unit vectors: euclidean is monotone in cosine
np.fill_diagonal(D, 0)
Z = {m: linkage(squareform(D, checks=False), method=m) for m in ["ward", "complete", "average"]}
say(f"sweep sample {len(idx):,} topics | linkage done")

rows = []
for k in range(K_MIN, K_MAX + 1):
    labs = {m: fcluster(Z[m], k, criterion="maxclust") for m in Z}
    labs["kmeans"] = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Xs)
    for m, lab in labs.items():
        rows.append(dict(method=m, k=k, sil=silhouette_score(D, lab, metric="precomputed"),
                         ch=calinski_harabasz_score(Xs, lab), db=davies_bouldin_score(Xs, lab),
                         max_share=np.bincount(lab).max() / len(lab)))
    pd.DataFrame(rows).to_csv(f"{OUT}/k_sweep.csv", index=False)
    print(f"K={k} done", flush=True)
res = pd.DataFrame(rows)

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, col in zip(ax, ["sil", "ch", "db", "max_share"]):
    res.pivot(index="k", columns="method", values=col).plot(ax=a, title=col)
    a.axvspan(PICK_LOW, PICK_HIGH, alpha=0.1)
plt.tight_layout()
plt.savefig(f"{OUT}/k_sweep.png", dpi=150)

eligible = res[res["k"].between(PICK_LOW, PICK_HIGH) & (res["max_share"] <= MAX_CLUSTER_SHARE)]
if eligible.empty:
    raise SystemExit("No solution in K = 20-35 keeps every cluster under 15% of topics. Stop here.")
pick = eligible.loc[eligible["sil"].idxmax()]
METHOD, K = pick["method"], int(pick["k"])
say(f"\nRULE: highest silhouette in K = {PICK_LOW}-{PICK_HIGH} with no cluster above {MAX_CLUSTER_SHARE:.0%}"
    f" -> {METHOD}, K = {K}")
say(res[res["k"].between(PICK_LOW, PICK_HIGH)].round(3).to_string(index=False))

# ---------- 2. final fit on all topics ----------
if METHOD == "kmeans":
    fits = [KMeans(n_clusters=K, n_init=10, random_state=s).fit(X) for s in range(5)]
    best = min(fits, key=lambda f: f.inertia_)
    labels, centres = best.labels_, best.cluster_centers_
    stab = [ari(fits[i].labels_, fits[j].labels_) for i in range(5) for j in range(i + 1, 5)]
    say(f"\nseed stability, pairwise ARI over 5 seeds: mean {np.mean(stab):.3f}, min {np.min(stab):.3f}")
else:
    sample_labels = fcluster(Z[METHOD], K, criterion="maxclust")
    centres = np.vstack([Xs[sample_labels == c].mean(0) for c in np.unique(sample_labels)])
    labels = np.argmax(X @ (centres / np.linalg.norm(centres, axis=1, keepdims=True)).T, axis=1)
    say(f"\n{METHOD} linkage fitted on the sweep sample; all topics assigned to the nearest cluster centre")
topics["cluster"] = labels
sil_full = silhouette_score(X, labels, sample_size=min(15000, len(X)), random_state=SEED)
say(f"silhouette on all topics (15k sample): {sil_full:.4f}")

# ---------- 3. validity ----------
def cramers_v(a, b):
    ct = pd.crosstab(a, b)
    return np.sqrt(chi2_contingency(ct)[0] / (ct.values.sum() * (min(ct.shape) - 1)))

rng = np.random.default_rng(SEED)
v = cramers_v(topics["cluster"].values, topics["year"].values)
v_null = [cramers_v(rng.permutation(topics["cluster"].values), topics["year"].values) for _ in range(20)]
span = topics.groupby("cluster")["year"].nunique()
say(f"\nCramer's V cluster x year: {v:.3f} (shuffled baseline mean {np.mean(v_null):.3f}, max {np.max(v_null):.3f})")
say(f"mean YearSpan {span.mean():.2f} | clusters covering all six years: {(span == 6).sum()}/{K}")
yr = pd.DataFrame([dict(year=y,
                        ari_imbalanced=ari(g["imbalanced_hyper_cluster"], g["cluster"]),
                        ami_imbalanced=ami(g["imbalanced_hyper_cluster"], g["cluster"]),
                        ari_balanced=ari(g["balanced_hyper_cluster"], g["cluster"]),
                        ami_balanced=ami(g["balanced_hyper_cluster"], g["cluster"]))
                   for y, g in topics.groupby("year")])
say("\nAGREEMENT WITH THE SUPPLIED HYPER-CLUSTERS, WITHIN EACH YEAR\n" + yr.round(3).to_string(index=False))
say("mean: " + str(yr.drop(columns="year").mean().round(3).to_dict()))

# ---------- 4. cluster profiles for hand-labelling ----------
con = duckdb.connect()
con.execute(f"ATTACH '{ROOT}/data/misinfo-general/metadata.db' AS meta (READ_ONLY)")
counts = con.execute("SELECT topic_id, COUNT(*) AS n FROM meta.articles GROUP BY topic_id").df()
topics["n_articles"] = topics["topic_id"].map(counts.set_index("topic_id")["n"]).fillna(0).astype(int)
C = centres / np.linalg.norm(centres, axis=1, keepdims=True)
sims = X @ C.T
total = topics["n_articles"].sum()
prof = []
for c in range(K):
    m = (topics["cluster"] == c).values
    g = topics[m].assign(sim=sims[m, c])
    kw = Counter(w.strip() for r in g["representation"] for w in r.split(","))
    prof.append(f"\n=== Cluster {c} | topics {len(g):,} | articles {g['n_articles'].sum():,} "
                f"({100 * g['n_articles'].sum() / total:.1f}%) | years {g['year'].nunique()}")
    prof.append("keywords: " + ", ".join(w for w, _ in kw.most_common(15)))
    prof.append("medoids:")
    prof += [f"  {r}" for r in g.nlargest(5, "sim")["representation"]]
    prof.append("largest topics:")
    prof += [f"  [{int(a):,}] {r}" for r, a in g.nlargest(5, "n_articles")[["representation", "n_articles"]].values]
with open(f"{OUT}/cluster_profiles.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(prof))

sizes = topics.groupby("cluster").agg(topics=("topic_id", "size"), articles=("n_articles", "sum"))
sizes["article_pct"] = (100 * sizes["articles"] / total).round(2)
say("\nCLUSTER SIZES\n" + sizes.to_string())
say("\n15 LARGEST TOPICS OVERALL\n" +
    topics.nlargest(15, "n_articles")[["topic_id", "year", "cluster", "n_articles", "representation"]].to_string(index=False))

# ---------- 5. assign topic_cluster to every article ----------
con.register("tc_out", topics[["topic_id", "year", "cluster", "n_articles"]])
con.execute(f"COPY (SELECT * FROM tc_out ORDER BY topic_id) TO '{OUT}/topic_clusters.parquet' (FORMAT PARQUET)")
con.register("tc", topics[["topic_id", "cluster"]])
con.execute(f"""COPY (SELECT a.article_id, a.year, a.source, tc.cluster AS topic_cluster
                      FROM meta.articles a JOIN tc USING (topic_id))
                TO '{OUT}/article_topic_cluster.parquet' (FORMAT PARQUET)""")
n_assigned = con.execute(f"SELECT COUNT(*) FROM '{OUT}/article_topic_cluster.parquet'").fetchone()[0]
n_articles = con.execute("SELECT COUNT(*) FROM meta.articles").fetchone()[0]
say(f"\narticles assigned a topic cluster: {n_assigned:,} of {n_articles:,}")

with open(f"{OUT}/02_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(L))
