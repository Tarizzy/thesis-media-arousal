"""Embed the 41,705 per-year topic representations for clustering."""
import os
import duckdb, numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
OUT = f"{ROOT}/clustering"
os.makedirs(OUT, exist_ok=True)
con = duckdb.connect()
con.execute(f"ATTACH '{ROOT}/data/misinfo-general/metadata.db' AS meta (READ_ONLY)")

topics = con.sql("""
    SELECT topic_id, year, representation, imbalanced_hyper_cluster, balanced_hyper_cluster
    FROM meta.topics ORDER BY topic_id
""").df()
if len(topics) != 41705 or not topics["topic_id"].is_unique or topics["representation"].isna().any():
    raise SystemExit(f"Expected 41,705 unique topics with no missing representations, got {len(topics):,}. Stop here.")

topics["text"] = (topics["representation"].str.replace(r"^-?\d+_", "", regex=True)
                  .str.replace("_", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip())
if (topics["text"].str.len() == 0).any():
    raise SystemExit("Empty topic strings after cleaning. Stop here.")
print(topics[["representation", "text"]].head(5).to_string())

model = SentenceTransformer("all-mpnet-base-v2")
print("device:", model.device, flush=True)
X = model.encode(topics["text"].tolist(), batch_size=128, normalize_embeddings=True,
                 show_progress_bar=True).astype(np.float32)
np.save(f"{OUT}/topic_emb.npy", X)
con.register("t_out", topics)
con.execute(f"COPY (SELECT * FROM t_out ORDER BY topic_id) TO '{OUT}/topics.parquet' (FORMAT PARQUET)")
print("saved", X.shape)
