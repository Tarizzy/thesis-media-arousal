import duckdb

db = duckdb.connect("data/misinfo-general/metadata.db", read_only=True)

print("=== TOPICS SCHEMA ===")
print(db.sql("DESCRIBE topics"))

print("\n=== ARTICLES SCHEMA ===")
print(db.sql("DESCRIBE articles"))

print("\n=== SAMPLE (topics) ===")
print(db.sql("SELECT * FROM topics LIMIT 25"))

print("\n=== HOW MANY TOPICS ===")
print(db.sql("SELECT COUNT(*) FROM topics"))

print("\n=== BIGGEST TOPICS ===")
# First, detect the join column name dynamically
cols = db.sql("DESCRIBE articles").fetchdf()["column_name"].tolist()
print(f"[articles columns: {cols}]")

# Try common join column names
join_col = None
for candidate in ["topic", "topic_id", "cluster", "cluster_id"]:
    if candidate in cols:
        join_col = candidate
        break

if join_col:
    print(f"[joining on articles.{join_col} = topics.topic_id]")
    print(db.sql(f"""
        SELECT t.topic_id, t.year, t.representation, t.imbalanced_hyper_cluster, t.balanced_hyper_cluster,
               COUNT(*) AS n_articles
        FROM articles a JOIN topics t ON a.{join_col} = t.topic_id
        GROUP BY t.topic_id, t.year, t.representation, t.imbalanced_hyper_cluster, t.balanced_hyper_cluster
        ORDER BY n_articles DESC LIMIT 40
    """))
else:
    print(f"[could not find join column — articles columns are: {cols}]")
    print("[showing raw articles sample instead]")
    print(db.sql("SELECT * FROM articles LIMIT 10"))
