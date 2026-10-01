import duckdb
db = duckdb.connect("data/misinfo-general/metadata.db", read_only=True)

for col in ["label", "bias", "factuality", "credibility", "country"]:
    print(f"\n=== {col.upper()} ===")
    print(db.sql(f"""
        SELECT {col}, COUNT(*) AS n
        FROM sources GROUP BY {col} ORDER BY n DESC
    """))

print("\n=== HOW MANY HAVE BOTH BIAS AND FACTUALITY ===")
print(db.sql("""
    SELECT
      COUNT(*) AS total,
      COUNT(*) FILTER (WHERE bias IS NOT NULL) AS has_bias,
      COUNT(*) FILTER (WHERE factuality IS NOT NULL) AS has_fact,
      COUNT(*) FILTER (WHERE bias IS NOT NULL AND factuality IS NOT NULL) AS has_both
    FROM sources
"""))

print("\n=== CROSS-TAB: BIAS x FACTUALITY ===")
print(db.sql("""
    SELECT bias, factuality, COUNT(*) AS n
    FROM sources
    WHERE bias IS NOT NULL AND factuality IS NOT NULL
    GROUP BY 1, 2 ORDER BY 1, 2
"""))

print("\n=== ARTICLES PER SOURCE (for your >=50 trim) ===")
print(db.sql("""
    SELECT
      COUNT(*) AS n_sources,
      MIN(n_articles) AS min_art,
      MEDIAN(n_articles) AS med_art,
      MAX(n_articles) AS max_art,
      COUNT(*) FILTER (WHERE n_articles >= 50) AS surviving_50
    FROM (SELECT source, COUNT(*) AS n_articles FROM articles GROUP BY source)
"""))
