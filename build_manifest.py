import duckdb

db = duckdb.connect("data/misinfo-general/metadata.db", read_only=True)

manifest = db.sql("""
    SELECT
      s.source,
      s.url,
      s.country,
      s.label,
      s.bias,
      s.factuality,
      s.credibility,
      s.conspiracy,
      s.pseudosci,
      s.check_date,
      a.n_articles
    FROM sources s
    LEFT JOIN (
      SELECT source, COUNT(*) AS n_articles
      FROM articles GROUP BY source
    ) a ON s.source = a.source
    ORDER BY a.n_articles DESC NULLS LAST
""").df()

manifest.to_csv("source_manifest.csv", index=False)

print(f"rows: {len(manifest)}")
print(manifest.head(30).to_string())

# the subset you'll actually try to match
eligible = manifest[
    manifest.bias.notna()
    & manifest.factuality.notna()
    & (manifest.label != "Satire")
    & (manifest.n_articles >= 50)
]
eligible.to_csv("source_manifest_eligible.csv", index=False)
print(f"\neligible for Ad Fontes matching: {len(eligible)}")
print(eligible.bias.value_counts())
