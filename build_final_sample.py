import re
import duckdb
import pandas as pd

MIN_POLITICAL_ARTICLES = 200
MIN_POLITICAL_SHARE = 0.10

POLITICAL_KEYWORDS = {
    "white house", "congress", "senate", "house of representatives",
    "supreme court", "pentagon", "cia", "fbi", "doj", "department of",
    "parliament", "kremlin", "united nations", "u.n.", "nato",
    "european union", "european commission", "european parliament",
    "downing street", "westminster", "house of commons", "house of lords",
    "bundestag", "state department", "foreign ministry", "security council",
    "imf", "world bank", "opec", "g7", "g20",
    "president", "vice president", "senator", "congressman", "congresswoman",
    "governor", "mayor", "secretary of state", "attorney general",
    "democrat", "republican", "gop", "liberal", "conservative",
    "labour", "tory", "tories", "lawmaker", "politician", "administration",
    "trump", "biden", "harris", "obama", "pelosi", "mcconnell",
    "prime minister", "chancellor", "minister", "diplomat", "ambassador",
    "putin", "zelensky", "xi jinping", "modi", "netanyahu", "erdogan",
    "macron", "scholz", "merkel", "starmer", "sunak", "trudeau", "albanese",
    "election", "vote", "voting", "ballot", "midterm",
    "primary election", "presidential primary",
    "referendum", "general election", "by-election", "no confidence",
    "legislation", "bill", "law", "regulation", "executive order",
    "impeach", "filibuster", "veto", "amendment", "constitution",
    "cabinet", "reshuffle", "coalition government", "brexit",
    "immigration", "border", "tariff", "sanctions", "foreign policy",
    "climate", "healthcare", "abortion", "gun control", "gun laws",
    "gun violence", "tax", "budget", "deficit", "inflation",
    "federal reserve", "spending", "debt ceiling",
    "military", "war", "ceasefire", "treaty", "diplomacy", "airstrike",
    "nuclear deal", "summit", "annex", "occupation",
    "scandal", "investigation", "indictment", "probe", "hearing",
    "testimony", "protest", "riot", "coup", "geopolitical", "political",
}

pattern = r"\b(?:" + "|".join(
    re.escape(k) for k in sorted(POLITICAL_KEYWORDS, key=len, reverse=True)
) + r")\b"

with open("political_pattern.txt", "w") as f:
    f.write(pattern)

db = duckdb.connect("data/misinfo-general/metadata.db", read_only=True)

print("=== TYPE DISTRIBUTION ===")
print(db.sql("SELECT type, COUNT(*) n FROM sources GROUP BY type ORDER BY n DESC"))

print("\nScanning titles - takes a minute...")
stats = db.execute("""
    SELECT
      source,
      COUNT(*) AS n_articles,
      COUNT(*) FILTER (WHERE regexp_matches(lower(title), ?)) AS n_political
    FROM articles
    GROUP BY source
""", [pattern]).df()

stats["political_share"] = stats.n_political / stats.n_articles

print("Pulling domains from parquet...")
try:
    dom = db.execute("""
        SELECT source, MODE(domain) AS domain
        FROM read_parquet('data/misinfo-general/data/*.parquet')
        WHERE domain IS NOT NULL AND domain != 'N/A'
        GROUP BY source
    """).df()
    stats = stats.merge(dom, on="source", how="left")
except Exception as e:
    print(f"domain lookup failed ({e}) - continuing without it")
    stats["domain"] = ""

src = db.sql("SELECT * FROM sources").df()
df = src.merge(stats, on="source", how="left")
df[["n_articles", "n_political", "political_share"]] = \
    df[["n_articles", "n_political", "political_share"]].fillna(0)
if "domain" not in df.columns:
    df["domain"] = ""
df["domain"] = df["domain"].fillna("")

def reason(r):
    if pd.isna(r.bias):       return "no bias rating"
    if pd.isna(r.factuality): return "no factuality rating"
    if r.label in ("Satire", "Pro-Science"):
        return f"source type: {r.label}"
    if r.n_political < MIN_POLITICAL_ARTICLES:
        return f"too few political articles ({int(r.n_political)})"
    if r.political_share < MIN_POLITICAL_SHARE:
        return f"political share too low ({r.political_share:.0%})"
    return "KEEP"

df["cut_reason"] = df.apply(reason, axis=1)
eligible = df[df.cut_reason == "KEEP"].sort_values("n_political", ascending=False)

base = df.bias.notna() & df.factuality.notna()
notype = base & ~df.label.isin(["Satire", "Pro-Science"])
enough = notype & (df.n_political >= MIN_POLITICAL_ARTICLES)
funnel = pd.DataFrame({
    "stage": ["All sources", "Has bias", "Has factuality",
              "Not satire/pro-science",
              f"Political articles >= {MIN_POLITICAL_ARTICLES}",
              f"Political share >= {MIN_POLITICAL_SHARE:.0%}"],
    "n": [len(df), df.bias.notna().sum(), base.sum(), notype.sum(),
          enough.sum(), len(eligible)],
})

cols = ["source", "domain", "n_articles", "n_political", "political_share",
        "label", "bias", "factuality", "credibility", "type", "country",
        "conspiracy", "pseudosci", "check_date"]
out = eligible[cols].copy()
out["adfontes_bias"] = ""
out["adfontes_reliability"] = ""
out["match_notes"] = ""

borderline = df[notype & (df.n_political >= MIN_POLITICAL_ARTICLES)] \
    .sort_values("political_share")[cols + ["cut_reason"]]

with pd.ExcelWriter("final_sample.xlsx", engine="openpyxl") as w:
    out.to_excel(w, sheet_name="Eligible", index=False)
    df[cols + ["cut_reason"]].sort_values("n_articles", ascending=False) \
      .to_excel(w, sheet_name="All_With_Reasons", index=False)
    borderline.to_excel(w, sheet_name="Share_Calibration", index=False)
    funnel.to_excel(w, sheet_name="Funnel", index=False)

print(funnel.to_string(index=False))
print(f"\nFINAL N: {len(eligible)}")
print(eligible.bias.value_counts())
print(eligible.label.value_counts())
