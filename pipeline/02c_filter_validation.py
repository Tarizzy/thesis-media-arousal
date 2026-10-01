import math, os, re
import duckdb, pandas as pd

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
DATA = f"{ROOT}/data/misinfo-general"
PQ = f"{DATA}/data/*.parquet"
OUT = f"{ROOT}/pipeline"
L = []
def say(*parts):
    L.append(" ".join(str(p) for p in parts))

# ---------- keyword lists ----------
def term_re(t, names=frozenset()):
    left = r"\b" if re.match(r"\w", t[0], re.ASCII) else ""
    right = r"\b" if re.match(r"\w", t[-1], re.ASCII) else ""
    return left + re.escape(t) + ("s?" if t in names and right else "") + right

v1 = open(f"{ROOT}/political_pattern.txt", encoding="utf-8").read().strip()
v2 = open(f"{ROOT}/political_pattern_v2.txt", encoding="utf-8").read().strip()
terms = [line.split("\t")[0] for line in open(f"{ROOT}/political_terms_v2.txt", encoding="utf-8").read().splitlines() if line.strip()]
if "(?:" + "|".join(term_re(t) for t in terms) + ")" != v2:
    raise SystemExit("political_terms_v2.txt does not rebuild political_pattern_v2.txt. Stop here.")

# v2.1 = v2 terms + optional possessive 's' on names and organisations (2018 titles lost apostrophes: "Trumps")
NAMES = {
    "trump", "biden", "harris", "obama", "pelosi", "mcconnell", "putin", "zelensky", "xi jinping", "modi",
    "netanyahu", "erdogan", "macron", "scholz", "merkel", "starmer", "sunak", "trudeau", "albanese",
    "pence", "schumer", "mueller", "comey", "kavanaugh", "pompeo", "tillerson", "giuliani", "manafort",
    "kushner", "bernie sanders", "elizabeth warren", "hillary", "clinton", "ocasio-cortez", "desantis",
    "cuomo", "newsom", "cheney", "kamala", "buttigieg", "romney", "ted cruz", "lindsey graham",
    "kevin mccarthy", "theresa may", "boris johnson", "boris", "corbyn", "farage", "sturgeon", "raab",
    "gove", "truss", "javid", "priti patel", "dominic cummings", "bolsonaro", "maduro", "kim jong un",
    "kim jong-un", "duterte", "orban", "orbán", "lukashenko", "assad", "khamenei", "rouhani", "shinzo abe",
    "scott morrison", "ardern", "zelenskyy", "zelenskiy", "navalny",
    "gop", "fbi", "cia", "doj", "nato", "imf", "opec", "kremlin", "pentagon", "dnc", "rnc", "snp", "ukip",
    "scotus", "potus", "taliban", "hamas", "white house", "downing street", "state department",
    "supreme court", "senate", "bundestag", "westminster", "whitehall",
}
missing = sorted(NAMES - set(terms))
if missing:
    raise SystemExit(f"Names not in the v2 term list: {missing}. Stop here.")
v21 = "(?:" + "|".join(term_re(t, NAMES) for t in terms) + ")"
with open(f"{ROOT}/political_pattern_v21.txt", "w", encoding="utf-8") as f:
    f.write(v21)

# title repair applied before lowercasing: HTML entities, and entities that lost their & and ; ("Californiaaposs")
ENTITIES = [("&apos;", "'"), ("&#039;", "'"), ("&#39;", "'"), ("&#8217;", "'"), ("&#8216;", "'"),
            ("&quot;", '"'), ("&#034;", '"'), ("&#8220;", '"'), ("&#8221;", '"'), ("&amp;", "&"), ("&#8230;", "...")]
GLUED = [(r"apos([A-Z])", r"'\1"), (r"([A-Za-z])apos(s|t|m|d|re|ve|ll)\b", r"\1'\2"), (r"([A-Za-z])apos\b", r"\1'"),
         (r"quot([A-Z])", r'"\1'), (r"([A-Za-z.,!?])quot\b", r'\1"')]

def lit(s):
    return "'" + s.replace("'", "''") + "'"

def title_sql(col):
    e = f"coalesce({col}, '')"
    for a, b in ENTITIES:
        e = f"replace({e}, {lit(a)}, {lit(b)})"
    for p, r in GLUED:
        e = f"regexp_replace({e}, {lit(p)}, {lit(r)}, 'g')"
    return f"lower({e})"

T21 = title_sql("a.title")
PUNCT = "[.,:;?!\"'\u2019\u201c\u201d]"
GLUE = r"[A-Za-z]apos(s|t|m|d|re|ve|ll)?\b|apos[A-Z]"
ENT = r"&(apos|quot|amp|#[0-9]+);"

con = duckdb.connect()
con.execute(f"ATTACH '{DATA}/metadata.db' AS meta (READ_ONLY)")
elig = pd.read_excel(f"{ROOT}/final_sample.xlsx", "Eligible")[["source"]].astype(str)
con.register("elig", elig)

# ---------- 1. validation against the 300 hand labels ----------
lab = pd.read_excel(f"{OUT}/political_validation.xlsx")
if len(lab) != 300 or not set(lab["political"].dropna().unique()) <= {0, 1} or lab["political"].isna().any():
    raise SystemExit("political_validation.xlsx must have 300 rows labelled 0 or 1. Stop here.")
con.register("lab", lab[["article_id", "political"]].astype({"article_id": str, "political": int}))
val = con.execute(f"""
    SELECT l.article_id, l.political, a.year, CAST(s.factuality AS VARCHAR) AS factuality,
           regexp_matches(lower(a.title), $v1) AS v1,
           regexp_matches(lower(a.title), $v2) AS v2,
           regexp_matches({T21}, $v21) AS v21
    FROM lab l JOIN meta.articles a USING (article_id) JOIN meta.sources s ON a.source = s.source
""", {"v1": v1, "v2": v2, "v21": v21}).df()
if len(val) != 300:
    raise SystemExit(f"Only {len(val)} of 300 labelled headlines joined to articles/sources. Stop here.")
GROUPS = {"Very High": "1 High/Very High", "High": "1 High/Very High", "Mostly Factual": "2 Mostly Factual",
          "Mixed": "3 Mixed", "Low": "4 Low/Very Low", "Very Low": "4 Low/Very Low"}
val["factuality_group"] = val["factuality"].map(GROUPS).fillna("5 other: " + val["factuality"].astype(str))

def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{p:.2f} [{c - h:.2f}-{c + h:.2f}]"

def scores(g, col):
    hit, pos = g[col].astype(bool), g["political"] == 1
    tp, fp, fn = int((hit & pos).sum()), int((hit & ~pos).sum()), int((~hit & pos).sum())
    return {"headlines": len(g), "labelled_political": tp + fn,
            "precision": wilson(tp, tp + fp) if tp + fp else "n/a",
            "recall": wilson(tp, tp + fn) if tp + fn else "n/a"}

say("1. KEYWORD FILTER vs YOUR 300 LABELS (95% intervals in brackets)")
say(pd.DataFrame({k: scores(val, k) for k in ["v1", "v2", "v21"]}).T.to_string())
say("\nBY OUTLET FACTUALITY (v1 and v2.1)")
say(pd.DataFrame([{"group": grp, "list": k, **scores(g, k)}
                  for grp, g in val.groupby("factuality_group") for k in ["v1", "v21"]]).to_string(index=False))
say("\nBY YEAR (v2.1)")
say(pd.DataFrame([{"year": y, **scores(g, "v21")} for y, g in val.groupby("year")]).to_string(index=False))

# ---------- 2. title damage by year (212 outlets) ----------
td = con.execute("""
    SELECT a.year, a.source, COUNT(*) AS titles,
           AVG(CAST(regexp_matches(a.title, $punct) AS INT)) AS has_punctuation,
           AVG(CAST(regexp_matches(a.title, $glue) AS INT)) AS glued_apos,
           AVG(CAST(regexp_matches(a.title, $ent) AS INT)) AS html_entity,
           MAX(length(a.title)) AS max_len,
           SUM(CAST(length(a.title) >= 99 AS INT)) AS at_99_plus
    FROM meta.articles a JOIN elig USING (source)
    WHERE a.title IS NOT NULL
    GROUP BY a.year, a.source
""", {"punct": PUNCT, "glue": GLUE, "ent": ENT}).df()
w = lambda c: (td[c] * td["titles"]).groupby(td["year"]).sum() / td.groupby("year")["titles"].sum()
yr = pd.DataFrame({"titles": td.groupby("year")["titles"].sum(),
                   "has_punctuation": w("has_punctuation").round(3),
                   "glued_apos": w("glued_apos").round(4),
                   "html_entity": w("html_entity").round(4),
                   "max_len": td.groupby("year")["max_len"].max(),
                   "share_99_plus_chars": (td.groupby("year")["at_99_plus"].sum() / td.groupby("year")["titles"].sum()).round(3),
                   "outlets": td.groupby("year")["source"].nunique(),
                   "outlets_punct_under_30pct": td[td["has_punctuation"] < 0.30].groupby("year")["source"].nunique()})
say("\n2. TITLE DAMAGE BY YEAR (212 outlets)")
say(yr.fillna(0).to_string())

# ---------- 3. effect of v2.1 on political counts and eligibility ----------
cnt = con.execute(f"""
    SELECT a.source, a.year, COUNT(*) AS n,
           COUNT(*) FILTER (WHERE regexp_matches(lower(a.title), $v1)) AS pol_v1,
           COUNT(*) FILTER (WHERE regexp_matches(lower(a.title), $v2)) AS pol_v2,
           COUNT(*) FILTER (WHERE regexp_matches({T21}, $v21)) AS pol_v21
    FROM meta.articles a JOIN meta.sources s ON a.source = s.source
    WHERE s.bias IS NOT NULL AND s.factuality IS NOT NULL
      AND coalesce(CAST(s.label AS VARCHAR), '') NOT IN ('Satire', 'Pro-Science')
    GROUP BY a.source, a.year
""", {"v1": v1, "v2": v2, "v21": v21}).df()
E = set(elig["source"])
y = cnt[cnt["source"].isin(E)].groupby("year")[["n", "pol_v1", "pol_v2", "pol_v21"]].sum()
y["v21_gain_over_v2_pct"] = (100 * (y["pol_v21"] / y["pol_v2"] - 1)).round(2)
say("\n3. POLITICAL ARTICLES BY YEAR (212 outlets)")
say(y.to_string())
s = cnt.groupby("source")[["n", "pol_v1", "pol_v2", "pol_v21"]].sum()
for k in ["v1", "v2", "v21"]:
    s[f"elig_{k}"] = (s[f"pol_{k}"] >= 200) & (s[f"pol_{k}"] / s["n"] >= 0.10)
say(f"eligible: v1 {int(s['elig_v1'].sum())} (same as Eligible sheet: {set(s.index[s['elig_v1']]) == E}) | "
    f"v2 {int(s['elig_v2'].sum())} | v2.1 {int(s['elig_v21'].sum())}")
meta = con.execute("""SELECT source, CAST(bias AS VARCHAR) AS bias, CAST(factuality AS VARCHAR) AS factuality, type
                      FROM meta.sources""").df().set_index("source")
new = s[s["elig_v21"] & ~s["elig_v1"]].join(meta)
new["share_v21"] = (new["pol_v21"] / new["n"]).round(3)
say(f"\nOUTLETS NEWLY ELIGIBLE UNDER v2.1: {len(new)}")
if len(new):
    say(new.sort_values("pol_v21", ascending=False)[["n", "pol_v21", "share_v21", "bias", "factuality", "type"]].to_string())
say(f"OUTLETS LOSING ELIGIBILITY UNDER v2.1: {int((s['elig_v1'] & ~s['elig_v21']).sum())} (expected 0)")

# ---------- 4. article text damage by year ----------
cd = con.execute(f"""
    SELECT CAST(left(article_id, 4) AS INTEGER) AS year, COUNT(*) AS articles,
           ROUND(AVG(length(content))) AS mean_chars,
           AVG(CAST(contains(content, '.') AS INT)) AS has_period,
           AVG(CAST(contains(content, ',') AS INT)) AS has_comma,
           AVG(CAST(contains(content, '!') AS INT)) AS has_exclamation,
           AVG(CAST(regexp_matches(content, $apos) AS INT)) AS has_apostrophe,
           AVG(CAST(regexp_matches(content, $dq) AS INT)) AS has_double_quote,
           1000.0 * SUM(length(content) - length(replace(content, '.', ''))) / nullif(SUM(length(content)), 0) AS periods_per_1000_chars
    FROM (SELECT article_id, source, coalesce(content, '') AS content FROM read_parquet('{PQ}')) p
    JOIN elig USING (source)
    GROUP BY 1 ORDER BY 1
""", {"apos": "['\u2019]", "dq": "[\"\u201c\u201d]"}).df()
say("\n4. ARTICLE TEXT BY YEAR (212 outlets)")
say(cd.round(3).to_string(index=False))

# ---------- 5. does the parquet hold a cleaner title? ----------
cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{PQ}')").df()["column_name"].tolist()
say("\n5. PARQUET COLUMNS:", cols)
if "title" in cols:
    share = con.execute(f"""SELECT AVG(CAST(regexp_matches(title, $punct) AS INT)) FROM read_parquet('{PQ}')
                            WHERE article_id LIKE '2018-%' AND title IS NOT NULL""", {"punct": PUNCT}).fetchone()[0]
    say(f"  2018 parquet titles with punctuation: {share:.3f}")
    ex = con.execute(f"""
        SELECT a.title AS metadata_title, p.title AS parquet_title
        FROM meta.articles a JOIN read_parquet('{PQ}') p USING (article_id)
        WHERE a.year = 2018 AND p.article_id LIKE '2018-%' AND NOT regexp_matches(a.title, $punct)
        ORDER BY hash(a.article_id) LIMIT 6
    """, {"punct": PUNCT}).df()
    for m, p in ex.itertuples(index=False):
        say(f"  metadata: {m!r}\n  parquet:  {p!r}")

report = "\n".join(L)
with open(f"{OUT}/02c_filter_validation.txt", "w", encoding="utf-8") as f:
    f.write(report)
print(report)
