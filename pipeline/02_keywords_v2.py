import os, re
import duckdb, pandas as pd

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
DATA = f"{ROOT}/data/misinfo-general"
OUT = f"{ROOT}/pipeline"

# v1: recover the exact terms behind political_pattern.txt
v1 = open(f"{ROOT}/political_pattern.txt", encoding="utf-8").read().strip()
if not (v1.startswith(r"\b(?:") and v1.endswith(r")\b")):
    raise SystemExit("political_pattern.txt has an unexpected format. Stop here.")
body = v1[len(r"\b(?:"):-len(r")\b")]
v1_terms = [re.sub(r"\\(.)", r"\1", t) for t in body.split("|")]
if "|".join(re.escape(t) for t in v1_terms) != body:
    raise SystemExit("Could not recover the v1 terms exactly. Stop here.")

ADD = {
    # 1. word forms v1 could never match (every v1 term needs a word boundary right after it)
    "congressional", "parliaments", "parliamentary", "presidents", "presidential", "presidency",
    "senators", "congressmen", "congresswomen", "governors", "mayors", "democrats", "democratic",
    "republicans", "liberals", "conservatives", "lawmakers", "politicians", "politics", "ministers",
    "diplomats", "diplomatic", "ambassadors", "elections", "electoral", "votes", "voted", "voter",
    "voters", "ballots", "midterms", "referendums", "legislative", "legislature", "laws",
    "regulations", "executive orders", "impeachment", "impeached", "vetoes", "vetoed", "amendments",
    "constitutional", "borders", "tariffs", "sanction", "sanctioned", "abortions", "taxes",
    "budgets", "deficits", "treaties", "airstrikes", "annexation", "annexed", "scandals",
    "investigations", "indictments", "indicted", "probes", "hearings", "protests", "protesters",
    "protestors", "riots", "rioters",
    # 2. core political vocabulary missing from v1
    "government", "governments", "democracy", "mp", "mps", "mep", "meps", "sen.", "gov.",
    "capitol hill", "whitehall", "holyrood", "stormont", "dnc", "rnc", "scotus", "potus",
    "immigrant", "immigrants", "migrant", "migrants", "asylum", "refugee", "refugees",
    # 3. actors and events of 2017-2022
    "pence", "schumer", "mueller", "comey", "kavanaugh", "pompeo", "tillerson", "giuliani",
    "manafort", "kushner", "bernie sanders", "elizabeth warren", "hillary", "clinton",
    "ocasio-cortez", "desantis", "cuomo", "newsom", "cheney", "kamala", "buttigieg", "romney",
    "ted cruz", "lindsey graham", "kevin mccarthy", "january 6", "january 6th", "jan. 6", "jan 6",
    "capitol riot", "capitol attack", "capitol siege", "insurrection", "roe v. wade", "roe v wade",
    "theresa may", "boris johnson", "boris", "corbyn", "farage", "sturgeon", "raab", "gove",
    "truss", "javid", "priti patel", "dominic cummings", "lib dem", "lib dems", "snp", "ukip",
    "bolsonaro", "maduro", "kim jong un", "kim jong-un", "duterte", "orban", "orbán", "lukashenko",
    "assad", "khamenei", "rouhani", "shinzo abe", "scott morrison", "ardern", "zelenskyy",
    "zelenskiy", "navalny", "taliban", "hamas",
}

def term_re(t):
    left = r"\b" if re.match(r"\w", t[0], re.ASCII) else ""
    right = r"\b" if re.match(r"\w", t[-1], re.ASCII) else ""
    return left + re.escape(t) + right

v1_set = set(v1_terms)
terms_v2 = sorted(v1_set | ADD, key=lambda t: (-len(t), t))
added = [t for t in terms_v2 if t not in v1_set]
v2 = "(?:" + "|".join(term_re(t) for t in terms_v2) + ")"
with open(f"{ROOT}/political_pattern_v2.txt", "w", encoding="utf-8") as f:
    f.write(v2)
with open(f"{ROOT}/political_terms_v2.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(f"{t}\t{'v1' if t in v1_set else 'added in v2'}" for t in terms_v2))

con = duckdb.connect()
con.execute(f"ATTACH '{DATA}/metadata.db' AS meta (READ_ONLY)")
elig = pd.read_excel(f"{ROOT}/final_sample.xlsx", "Eligible")
E = elig["source"].astype(str).tolist()

counts = con.execute("""
    WITH t AS (
        SELECT a.source, a.year,
               regexp_matches(lower(a.title), $v1) AS m1,
               regexp_matches(lower(a.title), $v2) AS m2
        FROM meta.articles a JOIN meta.sources s ON a.source = s.source
        WHERE s.bias IS NOT NULL AND s.factuality IS NOT NULL
          AND coalesce(CAST(s.label AS VARCHAR), '') NOT IN ('Satire', 'Pro-Science')
    )
    SELECT source, year, COUNT(*) AS n,
           COUNT(*) FILTER (WHERE m1) AS pol_v1,
           COUNT(*) FILTER (WHERE m2) AS pol_v2,
           COUNT(*) FILTER (WHERE m1 AND NOT m2) AS only_v1
    FROM t GROUP BY source, year
""", {"v1": v1, "v2": v2}).df()

s = counts.groupby("source")[["n", "pol_v1", "pol_v2", "only_v1"]].sum()
for v in ["v1", "v2"]:
    s[f"share_{v}"] = s[f"pol_{v}"] / s["n"]
    s[f"elig_{v}"] = (s[f"pol_{v}"] >= 200) & (s[f"share_{v}"] >= 0.10)
S = s.reindex(E)
wb = elig.set_index("source")["n_political"].reindex(E)

L = [f"TERMS: v1 {len(v1_terms)} | v2 {len(terms_v2)} | added {len(added)}"]
L.append("\nCHECK 1: v1 in DuckDB reproduces the workbook")
L.append(f"  candidate outlets: {len(s)} (expected 374)")
L.append(f"  eligible under v1: {int(s['elig_v1'].sum())} (expected 212); "
         f"same outlets as Eligible sheet: {set(s.index[s['elig_v1']]) == set(E)}")
L.append(f"  n_political mismatches vs workbook: {int((S['pol_v1'].values != wb.values).sum())} (expected 0)")
L.append(f"\nCHECK 2: titles matched by v1 but not v2: {int(s['only_v1'].sum())} (expected 0)")

inc = S["pol_v2"] / S["pol_v1"] - 1
L.append("\nIMPACT ON THE 212 OUTLETS")
L.append(f"  political articles: v1 {int(S['pol_v1'].sum()):,} -> v2 {int(S['pol_v2'].sum()):,} "
         f"({100 * (S['pol_v2'].sum() / S['pol_v1'].sum() - 1):+.1f}%)")
L.append(f"  still eligible under v2: {int(S['elig_v2'].sum())} of {len(S)}")
L.append(f"  per-outlet increase: median {100 * inc.median():+.0f}%, "
         f"min {100 * inc.min():+.0f}%, max {100 * inc.max():+.0f}%")
L.append("  largest increases: " + ", ".join(f"{k} {100 * v:+.0f}%" for k, v in inc.sort_values(ascending=False).head(8).items()))
L.append("  smallest increases: " + ", ".join(f"{k} {100 * v:+.0f}%" for k, v in inc.sort_values().head(8).items()))

Y = counts[counts["source"].isin(E)].groupby("year")[["n", "pol_v1", "pol_v2"]].sum()
Y["share_v1"] = (Y["pol_v1"] / Y["n"]).round(3)
Y["share_v2"] = (Y["pol_v2"] / Y["n"]).round(3)
L.append("\nPOLITICAL SHARE BY YEAR (212 outlets)\n" + Y.to_string())

meta = con.execute("""SELECT source, CAST(bias AS VARCHAR) AS bias,
                      CAST(factuality AS VARCHAR) AS factuality, type FROM meta.sources""").df().set_index("source")
new = s[s["elig_v2"] & ~s["elig_v1"]].join(meta).sort_values("pol_v2", ascending=False)
L.append(f"\nOUTLETS THAT WOULD NEWLY QUALIFY UNDER v2: {len(new)}")
if len(new):
    L.append(new[["n", "pol_v1", "pol_v2", "share_v1", "share_v2", "bias", "factuality", "type"]].round(3).to_string())
L.append(f"OUTLETS THAT WOULD STOP QUALIFYING: {int((s['elig_v1'] & ~s['elig_v2']).sum())} (expected 0)")

only = con.execute("""
    SELECT title FROM meta.articles
    WHERE list_contains($E, source)
      AND regexp_matches(lower(title), $v2) AND NOT regexp_matches(lower(title), $v1)
    ORDER BY hash(article_id || '-42')
    LIMIT 20000
""", {"E": E, "v1": v1, "v2": v2}).df()
low = only["title"].str.lower()
hits = pd.Series({t: int(low.str.contains(term_re(t), regex=True).sum()) for t in added}).sort_values(ascending=False)
L.append(f"\nADDED TERMS BEHIND THE NEW MATCHES ({len(only):,} random titles matched only by v2)")
L.append(hits.head(30).to_string())
L.append("\n40 RANDOM TITLES MATCHED ONLY BY v2")
L.extend("  " + t for t in only["title"].head(40))

vpath = f"{OUT}/political_validation.xlsx"
if os.path.exists(vpath):
    L.append("\nVALIDATION SAMPLE: political_validation.xlsx already exists, left untouched")
else:
    val = con.execute("""
        SELECT article_id, title FROM (
            SELECT article_id, title,
                   row_number() OVER (PARTITION BY year ORDER BY hash(article_id || '-42')) AS r
            FROM meta.articles
            WHERE list_contains($E, source) AND title IS NOT NULL AND trim(title) <> ''
        ) WHERE r <= 50
        ORDER BY hash(article_id || '-7')
    """, {"E": E}).df()
    val["political"] = ""
    val.to_excel(vpath, index=False)
    L.append(f"\nVALIDATION SAMPLE: {len(val)} titles written to {vpath} (expected 300)")

report = "\n".join(L)
with open(f"{OUT}/02_keywords_v2.txt", "w", encoding="utf-8") as f:
    f.write(report)
print(report)
