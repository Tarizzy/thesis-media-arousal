import os, sys
from importlib.metadata import version
import duckdb, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocessing as pp

print("VERSIONS", {p: version(p) for p in ["textstat", "lexical-diversity", "nltk", "duckdb", "pandas", "numpy"]})
results = []

def check(name, got, expected):
    ok = abs(got - expected) < 1e-9 if isinstance(expected, float) else got == expected
    results.append(ok)
    print(f"  {name}: got {got!r}, expected {expected!r} -> {'PASS' if ok else 'FAIL'}")

vad = pp.load_vad()
print(f"\nLEXICON: {len(vad):,} unigrams | excluded from VAD means: {len(pp.VAD_EXCLUDE)} words")

print("\nFEATURES (synthetic article)")
demo = ('Theresa May said on 5 May that talks may collapse. "This is a DISASTER," the FBI and GOP '
        'told NATO. It clearly proves nothing! <copyright> selfref reported <url> it.')
f = pp.extract(demo, vad, {"disaster"})
for k, e in dict(n_tokens=25, n_masked=3, n_sentences=3, attribution_density=3 / 25, quote_pairs=1.0,
                 hedging=1 / 25, certainty=2 / 25, caps_ratio=1 / 25, exclamation_rate=1 / 3).items():
    check(k, f[k], e)

print("\nNAMES ARE SKIPPED BY THE LEXICON")
toks, low, proper = pp.tokens("President Trump said the House may vote. Trump is angry. The house burned down.")
check("names flagged", [w for w, p in zip(toks, proper) if p], ["Trump", "House", "Trump"])
toks, low, proper = pp.tokens("Donald Trump spoke. TRUMP WON and NATO said DEAD wrong.")
check("capitals of a name flagged", [w for w, p in zip(toks, proper) if p], ["Trump", "TRUMP"])
check("caps_ratio counts DEAD only", pp.extract("Donald Trump spoke. TRUMP WON and NATO said DEAD wrong.", vad, {"dead", "trump"})["caps_ratio"], 1 / 10)
for w in ["says", "told", "could", "may", "would", "going", "one"]:
    check(f"'{w}' excluded from VAD means", w in pp.VAD_EXCLUDE, True)

print("\nLINE BREAKS END SENTENCES")
check("cleaned", pp.clean("First line without period\nSecond line.\n\n<copyright>\nThird")[0],
      "First line without period. Second line. Third.")

print("\nHTML CODES IN ARTICLE TEXT (Fusion)")
ent = "He said &#8220;no&#8221; and it&#8217;s &amp; fine!"
check("cleaned", pp.clean(ent)[0], "He said \u201cno\u201d and it's & fine!")
g = pp.extract(ent, vad, set())
check("n_tokens", g["n_tokens"], 6)
check("quote_pairs", g["quote_pairs"], 1.0)
check("attribution_density", g["attribution_density"], 1 / 6)

con = duckdb.connect()
P = {"pattern": pp.political_pattern()}
print("\nHEADLINE REPAIR + KEYWORD LIST v2.1")
tt = pd.DataFrame({"title": [
    "The 45 most bizarre lines from Donald Trumps wild Montana speech",
    "quotThis is a DISASTERquot says Trumpaposs aide",
    "Jeremy Corbyn aposNo Iaposm not an anti-Semite",
    "Apostle Paul and the quota system, Veuve Clicquot, Buffalo Bills win",
    "Another nationwide mushroom recall after state testing finds listeria"],
    "expected": [True, True, True, False, False]})
con.register("tt", tt)
got = con.execute(f"SELECT title, {pp.political_sql()} AS m FROM tt", P).df()
for (t, m), e in zip(got.itertuples(index=False), tt["expected"]):
    check(t[:45], bool(m), bool(e))

print("\nFILTER REPRODUCES 02c (political articles per year, 212 outlets; takes a few minutes)")
con.execute(f"ATTACH '{pp.METADATA}' AS meta (READ_ONLY)")
con.register("elig", pd.DataFrame({"source": pp.eligible_sources()}))
y = con.execute(f"""SELECT year, COUNT(*) FILTER (WHERE {pp.political_sql()}) AS pol
                    FROM meta.articles JOIN elig USING (source) GROUP BY year ORDER BY year""", P).df()
EXPECTED = {2017: 61534, 2018: 178076, 2019: 217852, 2020: 357281, 2021: 341906, 2022: 326582}
for yr, pol in y.itertuples(index=False):
    check(str(yr), int(pol), EXPECTED.get(int(yr), -1))

print("\nALL CHECKS PASS" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")
