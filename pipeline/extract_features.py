"""Compute the 11 features for every sampled article, flag articles with too few lexicon matches,
and write the sanity check.
Usage: python extract_features.py                     (political sample)
       python extract_features.py sample_unfiltered   (robustness sample, later)"""
import os, sys, time
from collections import Counter
from multiprocessing import Pool
import duckdb, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocessing as pp

NAME = sys.argv[1] if len(sys.argv) > 1 else "sample"
OUT = f"{pp.ROOT}/pipeline"
VAD, SHOUT = None, None


def init(shout):
    global VAD, SHOUT
    VAD, SHOUT = pp.load_vad(), shout


def work(item):
    article_id, content = item
    return {"article_id": article_id, **pp.extract(content, VAD, SHOUT)}


def drivers(text, vad, highest, k=8):
    t, _ = pp.clean(text)
    _, low, proper = pp.tokens(t)
    c = Counter(pp.lexicon_words(low, proper, vad))
    ranked = sorted(c.items(), key=lambda wc: wc[1] * vad[wc[0]][1], reverse=highest)[:k]
    return ", ".join(f"{w} x{n} ({vad[w][1]:+.2f})" for w, n in ranked)


if __name__ == "__main__":
    t0 = time.time()
    con = duckdb.connect()
    vad = pp.load_vad()

    # capitals rule: shares come from the political sample, so both samples use the same word list
    political = con.execute(f"SELECT content FROM '{OUT}/sample.parquet'").df()["content"]
    shares = pp.caps_shares(political, vad)
    shout = pp.shouting_words(shares, vad)
    pd.DataFrame({"word": list(shares), "caps_share": list(shares.values())}) \
        .assign(counts_as_emphasis=lambda d: d["word"].isin(shout)).sort_values("word") \
        .to_csv(f"{OUT}/caps_shares.csv", index=False)
    del political

    s = con.execute(f"SELECT article_id, source, year, title, content FROM '{OUT}/{NAME}.parquet'").df()
    rows = []
    with Pool(initializer=init, initargs=(shout,)) as pool:
        for i, r in enumerate(pool.imap(work, zip(s["article_id"], s["content"]), chunksize=200), 1):
            rows.append(r)
            if i % 10000 == 0:
                print(f"{i:,} / {len(s):,} articles ({time.time() - t0:.0f}s)", flush=True)
    feats = s[["article_id", "source", "year"]].merge(pd.DataFrame(rows), on="article_id", how="left")
    feats["keep"] = feats["n_matched"] >= pp.MIN_MATCHED

    path, v1 = f"{OUT}/features_{NAME}.parquet", f"{OUT}/features_{NAME}_v1.parquet"
    if os.path.exists(path) and not os.path.exists(v1):
        os.rename(path, v1)  # keep the first run for the before/after comparison
    con.register("f_out", feats)
    con.execute(f"COPY (SELECT * FROM f_out ORDER BY source, year, article_id) TO '{path}' (FORMAT PARQUET)")
    feats.to_csv(f"{OUT}/features_{NAME}.csv", index=False)

    # ---------- sanity check ----------
    con.execute(f"ATTACH '{pp.METADATA}' AS meta (READ_ONLY)")
    fact = con.execute("SELECT source, CAST(factuality AS VARCHAR) AS factuality FROM meta.sources").df()
    f = feats.merge(fact, on="source", how="left")
    k = f[f["keep"]]
    L = [f"{NAME}: {len(f):,} articles | kept (>= {pp.MIN_MATCHED} content-word matches): {len(k):,} "
         f"({len(k) / len(f):.1%}) | run time {time.time() - t0:.0f}s"]

    if os.path.exists(v1):
        old = con.execute(f"SELECT article_id, arousal AS arousal_v1, keep AS keep_v1 FROM '{v1}'").df()
        cmp = f.merge(old, on="article_id")
        both = cmp[cmp["keep"] & cmp["keep_v1"]]
        L.append(f"\nCOMPARED WITH THE FIRST RUN: arousal correlation {both['arousal'].corr(both['arousal_v1']):.3f} "
                 f"(articles kept in both: {len(both):,}) | kept then, dropped now: {int((cmp['keep_v1'] & ~cmp['keep']).sum()):,}")

    L.append("\nDROPPED SHARE BY YEAR\n" + (1 - f.groupby("year")["keep"].mean()).round(3).to_string())
    L.append("\nDROPPED SHARE BY FACTUALITY\n" + (1 - f.groupby("factuality")["keep"].mean()).round(3).to_string())
    per = f.groupby("source").agg(articles=("keep", "size"), kept=("keep", "sum"))
    per["dropped_share"] = (1 - per["kept"] / per["articles"]).round(3)
    L.append("\n12 OUTLETS WITH THE HIGHEST DROPPED SHARE\n" + per.sort_values("dropped_share", ascending=False).head(12).to_string())
    low_outlets = per[per["kept"] < pp.MIN_SCORABLE_PER_OUTLET].sort_values("kept")
    L.append(f"outlets below {pp.MIN_SCORABLE_PER_OUTLET} kept articles: {len(low_outlets)} {low_outlets.index.tolist()}")
    L.append(f"outlets below 200 kept articles: {int((per['kept'] < 200).sum())}")
    L.append("\nn_tokens / n_matched / n_proper_skipped (all articles)\n" +
             f[["n_tokens", "n_matched", "n_proper_skipped"]].describe(percentiles=[.05, .25, .5, .75, .95]).round(0).T.to_string())
    L.append(f"lexicon hits skipped as names: {f['n_proper_skipped'].sum() / (f['n_proper_skipped'].sum() + f['n_matched'].sum()):.1%}")

    L.append("\nFEATURES (kept articles)\n" + k[pp.FEATURES + ["arousal_allwords", "quote_pairs"]]
             .describe(percentiles=[.01, .25, .5, .75, .99]).round(3).T.to_string())
    L.append("\nmissing values (kept): " + str({c: int(n) for c, n in k[pp.FEATURES].isna().sum().items() if n}))
    L.append(f"flesch_kincaid above 30: {int((k['flesch_kincaid'] > 30).sum())} articles")
    L.append(f"correlation arousal vs arousal_allwords: {k['arousal'].corr(k['arousal_allwords']):.3f}")

    counted, skipped, matches = Counter(), Counter(), Counter()
    for text in s.loc[s["article_id"].isin(k["article_id"]), "content"]:
        t, _ = pp.clean(text)
        toks, low, proper = pp.tokens(t)
        for w, lw, p in zip(toks, low, proper):
            if w.isupper() and len(w) >= pp.CAPS_MIN_LEN:
                (counted if lw in shout and not p else skipped)[w] += 1
        matches.update(pp.lexicon_words(low, proper, vad))
    total = sum(matches.values())
    L.append("\n25 MOST COMMON CAPITALISED WORDS COUNTED AS EMPHASIS\n  " + ", ".join(f"{w} {n:,}" for w, n in counted.most_common(25)))
    L.append("\n25 MOST COMMON CAPITALISED WORDS NOT COUNTED (acronyms, names, site labels)\n  " + ", ".join(f"{w} {n:,}" for w, n in skipped.most_common(25)))
    L.append("\n40 MOST FREQUENT LEXICON MATCHES (word, share of all matches, arousal score)")
    L.extend(f"  {w:<14} {n / total:.2%}  {vad[w][1]:+.3f}" for w, n in matches.most_common(40))

    text_by_id = s.set_index("article_id")
    for label, highest in [("HIGHEST", True), ("LOWEST", False)]:
        L.append(f"\n10 {label}-AROUSAL ARTICLES (kept)")
        top = k.sort_values("arousal", ascending=not highest).head(10)
        for r in top.itertuples():
            row = text_by_id.loc[r.article_id]
            body, _ = pp.clean(row["content"])
            L.append(f"\n  arousal {r.arousal:+.3f} | {r.source} | {r.year} | {r.factuality} | tokens {r.n_tokens} | matched {r.n_matched}")
            L.append(f"  TITLE: {row['title']}")
            L.append(f"  DRIVING WORDS: {drivers(row['content'], vad, highest)}")
            L.append(f"  TEXT: {body[:300]}")

    report = "\n".join(L)
    with open(f"{OUT}/sanity_{NAME}.txt", "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
