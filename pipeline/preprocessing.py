"""Shared preprocessing, political filter and feature definitions.
Imported by test_preprocessing.py, sample_articles.py and extract_features.py. Change definitions here only."""
import html, math, os, re
import nltk
import pandas as pd
import textstat
from lexical_diversity import lex_div as ld

ROOT = os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis"))
DATA = f"{ROOT}/data/misinfo-general"
METADATA = f"{DATA}/metadata.db"
PARQUET = f"{DATA}/data/*.parquet"
ELIGIBLE_XLSX = f"{ROOT}/final_sample.xlsx"
LEXICON_PATH = f"{ROOT}/NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt"
PATTERN_PATH = f"{ROOT}/political_pattern_v21.txt"  # keyword list v2.1 (built and validated in 02c)

SEED = 42
MAX_PER_OUTLET = 300
MIN_POLITICAL = 200            # outlet eligibility: political articles
MIN_SHARE = 0.10               # outlet eligibility: political share
MIN_MATCHED = 50               # content-word lexicon matches needed for a stable VAD mean
MIN_SCORABLE_PER_OUTLET = 100  # outlets below this hold mostly short blurbs in the dataset
CAPS_MIN_LEN = 4               # 3 would keep GOP, FBI, CNN, BBC
CAPS_MAX_SHARE = 0.5           # a word written in capitals this often is an acronym, not emphasis


def eligible_sources():
    return pd.read_excel(ELIGIBLE_XLSX, "Eligible")["source"].dropna().astype(str).tolist()


# ---------- political filter (headlines, run in DuckDB) ----------
# Headlines are repaired before matching: HTML codes, and codes that lost their & and ; ("Californiaaposs").
TITLE_ENTITIES = [("&apos;", "'"), ("&#039;", "'"), ("&#39;", "'"), ("&#8217;", "'"), ("&#8216;", "'"),
                  ("&quot;", '"'), ("&#034;", '"'), ("&#8220;", '"'), ("&#8221;", '"'), ("&amp;", "&"),
                  ("&#8230;", "...")]
TITLE_GLUED = [(r"apos([A-Z])", r"'\1"), (r"([A-Za-z])apos(s|t|m|d|re|ve|ll)\b", r"\1'\2"),
               (r"([A-Za-z])apos\b", r"\1'"), (r"quot([A-Z])", r'"\1'), (r"([A-Za-z.,!?])quot\b", r'\1"')]


def _lit(s):
    return "'" + s.replace("'", "''") + "'"


def title_sql(col="title"):
    e = f"coalesce({col}, '')"
    for a, b in TITLE_ENTITIES:
        e = f"replace({e}, {_lit(a)}, {_lit(b)})"
    for p, r in TITLE_GLUED:
        e = f"regexp_replace({e}, {_lit(p)}, {_lit(r)}, 'g')"
    return f"lower({e})"


def political_sql(col="title"):
    """DuckDB boolean expression. Pass {"pattern": political_pattern()} as the query parameters."""
    return f"regexp_matches({title_sql(col)}, $pattern)"


def political_pattern():
    with open(PATTERN_PATH, encoding="utf-8") as f:
        return f.read().strip()


# ---------- article text ----------
# Dataset masking tokens: removed before tokenising, so they never enter a denominator
MASK_RE = re.compile(r"<copyright>|<twitter>|<url>|(?<!\w)<?selfref>?(?!\w)", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")
LINE_END = set(".!?:;,\"')]\u201d")  # a line ending in anything else (a heading, a list item) ends a sentence

# Attribution counts verbs only: quotation marks differ by house style (UK single quotes)
# and were stripped from most 2018 articles
ATTRIB_RE = re.compile(r"\b(?:according\s+to|said|says|told|reported|stated|wrote|added|noted|"
                       r"confirmed|announced|testified|acknowledged)\b", re.IGNORECASE)
HEDGE_RE = re.compile(r"\b(?:might|could|reportedly|allegedly|apparently|seemingly|suggests|"
                      r"appears\s+to|possibly|perhaps|likely|some\s+say)\b", re.IGNORECASE)
MAY_RE = re.compile(r"\bmay\b")  # lowercase only: excludes the month and Theresa May
CERT_RE = re.compile(r"\b(?:clearly|obviously|undeniable|undeniably|proves|proven|definitely|"
                     r"certainly|without\s+question|no\s+doubt|absolutely|indisputable)\b", re.IGNORECASE)
QUOTES = ('"', "\u201c", "\u201d")

try:
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))

# Words left out of the VAD means: grammar words, and the words that define the other features
# (otherwise hedging and attribution would be mechanically tied to arousal)
FUNCTION_WORDS = {"would", "could", "may", "might", "must", "shall", "cannot", "ought", "also", "going", "gonna",
                  "get", "gets", "got", "getting", "like", "us", "one", "two", "three", "four", "five", "six",
                  "seven", "eight", "nine", "ten", "hundred", "thousand", "million", "billion", "trillion",
                  "first", "second", "third", "last"}
FEATURE_WORDS = {"according", "said", "says", "told", "reported", "stated", "wrote", "added", "noted", "confirmed",
                 "announced", "testified", "acknowledged", "might", "could", "may", "reportedly", "allegedly",
                 "apparently", "seemingly", "suggests", "appears", "possibly", "perhaps", "likely", "clearly",
                 "obviously", "undeniable", "undeniably", "proves", "proven", "definitely", "certainly",
                 "absolutely", "indisputable"}
VAD_EXCLUDE = STOPWORDS | FUNCTION_WORDS | FEATURE_WORDS
# Site navigation and labels written in capitals ("READ MORE", "RELATED", "DONATE"): not emphasis
CAPS_BOILERPLATE = {"related", "read", "click", "donate", "trending", "subscribe", "advertisement", "share",
                    "follow", "newsletter", "news", "watch", "video", "update", "updated", "breaking",
                    "exclusive", "editor", "note", "photo", "photos", "image", "source"}

FEATURES = ["arousal", "valence", "dominance", "attribution_density", "hedging", "certainty",
            "caps_ratio", "exclamation_rate", "flesch_kincaid", "mtld", "log_length"]
DIAGNOSTICS = ["n_tokens", "n_matched", "n_proper_skipped", "n_masked", "n_sentences", "quote_pairs",
               "arousal_allwords"]


def load_vad():
    lex = pd.read_csv(LEXICON_PATH, sep="\t", keep_default_na=False, quoting=3)
    cols = ["term", "valence", "arousal", "dominance"]
    if list(lex.columns[:4]) != cols:
        raise ValueError(f"Unexpected lexicon columns: {list(lex.columns)}")
    lex["term"] = lex["term"].astype(str).str.strip().str.lower()
    lex = lex[~lex["term"].str.contains(r"\s")]
    return {t: (float(v), float(a), float(d)) for t, v, a, d in lex[cols].itertuples(index=False)}


def clean(text):
    text = html.unescape(text) if isinstance(text, str) else ""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text, n_masked = MASK_RE.subn(" ", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return " ".join(ln if ln[-1] in LINE_END else ln + "." for ln in lines if ln), n_masked


def _sentence_initial(t, start):
    i = start - 1
    while i >= 0 and t[i] in " \"'\u201c([*":
        i -= 1
    return i < 0 or t[i] in ".!?:;"


def tokens(t):
    """Tokens, lowercased tokens, and proper-noun flags.
    A capitalised word is a name when it appears capitalised mid-sentence somewhere in the article
    ("President Trump", "the House"): names are not emotion words, so the lexicon skips them."""
    mm = list(WORD_RE.finditer(t))
    toks = [m.group() for m in mm]
    low = [w.lower() for w in toks]
    title = [w[0].isupper() and not w.isupper() for w in toks]
    upper = [w.isupper() for w in toks]
    initial = [_sentence_initial(t, m.start()) for m in mm]
    names = {lw for lw, ti, ini in zip(low, title, initial) if ti and not ini}
    proper = [(ti and not ini) or ((ti or up) and lw in names) for lw, ti, up, ini in zip(low, title, upper, initial)]
    return toks, low, proper


def lexicon_words(low, proper, vad, content_only=True):
    return [w for w, p in zip(low, proper) if not p and w in vad and not (content_only and w in VAD_EXCLUDE)]


def caps_shares(texts, vad):
    """Share of each lexicon word's occurrences (4+ letters) written in all capitals, across a set of articles."""
    total, caps = {}, {}
    for text in texts:
        t, _ = clean(text)
        for w in WORD_RE.findall(t):
            if len(w) >= CAPS_MIN_LEN:
                lw = w.lower()
                if lw in vad:
                    total[lw] = total.get(lw, 0) + 1
                    if w.isupper():
                        caps[lw] = caps.get(lw, 0) + 1
    return {w: caps.get(w, 0) / n for w, n in total.items()}


def shouting_words(caps_shares, vad):
    """Words whose all-capitals form counts as emphasis: real words, normally written in lower case."""
    return {w for w, share in caps_shares.items()
            if share < CAPS_MAX_SHARE and w in vad and w not in VAD_EXCLUDE and w not in CAPS_BOILERPLATE}


def extract(text, vad, shout):
    t, n_masked = clean(text)
    toks, low, proper = tokens(t)
    n = len(toks)
    out = dict.fromkeys(FEATURES + DIAGNOSTICS, float("nan"))
    out.update(n_tokens=n, n_masked=n_masked, n_matched=0, n_proper_skipped=0)
    if n == 0:
        return out
    matched = [vad[w] for w in lexicon_words(low, proper, vad)]
    all_arousal = [vad[w][1] for w in lexicon_words(low, proper, vad, content_only=False)]
    out["n_matched"] = len(matched)
    out["n_proper_skipped"] = sum(p and w in vad for w, p in zip(low, proper))
    if matched:
        out["valence"], out["arousal"], out["dominance"] = (
            sum(m[i] for m in matched) / len(matched) for i in range(3))
    if all_arousal:
        out["arousal_allwords"] = sum(all_arousal) / len(all_arousal)
    n_sent = max(1, textstat.sentence_count(t))
    out["n_sentences"] = n_sent
    out["attribution_density"] = len(ATTRIB_RE.findall(t)) / n
    out["quote_pairs"] = sum(t.count(q) for q in QUOTES) / 2
    out["hedging"] = (len(HEDGE_RE.findall(t)) + len(MAY_RE.findall(t))) / n
    out["certainty"] = len(CERT_RE.findall(t)) / n
    out["caps_ratio"] = sum(w.isupper() and len(w) >= CAPS_MIN_LEN and lw in shout and not p
                            for w, lw, p in zip(toks, low, proper)) / n
    out["exclamation_rate"] = t.count("!") / n_sent
    out["flesch_kincaid"] = textstat.flesch_kincaid_grade(t)
    try:
        out["mtld"] = ld.mtld(low)
    except ZeroDivisionError:
        pass
    out["log_length"] = math.log(n)
    return out
