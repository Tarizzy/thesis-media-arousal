"""00e_verify_final.py - after the country fix and the 05e random-stream fix: checks that every change is one
we expect and nothing else moved, against backup_before_country_fix/ (the files every model so far was fitted
on). Read-only. Exit code 0 when everything is as expected."""
import os, sys
import pandas as pd
os.chdir(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")))
B = "backup_before_country_fix"
ok = True


def check(cond, good, bad):
    global ok
    print(("   OK    " if cond else "   STOP  ") + (good if cond else bad))
    ok = ok and bool(cond)


load = lambda p: pd.read_parquet(p).sort_values("article_id").reset_index(drop=True)
old, new = load(f"{B}/analysis_table.parquet"), load("pipeline/analysis_table.parquet")
cols = [c for c in old.columns if c != "country"]
try:
    pd.testing.assert_frame_equal(old[cols], new[cols], check_dtype=False)
    same = True
except AssertionError:
    same = False
check(same, f"analysis table: {len(new):,} rows, every column except country identical",
      "analysis table changed outside the country column")
nc = new.groupby("source").country.first()
check(nc.isna().sum() == 0, "every outlet has a country",
      f"{int(nc.isna().sum())} outlet(s) still without a country: {list(nc[nc.isna()].index)}")

co = pd.read_csv(f"{B}/cluster_framing.csv", index_col=0)
cn = pd.read_csv("partisanship/cluster_framing.csv", index_col=0)
prim = ["framing", "raw", "ci_lo", "ci_hi", "half_a", "half_b"]
d = max(float((co[c] - cn[c]).abs().max()) for c in prim)
check(d == 0, "05e primary: scores, CIs and split halves for all 33 clusters IDENTICAL to the version the models used",
      f"05e primary changed (max difference {d:.1e})")
for v in ["framing_us_only", "framing_mixed_plus", "framing_no_emotion_words"]:
    if v in co and v in cn:
        print(f"   info  {v} recomputed: largest change {float((co[v] - cn[v]).abs().max()):.4f}")

fo = pd.read_csv(f"{B}/framing_model.csv").set_index("topic_cluster_model")
fn = pd.read_csv("partisanship/framing_model.csv").set_index("topic_cluster_model")
kept = [i for i in fo.index if i != -1]
check(float((fo.loc[kept, "framing"] - fn.loc[kept, "framing"]).abs().max()) == 0,
      "framing_model.csv: the 31 named clusters identical", "framing_model.csv: named clusters changed")
print(f"   info  Other (clusters 14 + 26): {fo.framing[-1]:.5f} -> {fn.framing[-1]:.5f}; standardised values "
      f"shift by at most {float((fo.framing_z - fn.framing_z).abs().max()):.3f} SD")

mo = pd.read_csv(f"{B}/model_frame.csv").sort_values("article_id").reset_index(drop=True)
mn = pd.read_csv("models/model_frame.csv").sort_values("article_id").reset_index(drop=True)
mc = [c for c in mo.columns if c not in ("country", "part_z")]
check(mo[mc].equals(mn[mc]), "model_frame.csv identical apart from country and the topic variable",
      "model_frame.csv changed in other columns")
dz = (mo.part_z - mn.part_z).abs()
print(f"   info  topic variable in the model: {int((mn.topic_cluster_model == -1).sum()):,} articles in Other move by "
      f"{float(dz[mn.topic_cluster_model == -1].max()):.3f} SD; all others by at most "
      f"{float(dz[mn.topic_cluster_model != -1].max()):.3f} SD")
print("\nVERDICT: " + ("every change is an expected one - the 06b refit can go ahead" if ok else "STOP and paste this"))
sys.exit(0 if ok else 1)
