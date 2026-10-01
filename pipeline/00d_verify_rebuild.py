"""00d_verify_rebuild.py - after the country fix: confirms that rebuilding changed the country column and
nothing else. Compares against ~/Desktop/thesis/backup_before_country_fix/. Read-only."""
import os
import pandas as pd
os.chdir(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")))
B = "backup_before_country_fix"
load = lambda p: pd.read_parquet(p).sort_values("article_id").reset_index(drop=True)
old, new = load(f"{B}/analysis_table.parquet"), load("pipeline/analysis_table.parquet")
print(f"analysis table: {len(old):,} -> {len(new):,} rows, {old.source.nunique()} -> {new.source.nunique()} outlets")
cols = [c for c in old.columns if c != "country"]
try:
    pd.testing.assert_frame_equal(old[cols], new[cols], check_dtype=False)
    print("every column except country: IDENTICAL")
except AssertionError as e:
    print("DIFFERENCES OUTSIDE COUNTRY - stop and paste this:\n", str(e)[:600])
oc, nc = old.groupby("source").country.first(), new.groupby("source").country.first()
ch = pd.DataFrame({"before": oc.fillna("<missing>"), "after": nc.fillna("<missing>")})
print(f"country changed for {int((ch.before != ch.after).sum())} outlets:")
print(ch[ch.before != ch.after].to_string())
print(f"outlets still without a country: {int(nc.isna().sum())} | 'Venezuala' still present: {bool((nc == 'Venezuala').any())}")
fo = pd.read_csv(f"{B}/framing_model.csv").set_index("topic_cluster_model").framing
fn = pd.read_csv("partisanship/framing_model.csv").set_index("topic_cluster_model").framing
d = (fo - fn.loc[fo.index]).abs().max()
print(f"framing_model.csv (the topic variable in every model): max change {d:.1e} -> "
      f"{'UNCHANGED' if d < 1e-9 else 'CHANGED - stop and paste this'}")
mo = pd.read_csv(f"{B}/model_frame.csv").sort_values("article_id").reset_index(drop=True)
mn = pd.read_csv("models/model_frame.csv").sort_values("article_id").reset_index(drop=True)
mc = [c for c in mo.columns if c != "country"]
print(f"model_frame.csv outside country: {'IDENTICAL' if mo[mc].equals(mn[mc]) else 'DIFFERENT - stop and paste this'}")
