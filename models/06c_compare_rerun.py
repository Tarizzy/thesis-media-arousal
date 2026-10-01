"""06c_compare_rerun.py - the mixed models before and after the final topic variable: how far did anything move?
Read-only; compares models/coefficients.csv and variance_components.csv with backup_before_06b_rerun/."""
import os
import pandas as pd
os.chdir(os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "models"))
B = "../backup_before_06b_rerun"
o, n = pd.read_csv(f"{B}/coefficients.csv"), pd.read_csv("coefficients.csv")
m = o.merge(n, on=["model", "term"], suffixes=(" before", " after"))
m["change"] = m["Estimate after"] - m["Estimate before"]
show = m[m.term.isin(["fact_c", "part_z", "fact_c:part_z", "bias_ext"])]
print("KEY COEFFICIENTS (SDs of article arousal)")
print(show[["model", "term", "Estimate before", "Estimate after", "change", "Pr(>|t|) before", "Pr(>|t|) after"]]
      .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
print(f"\nlargest change in any coefficient of any model: {m.change.abs().max():.4f}")
vo, vn = pd.read_csv(f"{B}/variance_components.csv"), pd.read_csv("variance_components.csv")


def absorbed(v, grp):
    g = v[(v.grp == grp) & (v.var1 == "(Intercept)")].set_index("model").vcov
    return 1 - g["M2 + main effects"] / g["M1 + year"]


for grp, what in [("source", "outlet variance absorbed by factuality"), ("cluster", "cluster variance absorbed by framing")]:
    print(f"{what} (M1 -> M2): {absorbed(vo, grp):.1%} before, {absorbed(vn, grp):.1%} after")
