"""05e_patch_rng.py - one-off edit to 05e_framing.py. Gives the three variants and 'Other' (clusters 14 + 26)
their own random streams, so none of them depends on what was computed before it. The primary loop is not
touched, so the 33 primary scores, their CIs and the split-half reliability stay exactly as they are.
Keeps the unedited file as 05e_framing_before_rng_fix.py."""
import os, shutil, sys
P = os.path.join(os.environ.get("THESIS_ROOT", os.path.expanduser("~/Desktop/thesis")), "partisanship") if len(sys.argv) < 2 else sys.argv[1]
f = f"{P}/05e_framing.py"
s = open(f).read()
if "own random stream" in s:
    sys.exit("05e_framing.py is already patched - nothing done")
loop = ('for name, mask in [("us_only", is_us), ("mixed_plus", mixed_plus), ("no_emotion_words", None)]:\n'
        '    vals = {}\n')
other = "other = one_score(prep(Xo, rows), li, ri)[0]"
if s.count(loop) != 1 or s.count(other) != 1:
    sys.exit(f"expected lines not found exactly once (variants: {s.count(loop)}, other: {s.count(other)}) - nothing changed")
shutil.copy(f, f"{P}/05e_framing_before_rng_fix.py")
s = s.replace(loop, loop + '    rng = np.random.default_rng([pp.SEED, {"us_only": 1, "mixed_plus": 2, "no_emotion_words": 3}[name]])'
                           '   # own random stream per variant\n')
s = s.replace(other, "rng = np.random.default_rng([pp.SEED, 1426])   # own random stream for Other (clusters 14 + 26)\n"
              + other)
open(f, "w").write(s)
print("patched 05e_framing.py (backup: 05e_framing_before_rng_fix.py)")
