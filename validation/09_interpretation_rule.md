# How the human validation will be read - fixed 27 September 2026
Written after the first coder's (Tarik's) ratings were analysed - Spearman 0.109 with lexicon arousal
over 120 items, 0 topics rated - and BEFORE the second coder's ratings were received or seen.

1. Agreement between the two coders (quadratic-weighted kappa, 120 articles):
   >= 0.60 reliable | 0.40-0.59 moderately reliable, usable with that qualifier |
   < 0.40 unreliable: the validation is inconclusive whatever the correlation in 2.
   Bands follow Landis & Koch (1977).
2. Spearman correlation of the two-coder mean rating with lexicon arousal (120 articles),
   read only if agreement is at least 0.40:
   >= 0.30 the lexicon tracks perceived emotive wording to a moderate degree | < 0.30 it does not.
   0.30 is the conventional threshold for a moderate correlation (Cohen 1988).
3. If the second coder's ratings do not arrive, the validation is reported as inconclusive,
   using the first coder's figures.
4. Topic contestedness (second coder only; the first coder has seen the framing ranking and is not
   blind to it): a positive correlation with framing divergence at p < 0.05 supports reading framing
   divergence as a measure of partisan contestation; otherwise it does not.

## Addendum 28 September 2026 - three coders returned
Article ratings came back from three coders: the author (returned_tarik.xlsx, 25 Sep) and two political
scientists (returned_glo.xlsx, returned_taner.xlsx). The rule above was written for two coders. It is applied
to all three: agreement = the mean of the three pairwise quadratic-weighted kappas (each pair also reported);
rho = Spearman of the three-coder mean rating with lexicon arousal. This definition was fixed in
09d_three_coders.py before any correlation with lexicon arousal was computed; the pairwise agreements had
already been inspected (every pair is at least 0.40, so the band is the same under any definition). Every
subset of coders is reported in 09d_combinations.csv; the branch is read from the three-coder result only.
Topics use the two blind coders (coder2, taner); the author's topic ratings were not used, as point 4 requires.
File note: the sheet sent as 'FRTarik_09_rating_sheet.xlsx' (kept as unused_FRTarik_sheet.xlsx) carried
article ratings identical to taner's on all 120 items, plus the author's topic ratings. It was not used.
The section headed EXPLORATORY in 09d_report.txt was not pre-specified.
