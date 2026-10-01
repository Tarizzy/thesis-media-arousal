# Day 5 - topic partisanship: method and expectations, fixed before running 05b

Saved before any delta was computed. Do not edit after 05b has run; record changes as dated additions.

## Measure
delta_c = coverage asymmetry of cluster c: the log-odds (Monroe, Colaresi & Quinn 2008, informative
Dirichlet prior) of c's share of right-leaning versus left-leaning outlets' political articles.
delta_c > 0 means the right over-covers c. The model uses |delta_c|. z_c is reported as a descriptive only,
because it grows with cluster size.

## Fixed choices
- Articles: every political article (v2.1 headline filter) in the full corpus from the 211 analysis outlets,
  counted by 05a. All 33 clusters get a delta; clusters 14 and 26 are also merged as Other for the model.
- Sides (MBFC bias): Left = Extreme Left 4 + Left 35 + Left-Center 43 = 82 outlets.
  Right = Right-Center 21 + Right 54 + Extreme Right 43 = 118 outlets. Least Biased (11) is on neither side.
- Outlet weighting: each outlet's shares across clusters sum to 1; a side's share is the mean of its
  outlets' shares; pseudo-counts = side share x the side's total political articles.
- Prior: alpha_c = alpha_0 x pi_c, with pi the equal-weight mean share over all 211 outlets and
  alpha_0 = 1,000. Sensitivity: alpha_0 = 100 and 10,000.
- Uncertainty: 1,000 bootstrap resamples of outlets within each side (82 and 118 kept fixed; Least Biased
  resampled within its own stratum for the prior), prior applied inside every resample, 95% percentile
  intervals, share of resamples with no articles on one side logged per cluster. Nothing skipped or
  imputed. Sensitivity: resampling within side x country (US / non-US).
- Split-half reliability: 200 random splits of outlets within side, Spearman rho between the halves'
  deltas across the 33 clusters, Spearman-Brown corrected.
- Variants saved for Day 10: US outlets only; outlets rated Mixed or better only; all articles
  instead of political articles.

## Decision rules
1. US-only versus all 211: if Spearman rho between the two delta vectors is at least 0.9, all 211 stay
   primary. Otherwise the US-only delta becomes primary and all 211 moves to robustness.
2. The sanity check passes if both hold:
   (a) at least 3 of the 5 contested clusters - 24 abortion & rights, 25 election administration,
       29 race & extremism, 28 health & COVID, 0 Latin America & border - rank in the top half
       (1-16 of 33) by |delta_c|;
   (b) the mean |delta_c| of those 5 exceeds the mean |delta_c| of the 8 non-political clusters
       (2, 3, 4, 10, 13, 14, 26, 32).
3. Split-half reliability (Spearman-Brown) below 0.7 counts as a failure of the measure.
4. If rule 2 or 3 fails, or the CES validation gives rho <= 0, the declared fallback is framing
   divergence: word-level log-odds between left and right within each cluster.

## Expectations (recorded to compare against; not pass/fail)
- Immigration is not one cluster: it is split between 0 (border patrol, Latin America) and 29
  (family separation, border wall).
- Right lean (delta > 0): 25 election administration (2020 fraud claims, Dominion), 24 abortion & rights
  (pro-life and transgender coverage), 28 health & COVID (vaccine scepticism, ivermectin),
  29 race & extremism (antifa, "tyranny"), 22 partisan media (media criticism), 0 border.
- Left lean (delta < 0): 9 environment & energy, 18 economy & welfare.
- Uncertain direction: 21 US investigations (both sides cover it heavily with opposite storylines, so
  it may sit near zero although contested - a known blind spot of coverage asymmetry); foreign affairs
  (5, 6, 20, 27), where left-center newspapers with foreign desks pull left and the 6 Russian outlets,
  all on the right, pull right. The US-only version isolates the second effect.
- Small |delta| relative to the contested five: 7 transport, 23 space & nature, 31 technology,
  11 business & markets.
- 8 US Congress: the second-highest political share but plausibly small |delta|, since both sides
  cover it. Political share and |delta| are different axes; the figure plots one against the other.
- Non-political clusters hold few political articles, so their intervals will be wide. 2 US pro sports
  may lean right (anthem protests): a culture-war effect, not a clustering failure.

## CES validation (capped at half a day)
Source: CCES Cumulative Policy Preferences, Harvard Dataverse doi:10.7910/DVN/OSXDQO, waves 2017-2022.
Gap = |weighted % support among Democrats - among Republicans| per item, correlated with |delta_c|
(Spearman). Mapping fixed now, items confirmed against the file's variable list:
abortion and same-sex marriage -> 24; immigration and border items -> 0; guns -> 15;
environment / EPA -> 9; health care (ACA) and taxes -> 18; military force abroad -> 27.
Several items on one cluster are averaged, so each cluster is one point.

## Clarification added 19 Sep 2026, before 05b was run
Rule 3 applies to |delta_c|, the model's predictor; the signed delta's split-half reliability is reported
alongside. Rule 1 compares the signed delta vectors, as written. The one outlet whose country could not be
confirmed (N/A) is left out of the US-only variant.

## Amendment added 19 Sep 2026, after 05b failed rule 3, before 05c was run
The equal-weight measure failed rule 3 (Spearman-Brown |delta| 0.24, signed 0.50). Before applying the
rule 4 fallback, a fixed set of seven outlet-weighting schemes (equal; sqrt(n); capped at the 90th
percentile; pooled counts; 10% trimmed mean; outlets with at least 1,000 political articles; mean of
per-outlet log-odds) is tested on split-half reliability alone. The outcome variable plays no part in the
choice and every scheme is reported. If the most reliable scheme reaches Spearman-Brown |delta| >= 0.70 it
becomes the primary measure and 05b is re-run with it. If none reaches 0.70, rule 4 applies and the
framing-divergence measure takes over.

## Note added 20 Sep 2026, after 05d, before 05e
05d sampled 265,501 political articles (median 57 per outlet x cluster cell, 49% at the cap of 60) and kept
73.1M tokens over a 10,000-word vocabulary. On that basis an outlet must have at least 2,000 tokens in a
cluster to contribute to that cluster's score, about seven articles; the 500-token version is kept as a
sensitivity check. The threshold was fixed before any divergence was computed.

## Amendment added 20 Sep 2026, before 05e was run
The coverage-asymmetry measure failed rule 3 under every weighting scheme (best Spearman-Brown 0.28) and
side explained a median 0.7% of the between-outlet variation in attention, so rule 4 applies and framing
divergence becomes the measure: word-level log-odds between left and right inside each cluster, outlets
weighted equally, with a permutation baseline (side labels shuffled among outlets) subtracted so clusters
of different size are comparable. Rule 3 is judged on the correlation between split halves of the scores
themselves rather than of their ranks, because the model uses the value and rank correlation is dominated
by noise whenever most clusters sit near zero; the rank version is reported alongside. All other rules
stand. The variant excluding NRC-VAD lexicon words is reported: if it correlates at least 0.9 with the
primary, the concern that framing divergence and arousal share vocabulary is answered empirically;
if not, the excluded-words version becomes primary.

## Note added 20 Sep 2026, after the first 05e run
05e returned Spearman-Brown 0.69 on values against the 0.70 threshold, with rule 2 passed and the
no-emotion-words variant correlating 0.899 with the primary. Each half-sample score subtracts a permutation
baseline estimated from only 5 shuffles, so the subtraction adds estimation noise to the half-scores and
depresses measured reliability. The baseline is now estimated from 40 shuffles in the full estimate and 20
in each half: this changes the precision of the same estimator, not its definition. Decision fixed before
the rerun: Spearman-Brown >= 0.70 makes framing divergence the topic variable; below 0.70 hands over to
hand-coded divisiveness, with framing divergence reported as a supporting measure.

## Measure settled, 20 Sep 2026
With the baseline estimated from 40 shuffles (20 per half), 05e returns Spearman-Brown 0.72 on values,
passing rule 3, with rule 2 passed (4 of 5 contested clusters in the top half; contested mean 0.1098 against
0.0707 for the non-political clusters). Variants against the primary: US outlets only 0.927, no NRC-VAD
words 0.935, Mixed-or-better outlets 0.705. Framing divergence is the topic-partisanship variable for the
model, taken from partisanship/framing_model.csv. Limitations to carry into the write-up: reliability sits
just above the threshold; the Mixed-or-better variant is the weakest agreement; non-political clusters sit
mid-table (US pro sports 0.1015 against election administration 0.1011), so part of the score is generic
house-style difference present in every topic; cluster 14 rests on 27 outlets and is folded into Other.

## Amendment added 26 Sep 2026, correcting the variant figures
CORRECTION to an earlier draft of this amendment, which wrongly described a "500-shuffle rerun" with 250
shuffles per half. No such rerun took place. The split-half computation still uses 20 shuffles per half
(05e_framing.py lines 145-146) and the Spearman-Brown value of 0.72 in the section above is unchanged and
NOT superseded. What did change is the variant comparison, which uses 500 permutations per variant score
(line 227). The variant figures in the "Measure settled, 20 Sep 2026" section above ARE superseded and must
not be quoted; the reliability and sanity-rule figures in that section stand.
Current variants against the primary, Spearman over the shared clusters: US outlets only 0.897 (32 clusters),
no NRC-VAD words 0.942 (33 clusters), Mixed-or-better outlets 0.732 (33 clusters).

Two consequences. The no-NRC-VAD-words variant clears its 0.90 bar at 0.942, so the concern that framing
divergence and arousal share vocabulary is answered empirically and the primary measure stands; the 0.899 in
the first-run note above is superseded and must not be cited as a failure. Rule 1, carried over by "All other
rules stand" in the 20 Sep amendment, requires the US-only variant to correlate at least 0.90 with the
primary; it reaches 0.897, so the rule is missed at the point estimate.

Decision recorded 26 Sep 2026, after the values were known: all 211 outlets are retained, and the missed
threshold is reported in the Methods chapter as a missed threshold - not rounded to 0.90, not omitted, and
not described as approximately meeting the rule.
