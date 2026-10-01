#!/usr/bin/env Rscript
# 06b_mixed.R - cross-classified model of article arousal: outlet factuality x topic partisanship,
# random intercepts for outlet and topic cluster. Reads models/model_frame.csv and writes
# models/06b_report.txt, models/coefficients.csv, models/variance_components.csv, two diagnostic figures.
suppressPackageStartupMessages({library(lme4); library(lmerTest)})

root <- Sys.getenv("THESIS_ROOT", file.path(path.expand("~"), "Desktop", "thesis"))
dir  <- file.path(root, "models")
d    <- read.csv(file.path(dir, "model_frame.csv"), stringsAsFactors = FALSE)

sink(file.path(dir, "06b_report.txt"), split = TRUE)
cat("06b_mixed -", format(Sys.time(), "%Y-%m-%d %H:%M"), "- lme4", format(packageVersion("lme4")),
    "| lmerTest", format(packageVersion("lmerTest")), "\n")

d$source    <- factor(d$source)
d$cluster   <- factor(d$topic_cluster_model)
d$year      <- factor(d$year)
d$bias_side <- factor(d$bias_side, levels = c("centre", "left", "right"))
stopifnot(!any(is.na(d$arousal_z)), !any(is.na(d$fact_c)), !any(is.na(d$part_z)))
cat(sprintf("%d articles | %d outlets | %d clusters | years %s\n", nrow(d), nlevels(d$source),
            nlevels(d$cluster), paste(levels(d$year), collapse = " ")))
cat("coefficients are in SDs of article arousal; fact_c is centred (1 unit = one factuality step),",
    "\npart_z is standardised across clusters\n")

coefs <- list(); vcs <- list()

fit <- function(label, form, data = d) {
  t0 <- Sys.time()
  m  <- lmer(form, data = data, REML = TRUE,
             control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("\n=== %s   (%.0fs%s)\n", label, secs, if (isSingular(m)) ", SINGULAR FIT" else ""))
  cat(deparse1(form), "\n")
  ct <- as.data.frame(coef(summary(m)))
  print(round(ct, 4))
  vc <- as.data.frame(VarCorr(m))
  vc <- vc[is.na(vc$var2), c("grp", "var1", "vcov")]
  vc$share <- vc$vcov / sum(vc$vcov)
  print(vc, row.names = FALSE, digits = 4)
  ct$term <- rownames(ct); ct$model <- label
  coefs[[label]] <<- ct
  vc$model <- label
  vcs[[label]] <<- vc
  m
}

m0 <- fit("M0 null",              arousal_z ~ 1 + (1 | source) + (1 | cluster))
m1 <- fit("M1 + year",            arousal_z ~ year + (1 | source) + (1 | cluster))
m2 <- fit("M2 + main effects",    arousal_z ~ year + fact_c + part_z + (1 | source) + (1 | cluster))
m3 <- fit("M3 + interaction",     arousal_z ~ year + fact_c * part_z + (1 | source) + (1 | cluster))
m4 <- fit("M4 + random slope",    arousal_z ~ year + fact_c * part_z + (1 + part_z | source) + (1 | cluster))
m5 <- fit("M5 + bias controls",   arousal_z ~ year + fact_c * part_z + bias_ext + bias_side +
                                              (1 + part_z | source) + (1 | cluster))
d6 <- d[d$bias_ext == 2, ]
cat(sprintf("\ncommon-support subset: bias extremity 2 only, %d articles from %d outlets\n",
            nrow(d6), length(unique(d6$source))))
print(table(unique(d6[, c("source", "fact")])$fact))
m6 <- fit("M6 common support",    arousal_z ~ year + fact_c * part_z + (1 | source) + (1 | cluster), data = d6)

cat("\nVARIANCE AT EACH STEP (how much outlet and cluster variance each predictor absorbs)\n")
vtab <- do.call(rbind, vcs)
wide <- reshape(vtab[, c("model", "grp", "vcov")], idvar = "model", timevar = "grp", direction = "wide")
print(wide, row.names = FALSE, digits = 4)

cat("\nICCs FROM THE NULL MODEL\n")
v0 <- vcs[["M0 null"]]
for (i in seq_len(nrow(v0))) cat(sprintf("   %-10s variance %.5f   ICC %.3f\n", v0$grp[i], v0$vcov[i], v0$share[i]))

cat("\nMODEL COMPARISONS (anova refits with ML)\n")
print(anova(m0, m1, m2, m3, m4, m5))

write.csv(do.call(rbind, coefs), file.path(dir, "coefficients.csv"), row.names = FALSE)
write.csv(vtab, file.path(dir, "variance_components.csv"), row.names = FALSE)

png(file.path(dir, "fig_diag_residuals.png"), width = 1400, height = 1100, res = 150)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot(fitted(m4), resid(m4), pch = ".", xlab = "fitted", ylab = "residual", main = "M4 residuals vs fitted")
abline(h = 0, col = "grey40")
qqnorm(resid(m4), pch = ".", main = "M4 residuals"); qqline(resid(m4), col = "grey40")
ro <- ranef(m4)$source[, 1]; qqnorm(ro, main = "outlet intercepts"); qqline(ro, col = "grey40")
rc <- ranef(m4)$cluster[, 1]; qqnorm(rc, main = "cluster intercepts"); qqline(rc, col = "grey40")
dev.off()

png(file.path(dir, "fig_diag_spread.png"), width = 1400, height = 700, res = 150)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
boxplot(resid(m4) ~ d$year, outline = FALSE, xlab = "year", ylab = "residual", main = "residual spread by year")
dec <- cut(d$n_matched, quantile(d$n_matched, probs = seq(0, 1, 0.1)), include.lowest = TRUE, labels = 1:10)
boxplot(resid(m4) ~ dec, outline = FALSE, xlab = "matched-word decile", ylab = "residual",
        main = "residual spread by article length")
dev.off()

cat("\nRESULT: fitted M0-M6, wrote coefficients.csv, variance_components.csv and two figures\n")
sink()
