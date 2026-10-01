# 10_robustness.R - does the arousal story survive different samples, controls and single outlets?
# Reads models/model_frame_plus.csv, model_frame_unfiltered.csv, exclude_outliers.csv.
# Writes 10_spec_table.csv, 10_bootstrap.csv, 10_influence.csv, 10_report.txt, fig_10_influence.png.
# QUICK=1 runs a 1-minute version. CORES=n overrides the core count.
suppressPackageStartupMessages({library(lme4); library(lmerTest); library(parallel)})
setwd(file.path(Sys.getenv("THESIS_ROOT", path.expand("~/Desktop/thesis")), "models"))
QUICK <- nchar(Sys.getenv("QUICK")) > 0
B_BOOT <- if (QUICK) 10 else 500
CORES  <- as.integer(Sys.getenv("CORES", max(1, detectCores() - 2)))
CTRL   <- lmerControl(calc.derivs = FALSE)
out_lines <- c()
say <- function(...) {s <- paste0(...); cat(s, "\n"); out_lines <<- c(out_lines, s)}

d <- read.csv("model_frame_plus.csv")
u <- read.csv("model_frame_unfiltered.csv")
need <- setdiff(c("arousal_z","fact_c","part_z","source","topic_cluster_model"), names(d))
if (length(need)) {say("STOP: model_frame_plus.csv is missing ", paste(need, collapse=", ")); quit(status=1)}
if (QUICK) {set.seed(1); d <- d[sample(nrow(d), 6000), ]; u <- u[sample(nrow(u), 6000), ]}
say(sprintf("10_robustness - %s - %d articles, %d outlets, %d clusters%s",
            format(Sys.time(), "%Y-%m-%d %H:%M"), nrow(d), length(unique(d$source)),
            length(unique(d$topic_cluster_model)), if (QUICK) " | QUICK TEST" else ""))
say(sprintf("bootstrap draws %d, leave-one-out refits %d outlets + %d clusters, %d cores",
            B_BOOT, length(unique(d$source)), length(unique(d$topic_cluster_model)), CORES))

F3 <- arousal_z ~ fact_c * part_z + (1 | source) + (1 | topic_cluster_model)
fit3 <- function(data, form = F3, w = NULL, reml = TRUE) {
  if (is.null(w)) return(lmer(form, data = data, REML = reml, control = CTRL))
  data$.w <- w
  lmer(form, data = data, weights = .w, REML = reml, control = CTRL)
}
grab <- function(m, name) {
  s <- summary(m)$coefficients
  terms <- c("fact_c", "part_z", "fact_c:part_z")
  data.frame(spec = name, term = terms, estimate = s[terms, 1], se = s[terms, 2],
             df = if (ncol(s) >= 5) s[terms, 3] else NA, p = s[terms, ncol(s)],
             n = nobs(m), outlets = length(unique(m@frame$source)),
             clusters = length(unique(m@frame$topic_cluster_model)), row.names = NULL)
}

specs <- list()
specs[["1 primary (M3)"]] <- function() fit3(d)
specs[["2 + bias extremity"]] <- function() fit3(d, update(F3, . ~ . + bias_ext))
specs[["3 common support (ext 2)"]] <- function() fit3(subset(d, bias_ext == 2))
if ("n_matched" %in% names(d)) specs[["4 weighted by matched words"]] <- function() fit3(d, w = d$n_matched)
if ("log_length" %in% names(d)) specs[["5 + article length"]] <- function() fit3(d, update(F3, . ~ . + log_length))
ex <- tryCatch(read.csv("exclude_outliers.csv")$source, error = function(e) character(0))
if (length(ex)) specs[["6 without outlier outlets"]] <- function() fit3(subset(d, !(source %in% ex)))
if ("country" %in% names(d)) specs[["7 US outlets only"]] <- function() fit3(subset(d, country == "USA"))
specs[["8 unfiltered sample"]] <- function() fit3(u)
if ("year" %in% names(d)) specs[["9 + year"]] <- function() fit3(d, update(F3, . ~ . + factor(year)))
if ("arousal_allwords" %in% names(d)) specs[["10 all-words arousal"]] <- function() {
  dd <- d; dd$arousal_z <- as.numeric(scale(dd$arousal_allwords)); fit3(dd)}

rows <- list()
for (nm in names(specs)) {
  t0 <- Sys.time()
  m <- tryCatch(specs[[nm]](), error = function(e) {say("   ", nm, " FAILED: ", conditionMessage(e)); NULL})
  if (!is.null(m)) {
    rows[[nm]] <- grab(m, nm); r <- rows[[nm]]
    say(sprintf("   %-28s b1 %+.4f (p %.3f)  b2 %+.4f (p %.3f)  b3 %+.4f (p %.3f)  n=%d  %.0fs",
                nm, r$estimate[1], r$p[1], r$estimate[2], r$p[2], r$estimate[3], r$p[3], r$n[1],
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }
}
write.csv(do.call(rbind, rows), "10_spec_table.csv", row.names = FALSE)

say("\nOUTLET BOOTSTRAP (resample the outlets with replacement, refit each time)")
outlets <- unique(d$source)
one_boot <- function(i) {
  set.seed(1000 + i)
  pick <- sample(outlets, length(outlets), replace = TRUE)
  dat <- do.call(rbind, lapply(seq_along(pick), function(k) {
    s <- d[d$source == pick[k], ]; s$source <- paste0(pick[k], "_", k); s}))
  m <- tryCatch(fit3(dat, reml = FALSE), error = function(e) NULL)
  if (is.null(m)) return(rep(NA_real_, 3))
  fixef(m)[c("fact_c", "part_z", "fact_c:part_z")]
}
bt <- do.call(rbind, mclapply(seq_len(B_BOOT), one_boot, mc.cores = CORES))
colnames(bt) <- c("fact_c", "part_z", "fact_c:part_z")
write.csv(bt, "10_bootstrap.csv", row.names = FALSE)
for (j in colnames(bt)) {
  v <- bt[, j][is.finite(bt[, j])]
  say(sprintf("   %-14s %+.4f  95%% CI [%+.4f, %+.4f]  P(<0) %.2f  (%d of %d draws converged)",
              j, mean(v), quantile(v, .025), quantile(v, .975), mean(v < 0), length(v), B_BOOT))
}

say("\nINFLUENCE: refit leaving out one outlet, then one topic cluster, at a time")
loo <- function(col, val) {
  m <- tryCatch(fit3(d[d[[col]] != val, ], reml = FALSE), error = function(e) NULL)
  if (is.null(m)) return(NULL)
  data.frame(kind = col, left_out = as.character(val), t(fixef(m)[c("fact_c", "part_z", "fact_c:part_z")]))
}
ids <- if (QUICK) head(unique(d$source), 8) else unique(d$source)
cls <- unique(d$topic_cluster_model)
inf <- do.call(rbind, c(mclapply(ids, function(v) loo("source", v), mc.cores = CORES),
                        mclapply(cls, function(v) loo("topic_cluster_model", v), mc.cores = CORES)))
names(inf)[3:5] <- c("fact_c", "part_z", "fact_c.part_z")
write.csv(inf, "10_influence.csv", row.names = FALSE)
base <- fixef(fit3(d, reml = FALSE))[c("fact_c", "part_z", "fact_c:part_z")]
for (k in unique(inf$kind)) for (j in 1:3) {
  v <- inf[inf$kind == k, 2 + j]
  worst <- inf[inf$kind == k, ][which.max(abs(v - base[j])), ]
  say(sprintf("   leave out one %-19s %-14s range [%+.4f, %+.4f]; largest shift dropping %s (%+.4f)",
              k, names(base)[j], min(v), max(v), worst$left_out, worst[[2 + j]] - base[j]))
}

png("fig_10_influence.png", width = 1600, height = 1200, res = 200)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
for (j in 1:3) {
  v <- inf[inf$kind == "source", 2 + j]
  hist(v, breaks = 30, main = names(base)[j], xlab = "estimate, one outlet left out",
       col = "grey80", border = "white")
  abline(v = base[j], lwd = 2); abline(v = 0, lty = 2)
}
dev.off()
say("\nRESULT: wrote 10_spec_table.csv, 10_bootstrap.csv, 10_influence.csv, fig_10_influence.png")
writeLines(out_lines, "10_report.txt")
