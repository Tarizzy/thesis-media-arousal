# 10b_no2018.R - the M3 model excluding 2018, whose source text is partly stripped of punctuation.
suppressPackageStartupMessages({library(lme4); library(lmerTest)})
setwd(file.path(Sys.getenv("THESIS_ROOT", path.expand("~/Desktop/thesis")), "models"))
d <- read.csv("model_frame_plus.csv")
f <- arousal_z ~ fact_c * part_z + factor(year) + (1 | source) + (1 | topic_cluster_model)
fit <- function(x) summary(lmer(f, data = x, REML = TRUE,
                 control = lmerControl(calc.derivs = FALSE)))$coefficients[
                 c("fact_c", "part_z", "fact_c:part_z"), c(1, 2, 5)]
o <- c(sprintf("10b_no2018 - %s", format(Sys.time(), "%Y-%m-%d %H:%M")),
       sprintf("all years: %d articles", nrow(d)), capture.output(print(round(fit(d), 4))),
       sprintf("\nwithout 2018: %d articles (%d dropped), %d outlets, %d clusters",
               sum(d$year != 2018), sum(d$year == 2018), length(unique(d$source[d$year != 2018])),
               length(unique(d$topic_cluster_model[d$year != 2018]))),
       capture.output(print(round(fit(subset(d, year != 2018)), 4))))
writeLines(o, "10b_no2018_report.txt"); cat(o, sep = "\n")
