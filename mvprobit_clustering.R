# =============================================================================
# mvprobit_clustering_simple.R
# Cluster LLMs by instruction-following performance with a multivariate probit.
#
# Model (i = LLM, j = prompt):
#   y_ij = 1 if z_ij > 0, else 0
#   z_ij = a_j + lambda_j' f_i + e_ij,     e_ij ~ N(0, 1)
#   f_i ~ N(0, I_d)               latent skills of LLM i (d numbers)
#   a_j, lambda_j ~ N(0, 4)       difficulty and loadings of prompt j
#
# Steps
#   1. Fit the model with a Gibbs sampler (Albert-Chib data augmentation).
#   2. Choose d by predicting 10% of cells that were hidden during fitting.
#   3. Cluster the LLMs' latent skills f_i with a Gaussian mixture (mclust).
#
# Same method and results as mvprobit_clustering.R, written to be easier to read.
# Run from the folder holding the two CSV files:
#   Rscript mvprobit_clustering/mvprobit_clustering_simple.R
# =============================================================================
library(mclust)
set.seed(2026)
out_dir <- "mvprobit_clustering/output"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- 0. Data ----------------------------------------------------------------
raw <- read.csv("openeval_ifeval_only_binary_matrix.csv", check.names = FALSE)
Y <- as.matrix(raw[, -1])                 # 122 LLMs x 534 prompts, 1 = pass
rownames(Y) <- raw$model_name
success <- rowMeans(Y)                    # overall success rate of each LLM

# family[j, f] is TRUE if prompt j contains an instruction of family f
meta   <- read.csv("openeval_item_instruction_metadata_long.csv")
family <- table(factor(meta$item_id, levels = colnames(Y)), meta$instruction_family) > 0

# ---- 1. Gibbs sampler -------------------------------------------------------

# Draw z ~ N(mu, 1) truncated to z > 0 when y = 1 and z < 0 when y = 0
# (inverse-CDF method). If y is NA (hidden cell), z is not truncated.
draw_latent <- function(mu, y) {
  p0 <- pnorm(-mu)                        # P(z < 0)
  u  <- runif(length(mu))
  p  <- ifelse(is.na(y), u, ifelse(y == 1, p0 + u * (1 - p0), u * p0))
  mu + qnorm(pmin(pmax(p, 1e-12), 1 - 1e-12))
}

# Draw coefficients B in the regression Z = X B + noise, noise ~ N(0, 1),
# prior B ~ N(0, 1 / prior_prec). Each column of Z is a separate regression.
# Posterior: N(V X'Z, V) with V = (X'X + prior_prec * I)^(-1).
draw_coefs <- function(X, Z, prior_prec) {
  R    <- chol(crossprod(X) + diag(prior_prec, ncol(X)))  # R'R = V^(-1)
  mean <- backsolve(R, forwardsolve(t(R), crossprod(X, Z)))
  mean + backsolve(R, matrix(rnorm(length(mean)), nrow(mean)))
}

fit_mvprobit <- function(Y, d, n_iter = 3000, burn = 1000, thin = 5) {
  n <- nrow(Y); J <- ncol(Y)

  # starting values
  F <- matrix(rnorm(n * d, sd = 0.5), n, d)                 # LLM skills
  a <- qnorm(pmin(pmax(colMeans(Y, na.rm = TRUE), 0.02), 0.98))
  B <- cbind(a, matrix(rnorm(J * d, sd = 0.1), J, d))       # prompts: [a_j, lambda_j]

  keep    <- seq(burn + 1, n_iter, by = thin)
  F_draws <- array(NA, c(length(keep), n, d))
  P_sum   <- 0
  loglik  <- numeric(n_iter)
  s       <- 0

  for (it in 1:n_iter) {
    # (a) latent utilities z_ij given everything else
    Z <- matrix(draw_latent(cbind(1, F) %*% t(B), Y), n, J)

    # (b) prompt parameters: regress each prompt's column of Z on [1, f_i]
    B <- t(draw_coefs(cbind(1, F), Z, prior_prec = 1 / 4))

    # (c) LLM skills: regress each LLM's (z_ij - a_j) on the loadings lambda_j
    F <- t(draw_coefs(B[, -1, drop = FALSE], t(Z) - B[, 1], prior_prec = 1))

    P <- pnorm(cbind(1, F) %*% t(B))                         # P(y_ij = 1)
    loglik[it] <- sum(log(pmax(ifelse(Y == 1, P, 1 - P), 1e-12)), na.rm = TRUE)

    if (it %in% keep) {
      s <- s + 1
      # f is only identified up to a rotation: rotate each draw to match the first
      if (s == 1) F_ref <- F
      sv <- svd(crossprod(F, F_ref))
      F_draws[s, , ] <- F %*% sv$u %*% t(sv$v)
      P_sum <- P_sum + P
    }
  }
  list(F_draws = F_draws,
       F       = apply(F_draws, c(2, 3), mean),              # posterior mean skills
       P       = P_sum / length(keep),                       # posterior mean P(pass)
       loglik  = loglik)
}

# ---- 2. Choose d: hide 10% of cells and predict them ------------------------
hidden <- matrix(runif(length(Y)) < 0.10, nrow(Y))
Y_train <- Y
Y_train[hidden] <- NA

log_loss <- function(y, p) {
  p <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

d_tab <- data.frame(d = 1:4, heldout_logloss = NA)
for (d in 1:4) {
  fit_d <- fit_mvprobit(Y_train, d, n_iter = 1500, burn = 500)
  d_tab$heldout_logloss[d] <- log_loss(Y[hidden], fit_d$P[hidden])
}
print(d_tab)
# smallest d within 0.002 of the best log loss
d_star <- min(d_tab$d[d_tab$heldout_logloss <= min(d_tab$heldout_logloss) + 0.002])
cat("Chosen d =", d_star, "\n")
write.csv(d_tab, file.path(out_dir, "d_selection.csv"), row.names = FALSE)

# ---- Final fit on all data --------------------------------------------------
fit <- fit_mvprobit(Y, d_star)
skills <- fit$F
rownames(skills) <- rownames(Y)
colnames(skills) <- paste0("f", 1:d_star)

png(file.path(out_dir, "trace_loglik.png"), 1000, 450, res = 120)
plot(fit$loglik, type = "l", xlab = "iteration", ylab = "log-likelihood", main = "Gibbs trace")
abline(v = 1000, lty = 2, col = 2)
invisible(dev.off())

# ---- 3. Cluster LLMs on their latent skills ---------------------------------
gm <- Mclust(skills, G = 1:8, verbose = FALSE)      # number of clusters K by BIC

# relabel clusters so that 1 = weakest, K = strongest (by mean success rate)
rank_order <- order(tapply(success, gm$classification, mean))
relabel    <- function(g) match(g, rank_order)
cluster    <- relabel(gm$classification)
K          <- max(cluster)
cat("Gaussian mixture:", gm$modelName, "with K =", K, "clusters (BIC)\n")

# stability: share of posterior draws in which an LLM lands in its own cluster
same <- matrix(NA, dim(fit$F_draws)[1], nrow(Y))
for (s in 1:nrow(same)) {
  draw_s    <- matrix(fit$F_draws[s, , ], ncol = d_star)
  same[s, ] <- relabel(predict(gm, draw_s)$classification) == cluster
}
stability <- colMeans(same)

# success rate of each cluster on each instruction family
family_rates <- sapply(colnames(family), function(f)
  sapply(1:K, function(k) mean(Y[cluster == k, family[, f]])))
profile <- data.frame(cluster = 1:K,
                      n_llms  = as.vector(table(cluster)),
                      overall = tapply(success, cluster, mean),
                      family_rates)
print(round(profile, 3))

res <- data.frame(model = rownames(Y), cluster = cluster,
                  stability = round(stability, 2), success = round(success, 3),
                  round(skills, 3))
res <- res[order(res$cluster, -res$success), ]
write.csv(res, file.path(out_dir, "llm_clusters.csv"), row.names = FALSE)
write.csv(round(profile, 3), file.path(out_dir, "cluster_profiles.csv"), row.names = FALSE)

for (k in 1:K) {
  cat(sprintf("\nCluster %d (%d LLMs): %s\n", k, sum(cluster == k),
              paste(res$model[res$cluster == k], collapse = ", ")))
}
cat(sprintf("\nMean membership stability %.2f\n", mean(stability)))

# ---- Plots ------------------------------------------------------------------
cols <- hcl.colors(K, "Dark 3")

png(file.path(out_dir, "latent_space.png"), 1100, 950, res = 130)
plot(skills[, 1], skills[, 2], col = cols[cluster], pch = 19, xlab = "f1", ylab = "f2",
     main = "LLMs in multivariate probit latent skill space")
legend("bottomright", paste("cluster", 1:K), col = cols, pch = 19, bty = "n")
invisible(dev.off())

png(file.path(out_dir, "cluster_profiles.png"), 1200, 700, res = 130)
par(mar = c(10, 4, 3, 1))
matplot(t(family_rates), type = "b", pch = 19, lty = 1, col = cols, ylim = c(0, 1),
        xaxt = "n", ylab = "strict success rate",
        main = "Cluster success rate by instruction family")
axis(1, at = 1:ncol(family), labels = colnames(family), las = 2)
legend("bottomleft", paste0("cluster ", 1:K, " (n=", table(cluster), ")"),
       col = cols, pch = 19, bty = "n")
invisible(dev.off())
