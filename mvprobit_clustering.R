# =============================================================================
# mvprobit_clustering.R
# Cluster LLMs on IFEval binary pass/fail data with a multivariate probit model.
#
# Model (factor / low-rank multivariate probit), i = LLM, j = prompt:
#   y_ij = 1{ z_ij > 0 }
#   z_ij = a_j + lambda_j' f_i + e_ij,   e_ij ~ N(0, 1)
#   f_i ~ N(0, I_d)                      LLM latent skill vector
#   a_j, lambda_j ~ N(0, s2)             prompt difficulty and loadings
# For one LLM, z_i is multivariate normal with
#   corr(z_ij, z_ik) = lambda_j'lambda_k / sqrt((1+|lambda_j|^2)(1+|lambda_k|^2)),
# a multivariate probit whose latent correlation matrix has rank d (a full
# 534 x 534 correlation matrix cannot be estimated from 122 LLMs).
# Fit: Albert-Chib data augmentation Gibbs sampler.
# Clustering: Gaussian mixture (mclust, K by BIC) on the continuous latent
# skills f_i, with membership uncertainty from the posterior draws.
#
# Run from the folder holding openeval_ifeval_only_binary_matrix.csv:
#   install.packages("mclust");  Rscript mvprobit_clustering/mvprobit_clustering.R
# =============================================================================
library(mclust)
set.seed(2026)
out_dir <- "mvprobit_clustering/output"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Data -------------------------------------------------------------------
raw <- read.csv("openeval_ifeval_only_binary_matrix.csv", check.names = FALSE)
Y <- as.matrix(raw[, -1]); rownames(Y) <- raw$model_name
meta <- read.csv("openeval_item_instruction_metadata_long.csv")
fam  <- table(factor(meta$item_id, levels = colnames(Y)), meta$instruction_family) > 0
cat(sprintf("%d LLMs x %d prompts, success rate %.3f\n", nrow(Y), ncol(Y), mean(Y)))

# ---- Gibbs sampler ----------------------------------------------------------
# z ~ N(mu, 1) truncated to z > 0 if y = 1, z < 0 if y = 0; y = NA -> untruncated
rtnorm_sign <- function(mu, y) {
  p0 <- pnorm(-mu); u <- runif(length(mu))
  pr <- ifelse(is.na(y), u, ifelse(y == 1, p0 + u * (1 - p0), u * p0))
  mu + qnorm(pmin(pmax(pr, 1e-12), 1 - 1e-12))
}

mvprobit_gibbs <- function(Y, d = 2, n_iter = 3000, burn = 1000, thin = 5, s2 = 4) {
  n <- nrow(Y); J <- ncol(Y); q <- d + 1
  F <- matrix(rnorm(n * d, sd = 0.5), n, d)
  B <- cbind(qnorm(pmin(pmax(colMeans(Y, na.rm = TRUE), .02), .98)),
             matrix(rnorm(J * d, sd = 0.1), J, d))            # [a_j, lambda_j]
  keep <- seq(burn + 1, n_iter, by = thin)
  F_draws <- array(NA, c(length(keep), n, d)); P <- 0; ref <- NULL; s <- 0
  loglik <- numeric(n_iter)
  for (it in 1:n_iter) {
    W <- cbind(1, F)
    # 1) latent utilities
    Z <- matrix(rtnorm_sign(W %*% t(B), Y), n, J)
    # 2) prompt parameters: Bayesian regression of each column of Z on [1, F]
    Ch <- chol(crossprod(W) + diag(1 / s2, q))
    B  <- t(backsolve(Ch, forwardsolve(t(Ch), crossprod(W, Z))) +
            backsolve(Ch, matrix(rnorm(q * J), q, J)))
    # 3) LLM skills: regression of each row of (Z - a) on loadings, prior N(0, I)
    L  <- B[, -1, drop = FALSE]
    Ch <- chol(crossprod(L) + diag(d))
    F  <- t(backsolve(Ch, forwardsolve(t(Ch), crossprod(L, t(Z) - B[, 1]))) +
            backsolve(Ch, matrix(rnorm(d * n), d, n)))
    Pr <- pnorm(cbind(1, F) %*% t(B))
    loglik[it] <- sum(log(pmax(ifelse(Y == 1, Pr, 1 - Pr), 1e-12)), na.rm = TRUE)
    if (it %in% keep) {
      s <- s + 1
      # rotation of (F, lambda) is not identified: Procrustes-align to first kept draw
      if (is.null(ref)) ref <- F
      sv <- svd(crossprod(F, ref)); F_draws[s, , ] <- F %*% sv$u %*% t(sv$v)
      P <- P + Pr
    }
  }
  list(F_draws = F_draws, F = apply(F_draws, c(2, 3), mean), P = P / length(keep),
       loglik = loglik, d = d)
}

# ---- Choose d: predict 10% randomly held-out cells -------------------------
mask <- matrix(runif(length(Y)) < 0.10, nrow(Y)); Ym <- Y; Ym[mask] <- NA
log_loss <- function(y, p) { p <- pmin(pmax(p, 1e-6), 1 - 1e-6); -mean(y * log(p) + (1 - y) * log(1 - p)) }
d_tab <- data.frame(d = 1:4, heldout_logloss = sapply(1:4, function(d)
  log_loss(Y[mask], mvprobit_gibbs(Ym, d, n_iter = 1500, burn = 500)$P[mask])))
print(d_tab)
d_star <- min(d_tab$d[d_tab$heldout_logloss <= min(d_tab$heldout_logloss) + 0.002])
cat("Chosen d =", d_star, "\n")
write.csv(d_tab, file.path(out_dir, "d_selection.csv"), row.names = FALSE)

# ---- Final fit on all data --------------------------------------------------
fit <- mvprobit_gibbs(Y, d_star)
Fm <- fit$F; rownames(Fm) <- rownames(Y); colnames(Fm) <- paste0("f", 1:d_star)
png(file.path(out_dir, "trace_loglik.png"), 1000, 450, res = 120)
plot(fit$loglik, type = "l", xlab = "iteration", ylab = "log-likelihood", main = "Gibbs trace")
abline(v = 1000, lty = 2, col = 2); invisible(dev.off())

# ---- Cluster LLMs on latent skills -----------------------------------------
gm <- Mclust(Fm, G = 1:8, verbose = FALSE)
cl <- match(gm$classification, order(tapply(rowMeans(Y), gm$classification, mean)))
K  <- max(cl)                                    # cluster 1 = weakest
cat("Gaussian mixture:", gm$modelName, "with K =", K, "clusters (BIC)\n")

# membership stability: share of posterior draws classified into the same cluster
stability <- rowMeans(sapply(seq_len(dim(fit$F_draws)[1]), function(s)
  match(predict(gm, matrix(fit$F_draws[s, , ], ncol = d_star))$classification,
        order(tapply(rowMeans(Y), gm$classification, mean))) == cl))

profile <- t(sapply(1:K, function(k) sapply(colnames(fam), function(f)
  mean(Y[cl == k, fam[, f]]))))
profile <- data.frame(cluster = 1:K, n_llms = as.vector(table(cl)),
                      overall = tapply(rowMeans(Y), cl, mean), profile)
print(round(profile, 3))
res <- data.frame(model = rownames(Y), cluster = cl, stability = round(stability, 2),
                  success = round(rowMeans(Y), 3), round(Fm, 3))
res <- res[order(res$cluster, -res$success), ]
write.csv(res, file.path(out_dir, "llm_clusters.csv"), row.names = FALSE)
write.csv(round(profile, 3), file.path(out_dir, "cluster_profiles.csv"), row.names = FALSE)
for (k in 1:K) cat(sprintf("\nCluster %d (%d LLMs): %s\n", k, sum(cl == k),
                           paste(res$model[res$cluster == k], collapse = ", ")))
cat(sprintf("\nMean membership stability %.2f\n", mean(stability)))

# ---- Plots ------------------------------------------------------------------
cols <- hcl.colors(K, "Dark 3")
png(file.path(out_dir, "latent_space.png"), 1100, 950, res = 130)
if (d_star >= 2) {
  plot(Fm[, 1:2], col = cols[cl], pch = 19, xlab = "f1", ylab = "f2",
       main = "LLMs in multivariate probit latent skill space")
} else stripchart(Fm[, 1] ~ cl, col = cols, pch = 19, method = "jitter", xlab = "f1")
legend("bottomright", paste("cluster", 1:K), col = cols, pch = 19, bty = "n")
invisible(dev.off())
png(file.path(out_dir, "cluster_profiles.png"), 1200, 700, res = 130)
par(mar = c(10, 4, 3, 1))
matplot(t(as.matrix(profile[, -(1:3)])), type = "b", pch = 19, lty = 1, col = cols,
        ylim = c(0, 1), xaxt = "n", ylab = "strict success rate",
        main = "Cluster success rate by instruction family")
axis(1, at = 1:ncol(fam), labels = colnames(fam), las = 2)
legend("bottomleft", paste0("cluster ", 1:K, " (n=", table(cl), ")"), col = cols, pch = 19, bty = "n")
invisible(dev.off())


