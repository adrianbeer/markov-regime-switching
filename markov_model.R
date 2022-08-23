# Import Dataset
library("quantmod")

start <- as.Date("1950-01-01")
end <- as.Date("2022-08-21")
df <- quantmod::getSymbols("^SP500TR", src = "yahoo", from = start, to = end, auto.assign = FALSE)
price = df[, "SP500TR.Close"]
lret = diff(log(price))[-1,]
plot(lret)


lret = lret*100

split_idx = round(length(lret)*0.9)
y = lret
train_y = lret[1:split_idx]
test_y = lret[(1+split_idx):length(lret)]

real_vol = abs(lret)


#HMM
library("MSGARCH")

# SINGLE REGIME

# ---------------------- Training the models -----------------------------------
# ------------------------------------------------------------------------------



# MULTI-REGIME (non-switching shape)
ms2.garch.n <- CreateSpec(variance.spec = list(model = c("tGARCH", "tGARCH")),
                   distribution.spec = list(distribution = c("sged", "sged")))
summary(ms2.garch.n)
fit.ml <- FitML(spec = ms2.garch.n, data = train_y)
summary(fit.ml)



# ---------------------- Creating Point Forecasts ------------------------------
# ------------------------------------------------------------------------------
horizons = c(1,5)
fcst <- xts(x = cbind(replicate(length(horizons), rep(-1, length(test_y)))), 
               order.by = index(test_y))
names(fcst) = sapply(horizons, function(x) paste("h_d", toString(x)))


for (dd in 1:length(test_y)) {
  pred <- predict(fit.ml, nahead = max(horizons), do.return.draw = FALSE, newdata=test_y[1:dd])
  fcst[dd,] = pred$vol[horizons]
}  
msgarch_fcst <- fcst

plot(msgarch_fcst[, 2])
lines(msgarch_fcst[, 1])
lines(real_vol[index(msgarch_fcst)], col="red")


# --------------- Point Forecast evaluation etc. -------------------------------
# ------------------------------------------------------------------------------

f_mse <- function(y_hat, y) {
  return (mean((y_hat - y)^2))
}
f_mae <- function(y_hat, y) {
  return (mean(abs(y_hat - y)))
}
f_rmse <- function(y_hat, y) {
  return (sqrt(f_mse(y_hat, y)))
}
get_err_table <- function(y_hat, y) {
  mse <- sapply(y_hat, f_mse, y)
  mae <- sapply(y_hat, f_mae, y)
  rmse <- sapply(y_hat, f_rmse, y)
  
  err_table <- cbind(mse, mae, rmse)
  return(err_table)
}

msgarch_err <- get_err_table(fcst, real_vol[index(msgarch_fcst)])
naive_err <- get_err_table(mean(real_vol), real_vol[index(msgarch_fcst)])


# --------------- Scenario Forecast evaluation etc. ----------------------------
# --------------- Latent State evaluation --------------------------------------
# ------------------------------------------------------------------------------











