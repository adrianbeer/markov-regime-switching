# Import Dataset
library("quantmod")
library(ggplot2)
library(dplyr)
library(latex2exp)

data_dir = "C:\\Users\\Adria\\Documents\\Github Projects\\data\\markov-switching"

start <- as.Date("1988-01-05")
end <- as.Date("2022-08-21")
df <- quantmod::getSymbols("^SP500TR", src = "yahoo", from = start, to = end, auto.assign = FALSE)
price = df[, "SP500TR.Close"]
colnames(price) <- "Price"
lret = diff(log(price))[-1,]
plot(lret)


lret = lret*100

split_idx = round(length(lret)*0.9)
y = lret
train_y = lret[1:split_idx]
test_y = lret[(1+split_idx):length(lret)]

real_vol = abs(lret)


# Imports realized volatility 
omi <- read.csv(paste(data_dir, "\\oxfordmanrealizedvolatilityindices.csv", sep=""))
omi <- omi[omi$Symbol==".SPX", ]
omi$Date <- as.Date(omi$X)
plot(omi$rk_parzen*sqrt(250))

omi_vol <- xts(select(omi, "rk_parzen"), order.by=as.Date(omi[, 21]))

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

# Unconditional volatility levels
sqrt(250) * sapply(ExtractStateFit(fit.ml), UncVol)

# Predicted in-sample Probabilities
fit.ml.state <- State(fit.ml)
pred_probs <- fit.ml.state$PredProb[, 1, 1]
pred_probs <- data.frame(Dates = as.Date(names(pred_probs)), value=pred_probs) # Convert to data.frame for ggplot2
ggplot() + 
  geom_line(data=pred_probs, aes(x=Dates, y=value, group=1)) + 
  scale_x_date(date_breaks = "years" , date_labels = "%y") + 
  xlab("Dates (Year)") + 
  ylab("Prob") + 
  ggtitle("Predicted in-sample Probability of state 1") + 
  theme(plot.title = element_text(hjust = 0.5))


# ---------------------- Creating Point Forecasts ------------------------------
# ------------------------------------------------------------------------------
start.time <- Sys.time()

horizons = c(1,5)
fcst <- xts(x = cbind(replicate(length(horizons), rep(-1, length(test_y)))), 
               order.by = index(test_y))
names(fcst) = sapply(horizons, function(x) paste("h_d", toString(x), sep=""))


for (dd in 1:length(test_y)) {
  pred <- predict(fit.ml, nahead = max(horizons), do.return.draw = FALSE, newdata=test_y[1:dd])
  fcst[dd,] = pred$vol[horizons]
}  
msgarch_fcst <- fcst

#TODO: Shift 5day horizon by 5... assign dates correctly the the predictions

ggplot() + geom_line(data=msgarch_fcst, aes(x=Index, y=h_d1, colour="1")) + 
  geom_line(data=msgarch_fcst, aes(x=Index, y=h_d5, colour="5")) +
  scale_color_manual(name = "horizon", values = c("1" = "darkblue", "5" = "red"))

ggplot() + 
  geom_line(data=real_vol[index(msgarch_fcst)], aes(x=Index, y=Price, colour="a")) +
  geom_line(data=omi_vol[index(msgarch_fcst)]*1000, aes(x=Index, y=rk_parzen, colour="b"))

plot(omi_vol)

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

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











