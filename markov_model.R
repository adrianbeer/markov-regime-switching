# Import Dataset
library("quantmod")
library(ggplot2)
library(dplyr)
library(latex2exp)
library("MSGARCH")
library(patchwork)
require(gridExtra)

theme_update(plot.title = element_text(hjust = 0.5))
msgarch_color = "green"
garch_color = "red"

# TODO: Include random walk error

data_dir = "C:\\Users\\Adria\\Documents\\Github Projects\\data\\markov-switching"
img_dir = "C:\\Users\\Adria\\Documents\\Github Projects\\markov-regime-switching\\LaTeX\\img"

start <- as.Date("1900-01-05")
end <- as.Date("2022-08-21")
df <- quantmod::getSymbols("^GSPC", src = "yahoo", from = start, to = end, auto.assign = FALSE)
price = df[, "GSPC.Close"]
colnames(price) <- "Price"
lret = diff(log(price))[-1,]
plot(lret)


corona_indices <- seq.Date(as.Date("2020-01-01"),as.Date("2022-05-01"),by="day")
housing_bubble <- seq.Date(as.Date("2008-01-01"),as.Date("2010-01-01"),by="day")
russia_war <- seq.Date(as.Date("2021-01-01"),as.Date("2023-01-01"),by="day")
oil <- seq.Date(as.Date("1986-01-01"),as.Date("1991-01-01"),by="day")

lret = lret*100 # For numerical purposes

split_idx = round(length(lret)*0.8)
y = lret
train_y = lret[1:split_idx]
test_y = lret[(1+split_idx):length(lret)]

real_vol = abs(lret)
train_vol = real_vol[1:split_idx]
test_vol = real_vol[(1+split_idx):length(lret)]


bear_test_dates <- c(seq.Date(as.Date("2020-02-01"),as.Date("2020-04-01"),by="day"),
                     seq.Date(as.Date("2021-01-01"),as.Date("2023-01-01"),by="day"))
bull_test_dates <- index(test_vol)[!(index(test_vol) %in% bear_test_dates)]
  
# # Imports OMI realized volatility
# omi <- read.csv(paste(data_dir, "\\oxfordmanrealizedvolatilityindices.csv", sep=""))
# omi <- omi[omi$Symbol==".SPX", ]
# omi$Date <- as.Date(omi$X)
# plot(omi$rk_parzen*sqrt(250))
# measure_name <- "rk_parzen"
# # In the oxfordmanrealizedvolatilityindices.csv file the vola measures give the
# # estimated variance, NOT the volatility, so we have to take the square root
# omi_vol <- xts(sqrt(omi[, c(measure_name)]), order.by=as.Date(omi[, 21]))
# colnames(omi_vol) <- measure_name
# 
# test_omi_vol <- omi_vol[index(test_vol)]

# ---------------------- Training the models -----------------------------------
# ------------------------------------------------------------------------------
set.seed(420)

# Training the models took approx. 1 min
start.time <- Sys.time()

# SINGLE-REGIME 
garch.n <- CreateSpec(variance.spec = list(model = c("gjrGARCH")),
                          distribution.spec = list(distribution = c("sstd")))
summary(garch.n)
garch.fit.ml <- FitML(spec = garch.n, data = train_y)
summary(garch.fit.ml)

# MULTI-REGIME (non-switching shape)
ms2.garch.n <- CreateSpec(variance.spec = list(model = c("gjrGARCH", "gjrGARCH")),
                   distribution.spec = list(distribution = c("sged", "sged")))
summary(ms2.garch.n)
msgarch.fit.ml <- FitML(spec = ms2.garch.n, data = train_y)
summary(msgarch.fit.ml)


# Unconditional volatility levels
sqrt(250) * sapply(ExtractStateFit(msgarch.fit.ml), UncVol)
sqrt(250) * sapply(ExtractStateFit(garch.fit.ml), UncVol)

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

# MSGARCH only: in-sample Probabilities
msgarch.fit.ml.state <- State(msgarch.fit.ml)
pred_probs <- msgarch.fit.ml.state$PredProb[, 1, 1]
smooth_probs <- msgarch.fit.ml.state$SmoothProb[, 1, 1]

probs <- data.frame(Dates = as.Date(names(pred_probs)), predicted=pred_probs, smoothed=smooth_probs) # Convert to data.frame for ggplot2

p1 <- ggplot() + 
  geom_line(data=probs, aes(x=Dates, y=predicted, colour="predicted")) + 
  geom_line(data=probs, aes(x=Dates, y=smoothed, colour="smoothed")) + 
  scale_x_date(date_breaks = "2 years" , date_labels = "%y") + 
  xlab("Dates (Year)") + 
  ylab("Prob") + 
  ggtitle("Predicted in-sample Probability of state 1")

p1
#ggsave(paste(img_dir, "\\PredProb_State1_InSample.pdf", sep=""))

# In-sample volatility
msgarch_train_vola <- Volatility(msgarch.fit.ml) %>% fortify.zoo
garch_train_vola <- Volatility(garch.fit.ml) %>% fortify.zoo


p2 <- ggplot(msgarch_train_vola) + geom_line(aes(x=Index, y=.)) + 
  ylab("Vola") + 
  ggtitle("Filtered in-sample Volatility") + 
  scale_x_date(date_breaks = "2 years" , date_labels = "%y")

p2
#ggsave(paste(img_dir, "\\FiltVola_InSample.pdf", sep=""))

grid.arrange(p1, p2, nrow=2)


# ---------------------- Creating Point Forecasts ------------------------------
# ------------------------------------------------------------------------------

# Calculating the Point forecasts took approx. 1 min.
start.time <- Sys.time()
horizons = 1:10 # currently only works for horizon=1!!


fcst <- xts(x = cbind(replicate(length(horizons), rep(-1, length(test_y)))), 
               order.by = index(test_y))
names(fcst) = sapply(horizons, function(x) paste("GARCH_h", toString(x), sep=""))

#SINGLE-REGIME
for (dd in 1:(length(test_y)-max(horizons))) {
  pred <- predict(garch.fit.ml, nahead = max(horizons), do.return.draw = FALSE, newdata=test_y[1:dd])
  for (hh in 1:length(horizons)) {
    stopifnot(index(fcst)[dd+h] == index(pred$vol)[hh])
    fcst[dd+hh,] = pred$vol[hh]
  }
}  
garch_fcst <- fcst

fcst <- xts(x = cbind(replicate(length(horizons), rep(-1, length(test_y)))), 
            order.by = index(test_y))
names(fcst) = sapply(horizons, function(x) paste("MSGARCH_h", toString(x), sep=""))

#MULTI-REGIME
for (dd in 1:(length(test_y)-max(horizons))) {
  pred <- predict(msgarch.fit.ml, nahead = max(horizons), do.return.draw = FALSE, newdata=test_y[1:dd])
  fcst[dd+1,] = pred$vol[horizons]
}  
msgarch_fcst <- fcst

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken



# --------------- Point Forecast Evaluation -------------------------------
# ------------------------------------------------------------------------------


f_mse <- function(y_hat, y) {
  return (mean((y_hat - y)^2))
}
f_mae <- function(y_hat, y) {
  return (mean(abs(y_hat - y)))
}
f_bias <- function(y_hat, y) {
  return (mean(y_hat - y))
}
f_rmse <- function(y_hat, y) {
  return (sqrt(f_mse(y_hat, y)))
}
get_err_table <- function(y_hat, y) {
  mse <- sapply(y_hat, f_mse, y)
  mae <- sapply(y_hat, f_mae, y)
  bias <- sapply(y_hat, f_bias, y)
  rmse <- sapply(y_hat, f_rmse, y)
  
  err_table <- cbind(mse, mae, rmse, bias)
  return(err_table)
}

# ---------------------------- Out of sample -------------------------------
msgarch_err <- get_err_table(msgarch_fcst, test_vol)
msgarch_err

garch_err <- get_err_table(garch_fcst, test_vol)
garch_err

avg_err <- get_err_table(mean(train_vol), test_vol)
avg_err
rownames(avg_err) = "Avg"

rbind(msgarch_err, garch_err, avg_err)
# ---------------------------- In sample -------------------------------


# --------------- Scenario Forecast evaluation etc. ----------------------------
# --------------- Latent State evaluation --------------------------------------
# ------------------------------------------------------------------------------

# # Probabilities for out-of-sample period
# dummy_ms2.garch.n <- CreateSpec(variance.spec = list(model = c("gjrGARCH", "gjrGARCH")),
#                           distribution.spec = list(distribution = c("sstd", "sstd")),
#                         constraint.spec = list(fixed=msgarch.fit.ml$par))
# summary(dummy_ms2.garch.n)
# dummy_msgarch.fit.ml <- FitML(spec = dummy_ms2.garch.n, data = train_y)
# summary(dummy_msgarch.fit.ml)

# -------------------------  Corona Case Study --------------------------------

# CORONA POINT FORECAST
p1 <- ggplot() + 
  geom_line(data=msgarch_fcst[index(msgarch_fcst) %in% corona_indices, ], aes(x=Index, y=MSGARCH_h1, colour="msgarch")) + 
  geom_line(data=garch_fcst[index(msgarch_fcst) %in% corona_indices, ], aes(x=Index, y=GARCH_h1, colour="garch")) +
  scale_color_manual(name = "Model", values = c("garch"=garch_color, "msgarch"=msgarch_color)) + 
  scale_x_date(date_breaks = "2 years" , date_labels = "%y") + 
  xlab("Dates (Year)") + 
  ylab("Prob") + 
  ggtitle("1-day-ahead volatility forecasts during the Corona Crash")

p1
#ggsave(paste(img_dir, "\\PredProb_State1_InSample.pdf", sep=""))

# RUSSIA WAR POINT FORECAST
p1 <- ggplot() + 
  geom_line(data=msgarch_fcst[index(msgarch_fcst) %in% russia_war, ], aes(x=Index, y=MSGARCH_h1, colour="msgarch")) + 
  geom_line(data=garch_fcst[index(msgarch_fcst) %in% russia_war, ], aes(x=Index, y=GARCH_h1, colour="garch")) +
  scale_color_manual(name = "Model", values = c("garch"=garch_color, "msgarch"=msgarch_color)) + 
  scale_x_date(date_breaks = "2 years" , date_labels = "%y") + 
  xlab("Dates (Year)") + 
  ylab("Prob") + 
  ggtitle("1-day-ahead volatility forecasts during the Russian War Crash")

p1


# HOUSING BUBBLE POINT FORECAST
p1 <- ggplot() + 
  geom_line(data=msgarch_fcst[index(msgarch_fcst) %in% housing_bubble, ], aes(x=Index, y=MSGARCH_h1, colour="msgarch")) + 
  geom_line(data=garch_fcst[index(msgarch_fcst) %in% housing_bubble, ], aes(x=Index, y=GARCH_h1, colour="garch")) +
  scale_color_manual(name = "Model", values = c("garch"=garch_color, "msgarch"=msgarch_color)) + 
  scale_x_date(date_breaks = "2 years" , date_labels = "%y") + 
  xlab("Dates (Year)") + 
  ylab("Prob") + 
  ggtitle("Volatility 1-day-ahead forecasts during the 2008 Crash")
p1





