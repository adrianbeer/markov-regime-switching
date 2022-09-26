# Import Dataset
library(quantmod)
library(ggplot2)
library(dplyr)
library(latex2exp)
library(MSGARCH)
library(patchwork)
require(gridExtra)
library(murphydiagram)
library(forecast)
library(xtable)
library(fGarch)

# Set working directory to this file's directory.

# Helpful Functions -------------------------------------------------------

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
  # This assumes the last column has the least entries, i.e. largest forecasting horizon
  y_hat <- y_hat[y_hat[, ncol(y_hat)] > 0, ] 
  new_idx <- intersect(index(y_hat), index(y))
  y_hat <- y_hat[index(y_hat) %in% new_idx,]
  y <- y[index(y) %in% new_idx]
  
  stopifnot(nrow(y_hat) == length(y))
  
  if (is.null(dim(y_hat))) {
    mse <- f_mse(y_hat, y)
    mae <- f_mae(y_hat, y)
    bias <- f_bias(y_hat, y)
    rmse <- f_rmse(y_hat, y) 
  } else {
    mse <- sapply(y_hat, f_mse, y)
    mae <- sapply(y_hat, f_mae, y)
    bias <- sapply(y_hat, f_bias, y)
    rmse <- sapply(y_hat, f_rmse, y)  
  }
  err_table <- cbind(mse, mae, rmse, bias)
  return(err_table)
}


# Section 1 ---------------------------------------------------------------
theme_update(plot.title = element_text(hjust = 0.5))
msgarch_color = "green"
garch_color = "red"
mssr1_color = "brown"
mssr2_color = "blue"

# TODO: Include scatter plot of returns/ Realized Vola
# TODO: Analyze Subsample performance
# TODO: Compare special state performance

data_dir = "C:\\Users\\Adria\\Documents\\Github Projects\\data\\markov-switching"
img_dir = "C:\\Users\\Adria\\Documents\\Github Projects\\markov-regime-switching\\LaTeX\\img"

start <- as.Date("1900-01-05")
end <- as.Date("2022-08-21")
df <- quantmod::getSymbols("^GSPC", src = "yahoo", from = start, to = end, auto.assign = FALSE)
price = df[, "GSPC.Close"]
colnames(price) <- "Price"
lret = diff(log(price))[-1,]
lret = lret - mean(lret) # Demeaning -> MSGARCH assumption
plot(lret)




lret = lret*100 # For numerical purposes

split_idx = round(length(lret)*0.8)
y = lret
train_y = lret[1:split_idx]
test_y = lret[(1+split_idx):length(lret)]

real_vol = abs(lret)
train_vol = real_vol[1:split_idx]
train_var = train_vol^2
test_vol = real_vol[(1+split_idx):length(lret)]
test_var = test_vol^2


bear_test_dates <- c(seq.Date(as.Date("2020-02-01"),as.Date("2020-04-01"),by="day"),
                     seq.Date(as.Date("2021-01-01"),as.Date("2023-01-01"),by="day"))
bull_test_dates <- index(test_vol)[!(index(test_vol) %in% bear_test_dates)]
  

corona_indices <- seq.Date(as.Date("2020-01-01"),as.Date("2022-05-01"),by="day")
housing_bubble <- seq.Date(as.Date("2008-01-01"),as.Date("2010-01-01"),by="day")
russia_war <- seq.Date(as.Date("2021-01-01"),as.Date("2023-01-01"),by="day")
oil <- seq.Date(as.Date("1986-01-01"),as.Date("1991-01-01"),by="day")


eighties <- seq.Date(as.Date("1986-01-01"),as.Date("1990-01-01"),by="day")


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





# Training the Models ----------------------------------------------------------

# Option: Execute this and skip rest of section
load("fitted_models.RData")
msgarch.fit.ml$spec = CreateSpec()
sr.fit[[1]]$spec = CreateSpec()
sr.fit[[2]]$spec = CreateSpec()
garch.fit.ml$spec = CreateSpec()
#---

set.seed(420)

# Training the models took approx. 1 min
start.time <- Sys.time()

# SINGLE-REGIME 
garch.n <- CreateSpec(variance.spec = list(model = c("gjrGARCH")),
                          distribution.spec = list(distribution = c("sged")))
summary(garch.n)
garch.fit.ml <- FitML(spec = garch.n, data = train_y)
summary(garch.fit.ml)

# MULTI-REGIME (non-switching shape)
ms2.garch.n <- CreateSpec(variance.spec = list(model = c("gjrGARCH", "gjrGARCH")),
                   distribution.spec = list(distribution = c("sged", "sged")))
summary(ms2.garch.n)
msgarch.fit.ml <- FitML(spec = ms2.garch.n, data = train_y)
summary(msgarch.fit.ml)
sr.fit <- ExtractStateFit(msgarch.fit.ml)

save(list=c("msgarch.fit.ml", "sr.fit", "garch.fit.ml"), file="fitted_models.RData")


# Preliminary Analysis ---------------------------------------------------------

# Make LaTeX table of parameters
params <- cbind(sr.fit[[1]]$par, sr.fit[[2]]$par, garch.fit.ml$par)
params <- as.data.frame(params)
colnames(params) <- c("MSSR1", "MSSR2", "GARCH")
xtable(params, label="tab:model-params", caption="Fitted parameters of the MSGARCH and the GARCH model.")

# Visualize densities
f <- function(x) { return(dsged(x, 0, 1, nu=params[5, 1], xi=params[4, 1])) }
g <- function(x) { return(dsged(x, 0, 1, nu=params[5, 2], xi=params[4, 2])) } # TODO: use params here
h <- function(x) { return(dsged(x, 0, 1, nu=params[5, 3], xi=params[4, 3])) }
x_grid <- seq(-5, 5, by=10/500)
y_values <- sapply(x_grid, f)
ggplot() + 
  geom_line(aes(x=x_grid, y=sapply(x_grid, f), color="MSSR1")) + 
  geom_line(aes(x=x_grid, y=sapply(x_grid, g), color="MSSR2" )) + 
  geom_line(aes(x=x_grid, y=sapply(x_grid, h), color="GARCH")) + 
  scale_color_manual(name = "Model", 
                     values = c("GARCH"=garch_color, "MSSR1"=mssr1_color, "MSSR2"=mssr2_color)) + 
  ggtitle("Fitted standardized distributions for the log-returns") +
  ylab("f(y)") +
  xlab("y")
ggsave(paste(img_dir, "\\FittedStandardizedDistr.pdf", sep=""))


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


# -------------------- In-sample volatility estimate comparison (state1)

msgarch_train_vola <- Volatility(msgarch.fit.ml) %>% fortify.zoo
garch_train_vola <- Volatility(garch.fit.ml) %>% fortify.zoo


p2 <- ggplot() + 
  geom_line(dat=msgarch_train_vola[msgarch_train_vola$Index %in% eighties, ], aes(x=Index, y=., colour="MSGARCH")) + 
  geom_line(dat=garch_train_vola[garch_train_vola$Index %in% eighties, ], aes(x=Index, y=., colour="GARCH")) + 
  ylab("Vola") + 
  ggtitle("Filtered in-sample Volatility during late eighties") + 
  scale_x_date(date_breaks = "2 years" , date_labels = "%y") + 
  scale_color_manual(name = "Model", values = c("GARCH"=garch_color, "MSGARCH"=msgarch_color))
p2
ggsave(paste(img_dir, "\\FiltVola_InSample_Eighties.pdf", sep=""))

#grid.arrange(p1, p2, nrow=2)



# Generating Point Forecasts ----------------------------------------------

horizons = 1:10 # currently only works for horizon=1!!
testing_anchor_points <- seq(1, (length(test_y)-max(horizons)), 1)


# Option: Load old forecasts and skip this section
load("point_forecasts.RData")





# -------------------- MSGARCH and GARCH
# models <- list(msgarch.fit.ml, garch.fit.ml)
# fcst <- xts(x = cbind(replicate(length(horizons)*length(models), rep(-9999, length(test_y)))), 
#                order.by = index(test_y))
# 
# names(fcst) = c(sapply(horizons, function(x) paste("MSGARCH_h", toString(x), sep="")),
#                 sapply(horizons, function(x) paste("GARCH_h", toString(x), sep="")))
# 

# for (mm in 1:length(models)) {
#   fcst[1, 1] <- predict(models[[mm]], nahead = 1, do.return.draw = FALSE)$vol[1]
#   for (dd in 1:(length(test_y)-1)) {
#     pred <- predict(models[[mm]], nahead = 1, do.return.draw = FALSE, newdata=test_y[1:dd])
#     fcst[dd+1, (mm-1)*(length(horizons))+1] = pred$vol[1] # Note that the dates in pred$vol aren't correct - they include weekend
#   } 
#   for (dd in testing_anchor_points) {
#     pred <- predict(models[[mm]], nahead = max(horizons), do.return.draw = FALSE, newdata=test_y[1:dd])
#     for (hh in 2:length(horizons)) {
#       fcst[dd+hh, (mm-1)*(length(horizons))+hh] = pred$vol[horizons[hh]]
#     }
#   }  
# }

# ---------------------- Single regime
fcst <- xts(x = cbind(replicate(length(horizons), rep(-1, length(test_y)))), 
            order.by = index(test_y))
names(fcst) = sapply(horizons, function(x) paste("GARCH_h", toString(x), sep=""))

fcst[1, 1] <- predict(garch.fit.ml, nahead = 1, do.return.draw = FALSE)$vol[1]

start.time <- Sys.time()

for (dd in 1:(length(test_y)-1)) {
  pred <- predict(garch.fit.ml, nahead = 1, do.return.draw = FALSE, newdata=test_y[1:dd])
  fcst[dd+1, 1] = pred$vol[1] # Note that the dates in pred$vol aren't correct - they include weekend
} 
for (dd in testing_anchor_points) {
  pred <- predict(garch.fit.ml, nahead = max(horizons), do.return.draw = FALSE, newdata=test_y[1:dd])
  for (hh in 2:length(horizons)) {
    fcst[dd+hh, hh] = pred$vol[horizons[hh]]
  }
  fcst
}

garch_fcst <- fcst


# --------------------- MSGARCH SINGLE-REGIMES
# SR FIT 1
model <- sr.fit[[1]]
fcst <- xts(x = cbind(rep(-1, length(test_y))), 
            order.by = index(test_y))
names(fcst) = "MSSR1_h1"

fcst[1, 1] <- predict(model, nahead = 1, do.return.draw = FALSE)$vol[1]
for (dd in 1:(length(test_y)-1)) {
  pred <- predict(model, nahead = 1, do.return.draw = FALSE, newdata=test_y[1:dd])
  fcst[dd+1, 1] = pred$vol[1] # Note that the dates in pred$vol aren't correct - they include weekend
} 
sr1_fcst <- fcst

# SR FIT 2
model <- sr.fit[[2]]
fcst <- xts(x = cbind(rep(-1, length(test_y))), 
            order.by = index(test_y))
names(fcst) = "MSSR2_h1"

fcst[1, 1] <- predict(model, nahead = 1, do.return.draw = FALSE)$vol[1]
for (dd in 1:(length(test_y)-1)) {
  pred <- predict(model, nahead = 1, do.return.draw = FALSE, newdata=test_y[1:dd])
  fcst[dd+1, 1] = pred$vol[1] # Note that the dates in pred$vol aren't correct - they include weekend
} 
sr2_fcst <- fcst

# ----------------------- MULTI-REGIME

fcst <- xts(x = cbind(replicate(length(horizons), rep(-1, length(test_y)))), 
            order.by = index(test_y))
names(fcst) = sapply(horizons, function(x) paste("MSGARCH_h", toString(x), sep=""))

fcst[1, 1] <- predict(msgarch.fit.ml, nahead = 1, do.return.draw = FALSE)$vol[1]
for (dd in 1:(length(test_y)-1)) {
  pred <- predict(msgarch.fit.ml, nahead = 1, do.return.draw = FALSE, newdata=test_y[1:dd])
  fcst[dd+1, 1] = pred$vol[1] # Note that the dates in pred$vol aren't correct - they include weekend
} 
for (dd in testing_anchor_points) {
  pred <- predict(msgarch.fit.ml, nahead = max(horizons), do.return.draw = FALSE, newdata=test_y[1:dd])
  for (hh in 2:length(horizons)) {
    fcst[dd+hh, hh] = pred$vol[horizons[hh]]
  }
}
msgarch_fcst <- fcst

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

save(list=c("msgarch_fcst", "garch_fcst", "sr1_fcst", "sr2_fcst", # Point Forecasts
           "train_vol", "test_vol"), file="point_forecasts.RData")

# Point Forecast Evaluation
# Point Forecast Evaluation - Out-of-Sample - single-step --------
msgarch_err <- get_err_table(msgarch_fcst[,1]^2, test_var)
garch_err <- get_err_table(garch_fcst[,1]^2, test_var)
sr1_err <- get_err_table(sr1_fcst^2, test_var)
sr2_err <- get_err_table(sr1_fcst^2, test_var)
avg_err <- get_err_table(mean(train_var), test_var)
rownames(avg_err) = "Avg"
oos_eval_1h <- rbind(msgarch_err, garch_err, sr1_err, sr2_err, avg_err)

oos_eval_1h <- as.data.frame(oos_eval_1h)
oos_eval_1h
rownames(oos_eval_1h) <- c("MSGARCH_h1", "GARCH_h1", "MSSR1_h1", "MSSR_h1", "Avg")
xtable(oos_eval_1h, digits=4, label="lab:oos_eval_1h", 
       caption="Out-of-sample error measurements for forecasting horizon h=1")


# - Diebold-Mariano Test
dm.test(msgarch_fcst[,1]^2, garch_fcst[,1]^2, h=1, power = 1)
dm.test(msgarch_fcst[,1]^2, garch_fcst[,1]^2, h=1, power = 2)

# - Murphy Diagram for one-day-ahead point forecasts
murphydiagram(as.vector(msgarch_fcst[,1]), as.vector(garch_fcst[,1]), as.vector(test_vol$Price), labels=c("MSGARCH", "GARCH"))

# Point Forecast Evaluation - Out-of-Sample - multi-step --------
avg_fcst <- test_var
avg_fcst[index(avg_fcst)] = mean(train_var)

msgarch_mserr <- get_err_table(msgarch_fcst^2, test_var)
garch_mserr <- get_err_table(garch_fcst^2, test_var)
avg_err <- get_err_table(avg_fcst, test_var)
rownames(avg_err) = "Avg"
oos_eval_multi_step <- rbind(msgarch_mserr, garch_mserr, avg_err)

ggplot() + 
  geom_line(aes(x=1:10, y=msgarch_mserr[, 1], color="MSGARCH")) +
  geom_line(aes(x=1:10, y=garch_mserr[, 1], color="GARCH")) + 
  scale_color_manual(name = "Model", values = c("GARCH"=garch_color, "MSGARCH"=msgarch_color)) + 
  scale_x_continuous(breaks=seq(1,10,1)) +
  ylab("MSE") + 
  xlab("Forecasting Horizon") + 
  ggtitle("MSE for different Forecasting Horizons")

# Point Forecast Evaluation - In-Sample -------------------------------
bip <- 100 # burn_in_period
msgarch_train_vola <- Volatility(msgarch.fit.ml)
garch_train_vola <- Volatility(garch.fit.ml)
mssr1_train_vola <- Volatility(sr.fit[[1]])
mssr2_train_vola <- Volatility(sr.fit[[2]])
partial_avg <- sqrt(cumsum(train_var)/seq_along(train_var))$Price %>% as.zooreg

forecasts <- list(msgarch_train_vola, garch_train_vola, mssr1_train_vola, mssr2_train_vola, partial_avg)
err_tables <- list()
for (ff in 1:length(forecasts)) {
  err_tables[[ff]] <- get_err_table(forecasts[[ff]][bip:length(train_vol),]^2, train_vol[bip:length(train_vol),]^2 %>% as.zooreg)
}
is_eval_1h <- do.call("rbind", err_tables)
is_eval_1h <- as.data.frame(is_eval_1h)
rownames(is_eval_1h) <- c("MSGARCH_h1", "GARCH_h1", "MSSR1_h1", "MSSR_h1", "Avg")
is_eval_1h
xtable(is_eval_1h, digits=4)
# Note: average has bias, because we start with the 1929 depression...

# Point Forecast Evaluation - Eighties Case Study ------------------------------
actual <- train_var[index(train_vol) %in% eighties] %>% as.zooreg

ms_garch_eighties <- msgarch_train_vola[index(msgarch_train_vola) %in% eighties, 2]
garch_eighties <- garch_train_vola[index(garch_train_vola) %in% eighties, 2]
mssr1_eighties <- mssr1_train_vola[index(mssr1_train_vola) %in% eighties, 2]
mssr2_eighties <- mssr2_train_vola[index(mssr2_train_vola) %in% eighties, 2]
start_of_eighties_idx <- match(as.Date(eighties[2]), as.Date(index(train_vol)))
avg_eighties <- mean(train_var[1:start_of_eighties_idx]) %>% sqrt

forecasts <- list(ms_garch_eighties, garch_eighties, mssr1_eighties, mssr2_eighties, avg_eighties)
err_tables <- list()
for (ff in 1:length(forecasts)) {
  err_tables[[ff]] <- get_err_table(forecasts[[ff]]^2, actual)
}

eighties_err_tabl <- do.call("rbind", err_tables)
rownames(eighties_err_tabl) <- c("MSGARCH", "GARCH", "MSSR1", "MSSR2", "Avg")
eighties_err_tabl


# Scenario Analysis and Visualization -------------------------------------


# # Probabilities for out-of-sample period
# dummy_ms2.garch.n <- CreateSpec(variance.spec = list(model = c("gjrGARCH", "gjrGARCH")),
#                           distribution.spec = list(distribution = c("sstd", "sstd")),
#                         constraint.spec = list(fixed=msgarch.fit.ml$par))
# summary(dummy_ms2.garch.n)
# dummy_msgarch.fit.ml <- FitML(spec = dummy_ms2.garch.n, data = train_y)
# summary(dummy_msgarch.fit.ml)

################## In-Sample
# In-sample volatility

msgarch_train_vola <- Volatility(msgarch.fit.ml) %>% fortify.zoo
garch_train_vola <- Volatility(garch.fit.ml) %>% fortify.zoo


p2 <- ggplot(msgarch_train_vola) + geom_line(aes(x=Index, y=.)) + 
  ylab("Vola") + 
  ggtitle("Filtered in-sample Volatility") + 
  scale_x_date(date_breaks = "2 years" , date_labels = "%y")

p2
#ggsave(paste(img_dir, "\\FiltVola_InSample.pdf", sep=""))



##################### Out-Of-Sample  

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


# CORONA POINT FORECAST - MSGARCH Decomposition
p1 <- ggplot() + 
  geom_line(data=sr1_fcst[index(msgarch_fcst) %in% corona_indices, ], aes(x=Index, y=MSSR1_h1, colour="mssr1")) + 
  geom_line(data=sr2_fcst[index(msgarch_fcst) %in% corona_indices, ], aes(x=Index, y=MSSR2_h1, colour="mssr2")) + 
  geom_line(data=garch_fcst[index(msgarch_fcst) %in% corona_indices, ], aes(x=Index, y=GARCH_h1, colour="garch")) +
  scale_color_manual(name = "Model", values = c("garch"=garch_color, "mssr1"="brown", "mssr2"="green")) + 
  scale_x_date(date_breaks = "2 years" , date_labels = "%y") + 
  xlab("Dates (Year)") + 
  ylab("Prob") + 
  ggtitle("1-day-ahead volatility forecasts during the Corona Crash")
p1
ggsave(paste(img_dir, "\\CoronaCrash_MSGARCH_Decomposition.pdf", sep=""))

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










