# Import Dataset
library("quantmod")

start <- as.Date("1950-01-01")
end <- as.Date("2022-08-21")
df <- quantmod::getSymbols("^SP500TR", src = "yahoo", from = start, to = end, auto.assign = FALSE)
price = df[, "SP500TR.Close"]
lret = diff(log(price))[-1,]
plot(lret)


split_idx = round(length(lret)*0.9)
train_y = lret[1:split_idx]
test_y = lret[(1+split_idx):length(lret)]

real_vola = abs(lret*100)



#HMM
library("MSGARCH")

# SINGLE REGIME


# MULTI-REGIME (non-switching shape)
ms2.garch.n <- CreateSpec(variance.spec = list(model = c("tGARCH", "tGARCH")),
                   distribution.spec = list(distribution = c("sged", "sged")))
summary(ms2.garch.n)
fit.ml <- FitML(spec = ms2.garch.n, data = train_y*100)
summary(fit.ml)

fit.mcmc <- FitMCMC(spec = ms2.garch.n, data = train_y)
summary(fit.mcmc)

# Forecasting
last_date = 10
pred <- predict(fit.ml, nahead = 5, do.return.draw = TRUE, newdata=test_y[0:last_date])

date = index(pred$draw[1, ])
date
lret[date]


pred <- predict(fit.mcmc, nahead = 1, do.return.draw = FALSE)
pred




