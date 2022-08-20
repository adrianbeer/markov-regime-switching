import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from config import data_dir

# Extract data from Shiller's dataset.
# ----------------
path = os.path.join(data_dir , "ie_data.xls")
xls = pd.ExcelFile(path, engine="xlrd")

data = pd.read_excel(xls, "Data", skiprows=7)
data.drop(index=data.index[-2:], inplace=True)  # Remove description at the bottom of Excel file

# Set DateTimeIndex
data.iloc[:, 0] = data.iloc[:, 0].astype("string")
data.iloc[:, 0] = data.iloc[:, 0].apply(lambda x: x.replace(".1", ".10") if len(x)==6 else x)
data.iloc[:, 0] = data.iloc[:, 0].apply(lambda x: str(x)[:7]+".01")
data['Datetime'] = pd.to_datetime(data.iloc[:, 0], format="%Y.%m.%d")
data = data.set_index('Datetime').sort_index()
# Data for some month-date represents average of that month and so it should be mapped to the last day of the month.
data = data.resample("M").last()


PRICE = pd.to_numeric(data.loc[:, "Price.1"])  # Real Total Return Prices
log_returns = np.log(PRICE).diff()


CPI = pd.to_numeric(data.loc[:, "CPI"])  # CPI

#INFLATION = CPI.pct_change()
#INFLATION.name = "Inflation"

print(log_returns.dropna().describe())
plt.hist(log_returns.dropna(), bins="scott")
plt.title("Distribution of S&P log returns")
plt.show()

plt.plot(np.log(PRICE))
plt.title("Log of Total Return S&P")
plt.show()

# FRED Data sets
# ----------------
dfs = []
for f_name in ["BAA", "AAA"]:
    path = os.path.join(data_dir, f"{f_name}.csv")
    df = pd.read_csv(path)
    df["DATE2"] = pd.to_datetime(df.pop("DATE"))
    df = df.set_index("DATE2")
    # The data points for monthly FRED time series initially "<M>.01", but they are only available at the end of the
    # month, so we need to reassign the dates to the end of the month.
    df = df.resample("M").last()
    dfs.append(df)
FRED_df = pd.concat(dfs, axis=1, join="inner")


# Merging Indicators (EXOG) and Feature Engineering
# -------------------------------------------------
EXOG = pd.concat([CPI, FRED_df], axis=1).dropna()
EXOG["SPREAD"] = (EXOG["BAA"] - EXOG["AAA"])
EXOG = EXOG.drop(columns=["AAA"])
EXOG = np.log(EXOG).diff().dropna()

# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit(EXOG).transform(EXOG)
EXOG = pd.DataFrame(data_scaled, columns=EXOG.columns, index=EXOG.index)

# Show data
# --------------
# f, ax = plt.subplots(1,1)
# ax.plot(np.log(PRICE))
#
# ax2 = ax.twinx()
# ax2.plot(np.log(EXOG), color='green', linestyle='dashed')
# plt.show()


# Principle Component analysis
# ----------------------------
plt.plot(EXOG, linewidth=0.5)
plt.title("Indicators")
plt.legend(EXOG.columns)
plt.show()

pca = PCA(1)
pca.fit(EXOG)
component = pd.Series(index=EXOG.index, data=pca.transform(EXOG).flatten())
length = len(component)

#plt.close()
plt.plot(component, label="First Principle Component", linewidth=0.5)
plt.plot(log_returns[-length:], linewidth=0.5, label="Log returns TR S&P")
plt.axhline(0, color='red')
plt.legend()
plt.show()

print(np.corrcoef(log_returns[-length:], component))

# Aligning exogeneous indicator with log_returns and export
common_idx = log_returns.index.intersection(component.index)
log_returns.loc[common_idx].to_pickle(os.path.join(data_dir, f"Y.pkl"))
X = component.loc[common_idx].copy().to_frame()
X.insert(0, "Intercept", 1)
X.to_pickle(os.path.join(data_dir, f"X.pkl"))


# Testing usefulness of the main PC



# Testing Forecastability of the main PC
# ----------------------------------------------------------------------
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white

ols_exog = np.vstack((np.repeat(1, len(component)-2), component[:-2], component[1:-1])).T

ar = sm.OLS(component[2:], ols_exog).fit()
print(ar.params)
print(ar.pvalues)
plt.plot(ar.resid)
plt.title("Residuals of a AR(1) estimation of the dynamic factor component")
plt.show()

# Check for heteroskedasticity and normality of the residuals.

_, lm_p, _, f_p = het_white(ar.resid, ols_exog)
print(lm_p)
print(f_p)

