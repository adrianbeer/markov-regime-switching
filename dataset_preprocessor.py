import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from config import data_dir

# Schiller S&P TR
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


data = (data.iloc[:, 1])# / data.iloc[:, 4]
data = pd.to_numeric(data)
SP_TR_df = data

print(np.log(SP_TR_df).diff().dropna().describe())
plt.hist(np.log(SP_TR_df).diff().dropna(), bins="scott")
plt.show()

plt.plot(np.log(SP_TR_df))
plt.show()

# FRED Data sets
# ----------------
dfs = []
for f_name in ["BAA", "AAA"]:
    path = os.path.join(data_dir, f"{f_name}.csv")
    df = pd.read_csv(path)
    df["DATE2"] = pd.to_datetime(df.pop("DATE"))
    df = df.set_index("DATE2")
    dfs.append(df)

# Feature engineering
# --------------------
FRED_df = pd.concat(dfs, axis=1, join="inner")
FRED_df["SPREAD"] = (FRED_df["BAA"] - FRED_df["AAA"])
FRED_df = FRED_df.drop(columns=["AAA"])
indicator_df = np.log(FRED_df).diff().dropna()

# Show data
# --------------
# f, ax = plt.subplots(1,1)
# ax.plot(np.log(SP_TR_df))
#
# ax2 = ax.twinx()
# ax2.plot(np.log(FRED_df), color='green', linestyle='dashed')
# plt.show()


# Principle Component analysis
# ----------------------------
plt.plot(indicator_df, linewidth=0.5)
plt.show()

pca = PCA(1)
pca.fit(indicator_df)
component = pd.Series(index=indicator_df.index, data=pca.transform(indicator_df).flatten())
length = len(component)

#plt.close()
plt.plot(component, label="1st PC", linewidth=0.5)
plt.plot(np.log(SP_TR_df).diff()[-length:], linewidth=0.5)
plt.axhline(0, color='red')
plt.legend()
plt.show()

print(np.corrcoef(np.log(SP_TR_df).diff()[-length:], component))

# Testing the Autoregressive assumption on the dynamic factor component
# ----------------------------------------------------------------------
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white

exog = np.vstack((np.repeat(1, len(component)-2), component[:-2], component[1:-1])).T

ar = sm.OLS(component[2:], exog).fit()
print(ar.params)
print(ar.pvalues)
plt.plot(ar.resid)
plt.show()

_, lm_p, _, f_p = het_white(ar.resid, exog)
print(lm_p)
print(f_p)

