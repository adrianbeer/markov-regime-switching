import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
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


data = (data.iloc[:, 1])/ data.iloc[:, 4]
data = pd.to_numeric(data)

print(data.tail(20))

f, ax = plt.subplots(1,1)
ax.plot(np.log(data))
f.show()

# EMRATIO Data set
# ----------------
path = os.path.join(data_dir, "EMRATIO.csv")
df = pd.read_csv(path)
df["DATE2"] = pd.to_datetime(df.pop("DATE"))
df = df.set_index("DATE2")

ax2 = ax.twinx()
ax2.plot(df, color='red')
f.show()


# Show data
# --------------
plt.show()
