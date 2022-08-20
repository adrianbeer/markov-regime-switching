from config import data_dir
import pandas as pd
import os
from hmmlearn import hmm
from matplotlib import pyplot as plt

Y = pd.read_pickle(os.path.join(data_dir, f"LOG_RETURNS.pkl")).to_frame()

remodel = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
remodel.fit(Y)

Z2 = remodel.predict(Y)

plt.plot(Y.cumsum())
plt.show()