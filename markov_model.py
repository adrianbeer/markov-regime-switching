from config import data_dir
import pandas as pd
import os
from scipy.stats import norm
import numpy as np
from itertools import product
import time
# Diebold et al. 1994

Y = pd.read_pickle(os.path.join(data_dir, f"Y.pkl"))
X = pd.read_pickle(os.path.join(data_dir, f"X.pkl"))
assert X.index.equals(Y.index)

# Parameters etc.
T = len(Y)
#theta
beta = (1, 1)

# probability of state1 being 0
p_first_state_0 = 0.8
p_first_state = {0: p_first_state_0, 1: 1-p_first_state_0}

def conditional_density(y, s: bool):
    # TODO implement
    return norm.pdf(y, loc=s)

def transition_probabilities(x):
    p11 = 1 / (1 + np.exp(x.dot(beta)))
    p22 = 1 / (1 + np.exp(x.dot(beta)))
    p12 = 1 - p11
    p21 = 1 - p12
    return np.array([p11, p12, p21, p22])


# 1. Calculate sequence of filtered marginal densities and trans. probs.
FMYD1 = conditional_density(Y, 0.5)
FMYD2 = conditional_density(Y, -0.5)
FMYD = pd.DataFrame({0:FMYD1, 1:FMYD2}, index=Y.index)
# FMYD.loc[t, <state_t>]

TPM = np.apply_along_axis(transition_probabilities, 1, X)
TPM = TPM.reshape(TPM.shape[0], 2, 2)
TPM_df = pd.DataFrame(index=X.index, columns=pd.MultiIndex.from_product([[0,1], [0,1]]), data=TPM.reshape(TPM.shape[0], 4))
# TPM.loc[t, (<state_t>, <state_t+1>)])


# 2. Calculate Filtered joint state probabilities
# 2a. Calculate joint conditional distribution of (y_t, s_t, s_{t-1})
def joint_cond_distr(y, tpm: np.ndarray, fmyd, p_first_state):
    JCD = np.zeros(shape=(y.shape[0],2,2))

    for s1, s2 in product([0, 1], [0, 1]):
        JCD[0][s1][s2] = fmyd.iloc[0, s1]*tpm[0][s1][s2]*p_first_state[s1]
    
    for i in range(1, len(y)):
        for s1, s2 in pd.MultiIndex.from_product([[0,1], [0,1]]):
            for k in range(2):
                filtered_prob = JCD[i-1]/JCD[i-1].sum().sum()
                JCD[i][s1][s2] += conditional_density(y.iloc[i], s1) * tpm[i-1][s1][s2] \
                    * filtered_prob[s1][s2]
    return JCD

# With Pandas 10 sec
# Wtih Numpy 1.78
start_time = time.time()

JCD = joint_cond_distr(Y, TPM, FMYD, p_first_state)
JCD_df = pd.DataFrame(index=Y.index, columns=pd.MultiIndex.from_product([[0,1], [0,1]]), data=JCD.reshape(JCD.shape[0], 4))

end_time = time.time()
print(end_time - start_time)
print(JCD)


#
# # 2b. calculate conditional likelihood of y_t (one number)(
# COND_LIK = pd.DataFrame(index=Y.index, data=0, columns=["Conditional Likelihood"])
# COND_LIK = JCD.sum(axis=1)
# #print(COND_LIK.describe())
#
# # 2c. time-t filtered state probabilities
# FILT_STATE_PROB = JCD.divide(JCD.sum(axis=1), axis=0)
# print(FILT_STATE_PROB.head())
#
# # 3 Calculate the smoothed joint state probabilities
# # 3a.
# s_t = 0
# s_tm1 = 0
#
# t=2
# SJSP = pd.DataFrame(index=Y.index, data=0, columns=pd.MultiIndex.from_product([[0,1], [0,1]]))
# SJSP.iloc[t+1] = conditional_density(Y[t+1], s)*TPM.loc[Y.index[t-1], (s, s_t)] * \
#     FILT_STATE_PROB.loc[Y.index[t], (s_tm1, s_t)]
# for tau in range(t+2, T+1):
#     # tau-1 ?
#     for s1, s0 in pd.MultiIndex.from_product([[0,1], [0,1]]):
#         # dont need a matrix here i think... just update the value each iteration
#         SJSP.loc[Y.index[tau], (s0,s1)] = conditional_density(Y[tau], s1)*TPM.loc[Y.index[tau], (s0, s1)] \
#         * SJSP.loc[Y.index[tau-1], (s0, s1)]