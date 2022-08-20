from functools import partial

from config import data_dir
import pandas as pd
import os
from scipy.stats import norm
import numpy as np
from itertools import product
import time
# Diebold et al. 1994

Y = pd.read_pickle(os.path.join(data_dir, f"LOG_RETURNS.pkl"))
X = pd.read_pickle(os.path.join(data_dir, f"X.pkl"))
assert X.index.equals(Y.index)

# Parameters etc.
T = len(Y)
#theta
beta0 = (-0.01, -0.01)
beta1 = (0.01, 0.01)

# probability of state1 being 0
p_first_state_0 = 0.8
p_first_state = {0: p_first_state_0, 1: 1-p_first_state_0}


def conditional_density(y, s: bool, alpha0, alpha1):
    # s=0 if state is 0 and s=1 if state is 1
    # phi0 is the parameter  vector for state 0
    # phi1 is the parameter vector for state 1
    if s == 1:
        return norm.pdf(y, loc=alpha1)
    else:
        return norm.pdf(y, loc=alpha0)


def transition_probabilities(x, beta0, beta1):
    p11 = 1 / (1 + np.exp(x.dot(beta0)))
    p22 = 1 / (1 + np.exp(x.dot(beta1)))
    p12 = 1 - p11
    p21 = 1 - p22
    return np.array([p11, p12, p21, p22])


# 1. Calculate sequence of filtered marginal densities and trans. probs.
FMYD1 = conditional_density(Y, 0, 0.5, -0.5)
FMYD2 = conditional_density(Y, 1, 0.5, -0.5)
FMYD_df = pd.DataFrame({0:FMYD1, 1:FMYD2}, index=Y.index) # FMYD.loc[t, <state_t>]
FMYD = FMYD_df.to_numpy()


TPM = np.apply_along_axis(partial(transition_probabilities, beta0=beta0, beta1=beta1), 1, X)  # First row is P[s(2)|s(1)], if time starts at 1
TPM = TPM.reshape(TPM.shape[0], 2, 2)
TPM_df = pd.DataFrame(index=X.index, columns=pd.MultiIndex.from_product([[0,1], [0,1]]), data=TPM.reshape(TPM.shape[0], 4))
assert (TPM_df.sum(axis=1) - 2 < 0.001).all()
# TPM.loc[t, (<state_t>, <state_t+1>)])


# 2. Calculate Filtered joint state probabilities
# 2a. Calculate joint conditional distribution of (y_t, s_t, s_{t-1})
def joint_cond_distr(tpm: np.ndarray, fmyd, p_first_state):
    JCD = np.zeros(shape=(tpm.shape[0],2,2))

    for s1, s2 in product([0, 1], [0, 1]):
        # JCD[1] is is time 2
        # JCS[0] is meaningless
        JCD[1][s1][s2] = fmyd[1][s2]*tpm[0][s1][s2]*p_first_state[s1]
    
    for i in range(2, tpm.shape[0]):
        for s1, s2 in pd.MultiIndex.from_product([[0,1], [0,1]]):
            for s0 in range(2):
                filtered_prob = JCD[i-1]/JCD[i-1].sum().sum()
                JCD[i][s1][s2] += fmyd[i][s2] * tpm[i-1][s1][s2] \
                    * filtered_prob[s0][s1]
    return JCD

# With Pandas 10 sec
# Wtih Numpy 0.88
start_time = time.time()

JCD = joint_cond_distr(TPM, FMYD, p_first_state)
JCD_df = pd.DataFrame(index=Y.index, columns=pd.MultiIndex.from_product([[0,1], [0,1]]), data=JCD.reshape(JCD.shape[0], 4))

end_time = time.time()
#print(end_time - start_time)
#print(JCD_df)


# 2b. calculate conditional likelihood of y_t (one number)
COND_LIK = JCD_df.sum(axis=1)
#print(COND_LIK.describe())

# 2c. time-t filtered state probabilities
FILT_STATE_PROB = JCD_df.divide(JCD_df.sum(axis=1), axis=0)
assert ((FILT_STATE_PROB[1:].sum(axis=1) - 1) < 0.001).all()
# FILT_STATE_PROB.loc[<t>, (state_t-1, state_t)]


# GIVEN
def calculate_smoothed_joint_state_probs():
    SJSP = np.ndarray(shape=(TPM.shape[0], 2, 2))  # SJSP_t[s_prev][s_next]
    for t in range(2,T):
        s = time.time()
        for s_t, s_tm1 in pd.MultiIndex.from_product([[0,1], [0,1]]):
            # 3 Calculate the smoothed joint state probabilities
            # 3a.
            SJSP_t = np.zeros(shape=(2, 2))
            for s_tp1 in [0, 1]:
                SJSP_t[s_t][s_tp1] += FMYD[t+1][s_tp1] * TPM[t][s_t][s_tp1] * \
                    FILT_STATE_PROB.loc[Y.index[t], (s_tm1, s_t)] / COND_LIK[t+1]

            for tau in range(t+2, T):
                SJSP_new = np.zeros(shape=(2, 2))
                for s_tau, s_taum1 in pd.MultiIndex.from_product([[0,1], [0,1]]):
                    SJSP_new[s_taum1][s_tau] = (FMYD[tau][s_tau] * TPM[tau-1][s_taum1][s_tau] * (SJSP_t[0][s_taum1] + SJSP_t[1][s_taum1])) / COND_LIK[tau]
                SJSP_t = SJSP_new

            # 3b.
            # Smoothed joint state probability
            SJSP[t][s_tm1][s_t] = SJSP_t.sum().sum()
        e = time.time()
        print(e-s)
    return SJSP


SJSP = calculate_smoothed_joint_state_probs()
print(SJSP[2])
print(SJSP[2].sum().sum())
