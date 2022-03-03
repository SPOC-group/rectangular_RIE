import numpy as np
import pandas as pd
import g_functions as g_f

R1 = 20
R2 = 2
M = 2000

DELTAS = np.logspace(-3,2,64)

NB_POINTS = 2**10
EPSILON_IMAG = 1e-8

parameters = {
    'M'             : M,
    'R1'            : R1,
    'R2'            : R2,
    'DELTAS'        : DELTAS,
    'NB_POINTS'     : NB_POINTS,
    'EPSILON_IMAG'  : EPSILON_IMAG,
    'verbosity'     : 1,
    'ENSAMBLE'      : 'Wishart'
}

result = g_f.RIE_MSE(parameters)
pd.DataFrame(result).to_csv(f"data/RIE/M{M}RA{R1}RB{R2}.csv")