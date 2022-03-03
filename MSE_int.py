import numpy as np
import g_functions as g_f

R1 = 2
R2 = 1/5
M = 2000
DELTAS = np.logspace(-2,1,36)

NB_POINTS = 2**18
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

result = g_f.compute_MMSE_MPI(parameters)
g_f.post_processing_MMSE(result, parameters, MPI_used=True)