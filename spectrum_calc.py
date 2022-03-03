import pandas as pd
import g_functions as g_f

R1 = 2
R2 = .2
M = 5000

NB_POINTS = 2**10
EPSILON_IMAG = 1e-8

parameters = {
    'M'             : M,
    'R1'            : R1,
    'R2'            : R2,
    'NB_POINTS'     : NB_POINTS,
    'EPSILON_IMAG'  : EPSILON_IMAG,
    'verbosity'     : 1,
    'ENSAMBLE'      : 'Wishart'
}



Delta = 30
S, Y = g_f.make_sample(parameters, Delta)
rho_theory = g_f.find_rho(parameters, Delta)
df = pd.DataFrame(data = {"Y" : g_f.find_spectrum_nonsym(Y), "S" : g_f.find_spectrum_nonsym(S), "Noise" : g_f.find_spectrum_nonsym(Y-S), "Denoised" : g_f.find_spectrum_nonsym(g_f.denoise_sample(Y, parameters, Delta))})
df.to_csv(f"DATA/SPECTRA/M{M}R1{R1}R2{R2}D{Delta}.csv")