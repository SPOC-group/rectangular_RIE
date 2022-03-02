import numpy as np
import matplotlib.pyplot as plt
import g_functions as g_f

R1 = 2
R2 = .6
M = 500
Delta = .1

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


# Compute sample
S, Y = g_f.make_sample(parameters, Delta)

# Computer rho from theory
rho_theory = g_f.find_rho(parameters, Delta)

# Compute deoising function from theory
denoiser_plot = np.zeros(parameters["NB_POINTS"])
for (i_z, z) in enumerate(rho_theory["zs"]):
    denoiser_plot[i_z] = g_f.denoiser(z, parameters, Delta)



plt.hist(g_f.find_spectrum(Y), 80, density=True)
# plt.hist(g_f.find_spectrum(g_f.denoise_sample(Y, parameters, Delta)), 160, density=True)

plt.plot(rho_theory['zs'],rho_theory['rho'],color='red')
# plt.plot(rho_theory['zs'],denoiser_plot)
plt.title(f"R2 = {parameters['R2']}, R1 = {parameters['R1']}")
plt.ylabel("Frequency")
plt.xlabel("Singular value")
plt.show()