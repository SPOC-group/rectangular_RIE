import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plot_tools as p_t

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 14
fig, ax = plt.subplots(figsize=(5.4,4))

R1 = 1
M = 2000
NB = 131072
R2 = 2

folder_RIE = 'data/RIE'
folder_MMSE = 'data/MMSE'


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def condition_RIE(file, M, R1, R2):
    return p_t.find_R1(file) == R1 and p_t.find_M(file) == M and p_t.find_R2(file) > 0.005 and p_t.find_R2(file) < 2

def condition_MMSE(file, NB, R1, R2):
    return p_t.find_NB(file) >= NB and p_t.find_R1(file) == R1 

def label_func(file):
    return f'$R_2 = {p_t.find_R2(file)}$'

color_idx = 0
filenames = p_t.file_select(folder_RIE, p_t.find_R2)
for i, file in enumerate(filenames):
    color_idx = p_t.plot_data_RIE(M, R1, R2, folder_RIE, file, colors, color_idx, ax, condition_RIE)

color_idx = 0
filenames = p_t.file_select(folder_MMSE, p_t.find_R2)
for i, file in enumerate(filenames):
    color_idx = p_t.plot_data_MMSE(NB, R1, R2, folder_MMSE, file, colors, color_idx, ax, condition_MMSE, label_func=label_func)

delta_plot = np.logspace(-3,2,1024)
ax.plot(delta_plot, delta_plot/(1+delta_plot), color='black', label='Scalar')

ax.set_title(f'$R_1 = {R1}$')
ax.set_xscale('log')
plt.ylabel('Y-MSE')
ax.set_xlabel('$\Delta$')
ax.set_ylim((-.05,1.05))
ax.set_xlim((1e-3,1e2))
plt.legend(frameon=False, loc='upper left')
plt.tight_layout()
fig.savefig('high_rank_approach.pdf')
plt.show()
