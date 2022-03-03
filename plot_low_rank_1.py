import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plot_tools as p_t

R1 = 1
M = 2000
NB = 131072
R2 = 0.2


# fig, ax = plt.subplots(figsize=(5.4,4))
plt.figure(figsize=(5.4,3.5))
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 14

folder_RIE = 'data/RIE'
folder_MMSE = 'data/MMSE'

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def condition_RIE(file, M,R1,R2):
    return p_t.find_R2(file) == R2 and p_t.find_M(file) == M

def condition_MMSE(file, NB,R1,R2):
    return p_t.find_NB(file) >= NB and p_t.find_R2(file) == R2

def label_func(file):
    return f'$R_1 = {p_t.find_R1(file)}$'

color_idx = 0
filenames = p_t.file_select(folder_RIE, p_t.find_R1)
for i, file in enumerate(filenames):
    pass
    color_idx = p_t.plot_data_RIE(M, R1, R2, folder_RIE, file, colors, color_idx, plt, condition_RIE)

color_idx = 0
filenames = p_t.file_select(folder_MMSE, p_t.find_R1)
for i, file in enumerate(filenames):
    pass
    color_idx = p_t.plot_data_MMSE(NB, R1, R2, folder_MMSE, file, colors, color_idx, plt, condition_MMSE, label_func=label_func)

delta_plot = np.logspace(-3,2,1024)
plt.plot(delta_plot, delta_plot/(1+delta_plot), color='black', label='Scalar')


plt.legend(frameon=False, loc='upper left')
plt.title(f'$R_2 = {R2}$')
plt.xscale('log')
plt.ylabel('Y-MSE')
plt.xlabel('$\Delta$')
plt.ylim((-.05,1.05))
plt.xlim((1e-3,1e2))
plt.tight_layout()
plt.savefig('low_rank_1.pdf')
plt.show()