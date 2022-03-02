import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plot_tools as p_t
import pandas as pd

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 14
fig, ax1 = plt.subplots(figsize=(5.4,3.5))

R1 = 1
M = 2000
NB = 131072
R2 = 2

folder_RIE = 'data/RIE'
folder_MMSE = 'data/MMSE'


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def label_func(file):
    return f'$R_1 = {p_t.find_R1(file)}$'

def condition_RIE(file, M, R1, R2):
    return p_t.find_R2(file) == R2 and p_t.find_M(file) == M

def condition_MMSE(file, NB, R1, R2):
    return p_t.find_NB(file) >= NB and p_t.find_R2(file) == R2

color_idx = 0
filenames = p_t.file_select(folder_RIE, p_t.find_R1)
for i, file in enumerate(filenames):
    color_idx = p_t.plot_data_RIE(M, R1, R2, folder_RIE, file, colors, color_idx, ax1, condition_RIE)

color_idx = 0
filenames = p_t.file_select(folder_MMSE, p_t.find_R1)
for i, file in enumerate(filenames):
    color_idx = p_t.plot_data_MMSE(NB, R1, R2, folder_MMSE, file, colors, color_idx, ax1, condition_MMSE, label_func=label_func)

delta_plot = np.logspace(-3,2,1024)
ax1.plot(delta_plot, delta_plot/(1+delta_plot), color='black', label='Scalar')

ax1.set_title(f'$R_2 = {R2}$')
ax1.set_xscale('log')
plt.ylabel('Y-MSE')
ax1.set_xlabel('$\Delta$')
ax1.set_ylim((-.05,1.05))
plt.legend(frameon=False, loc='lower right')
plt.tight_layout()

## Second plot
left, bottom, width, height = [0.22, 0.48, 0.32, 0.3]
mpl.rcParams['font.size'] = 10
ax2 = fig.add_axes([left, bottom, width, height])
ax2.set_title("$\mathrm{MMSE} / \mathrm{MMSE}_{\mathrm{scalar}}$")

def y_func(x, folder, file):
    df = pd.read_csv(f'{folder}/{file}', index_col=0)
    delta = df['DELTAS'].to_numpy()
    return x / (delta/(1+delta))


color_idx = 0
filenames = p_t.file_select(folder_RIE, p_t.find_R1)
for i, file in enumerate(filenames):
    color_idx = p_t.plot_data_RIE(M, R1, R2, folder_RIE, file, colors, color_idx, ax2, condition_RIE, y_func = lambda l: y_func(l, folder_RIE, file))


color_idx = 0
filenames = p_t.file_select(folder_MMSE, p_t.find_R1)
for i, file in enumerate(filenames):
    color_idx = p_t.plot_data_MMSE(NB, R1, R2, folder_MMSE, file, colors, color_idx, ax2, condition_MMSE,  y_func = lambda l: y_func(l, folder_MMSE, file))

ax2.axhline(y=1, color='black', label='Scalar')


ax2.set_xscale('log')
ax2.set_ylim(.895,1.01)
ax2.set_xlim(1e-3,10)
fig.savefig('high_rank.pdf')
plt.show()
