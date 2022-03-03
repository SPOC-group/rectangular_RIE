import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plot_tools as p_t

R1 = 1
M = 2000
R2 = 0.005


# fig, ax = plt.subplots(figsize=(5.4,4))
plt.figure(figsize=(5.4,3.5))
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 14

folder_RIE = 'data/RIE'

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

R1 = 1
delta_list, MSE = p_t.low_rank(delta_min_exp=-3, delta_max_exp=2, R1=R1)
plt.plot(delta_list, MSE, color='black', label='Low Rank')
plt.axvline(x=np.sqrt(R1),ymax=1,ymin=0,alpha=.3,color=colors[0])


R1 = 2
delta_list, MSE = p_t.low_rank(delta_min_exp=-3, delta_max_exp=2, R1=R1)
plt.plot(delta_list, MSE, color='black')
plt.axvline(x=np.sqrt(R1),ymax=1,ymin=0,alpha=.3,color=colors[1])

R1 = 5
delta_list, MSE = p_t.low_rank(delta_min_exp=-3, delta_max_exp=2, R1=R1)
plt.plot(delta_list, MSE, color='black')
plt.axvline(x=np.sqrt(R1),ymax=1,ymin=0,alpha=.3,color=colors[2])


def condition_RIE(file, M,R1,R2):
    return p_t.find_R2(file) == R2 and p_t.find_M(file) == M

def x_func(x):
    return x * R2

def label_func(file):
    return f'$R_1 = {p_t.find_R1(file)}$'

color_idx = 0
filenames = p_t.file_select(folder_RIE, p_t.find_R1)
for i, file in enumerate(filenames):
    color_idx = p_t.plot_data_RIE(M, R1, R2, folder_RIE, file, colors, color_idx, plt, condition_RIE, x_func=x_func, label_func=label_func)

plt.legend(frameon=False, loc='upper left')
plt.title(f'$R_2 = {R2}$')
plt.xscale('log')
plt.ylabel('Y-MSE')
plt.xlabel('$\Delta R_2$')
plt.ylim((-.05,1.05))
plt.xlim((1e-3,1e2))
plt.tight_layout()
plt.savefig('low_rank_3.pdf')
plt.show()