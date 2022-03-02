import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from os import walk

def find_R1(file):
    return int(file[file.find('RA')+2:file.find('RB')])

def find_R2(file):
    return float(file[file.find('RB')+2:-4])

def find_M(file):
    return int(file[file.find('M')+1:file.find('R')])

def find_NB(file):
    return int(file[file.find('NB')+2:file.find('R')])

def plot_data_RIE(M, R1, R2, folder, file, colors, color_idx, ax, condition, label_func=None, x_func=None, y_func=None):
    if condition(file, M,R1,R2):
        df = pd.read_csv(f'{folder}/{file}', index_col=0)
        delta = df['DELTAS'].to_numpy()
        MSE = df['y_MMSEs'].to_numpy()
        if x_func == None and y_func == None:
            if label_func == None:
                ax.plot(delta, MSE, marker='.', lw=0, color=colors[color_idx])
            else:
                ax.plot(delta, MSE, marker='.', label=label_func(file), lw=0, color=colors[color_idx])
            return color_idx + 1
        elif x_func == None:
            if label_func == None:
                ax.plot(delta, y_func(MSE), marker='.', lw=0, color=colors[color_idx])
            else:
                ax.plot(delta, y_func(MSE), marker='.', label=label_func(file), lw=0, color=colors[color_idx])
            return color_idx + 1
        elif y_func == None:
            if label_func == None:
                ax.plot(x_func(delta), MSE, marker='.', lw=0, color=colors[color_idx])
            else:
                ax.plot(x_func(delta), MSE, marker='.', label=label_func(file), lw=0, color=colors[color_idx])
            return color_idx + 1
        else:
            if label_func == None:
                ax.plot(x_func(delta), y_func(MSE), marker='.', lw=0, color=colors[color_idx])
            else:
                ax.plot(x_func(delta), y_func(MSE), marker='.', label=label_func(file), lw=0, color=colors[color_idx])
            return color_idx + 1
    return color_idx


def plot_data_MMSE(NB, R1, R2, folder, file, colors, color_idx, ax, condition, label_func=None, x_func=None, y_func=None):
    if condition(file, NB,R1,R2):
        df = pd.read_csv(f'{folder}/{file}', index_col=0)
        delta = df['DELTAS'].to_numpy()
        MSE = df['y_MMSEs'].to_numpy()
        if x_func == None and y_func == None:
            if label_func == None:
                ax.plot(delta, MSE, color=colors[color_idx])
            else:
                ax.plot(delta, MSE, label=label_func(file), color=colors[color_idx])
            return color_idx + 1
        elif x_func == None:
            if label_func == None:
                ax.plot(delta, y_func(MSE), color=colors[color_idx])
            else:
                ax.plot(delta, y_func(MSE), label=label_func(file), color=colors[color_idx])
            return color_idx + 1
        elif y_func == None:
            if label_func == None:
                ax.plot(x_func(delta), MSE, color=colors[color_idx])
            else:
                ax.plot(x_func(delta), MSE, label=label_func(file), color=colors[color_idx])
            return color_idx + 1
        else:
            if label_func == None:
                ax.plot(x_func(delta), y_func(MSE), color=colors[color_idx])
            else:
                ax.plot(x_func(delta), y_func(MSE), label=label_func(file), color=colors[color_idx])
            return color_idx + 1
    return color_idx

def file_select(folder, order_func):
    filenames = np.array(next(walk(folder), (None, None, []))[2])
    order_parameter = np.zeros(len(filenames))
    for i in range(len(filenames)):
        order_parameter[i] = order_func(filenames[i])
        order = np.argsort(order_parameter)
    return filenames[order]

def iterate_low_rank(init=0.001, tol=1e-9, max_steps = 10000, *, delta: float, R1: float):
    u = np.zeros(max_steps)
    v = np.zeros(max_steps)
    u[0] = init
    v[0] = init
    
    tconv = 0
    for t in range(max_steps-1):
        tconv += 1
        u[t+1] = R1 * v[t] / (delta + R1 * v[t])
        v[t+1] = u[t] / (delta + u[t])
        if np.abs(u[t+1]*v[t+1] - u[t]*v[t]) < tol:
            break

    return u[tconv]*v[tconv]

def low_rank(delta_min_exp, delta_max_exp, R1):
    delta_list = np.logspace(delta_min_exp, delta_max_exp, 2000)
    result = np.zeros_like(delta_list)
    for i,delta in enumerate(delta_list):
        m_star = iterate_low_rank(delta = delta, R1 = R1)
        result[i] = 1-m_star
    return delta_list, result
