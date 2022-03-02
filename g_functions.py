import numpy as np
from scipy import integrate
from mpi4py import MPI
import matplotlib.pyplot as plt
import pandas as pd

def g_theo_MP(z, gamma):
    return  ((1-gamma) - z + np.sqrt((z-gamma-1)**2 - 4*gamma)) / (2*z*gamma)

def g_theo(z, gamma, delta):
    return z * g_theo_MP(z**2 * np.sqrt(gamma) / delta, gamma) * gamma**(1.5) / delta


def g_Y(z, parameters, Delta):
    #Computes the Stieltjes-transform of Y = S + sqrt(Delta) Z
    assert np.imag(z) < 0, "ERROR: We need Im[z] < 0 to compute g(z). Here Im[z] = "+str(np.imag(z))

    R1 = parameters["R1"]
    R2 = parameters["R2"]

    if parameters["ENSAMBLE"] == "Wishart":
        phi = R2/R1
        psi = R2
        eta = (1+Delta)*np.sqrt(R1)
        zeta = np.sqrt(R1)
        a0 = -psi**3
        a1 = psi * ( zeta * (psi - phi) + psi * ( eta * (phi - psi) + psi * z**2 ) )
        a2 = - zeta**2 * (phi - psi)**2 + zeta * ( eta * (phi - psi)**2 + psi * z**2 * (2 * phi - psi) ) - eta * psi**2 * z**2 * phi
        a3 = - zeta * z**2 * phi * ( 2 * zeta * psi - 2 * zeta * phi - 2 * eta * psi + 2 * eta * phi + psi * z**2 )
        a4 = zeta * z**4 * phi**2 * (eta - zeta)
        p = np.array([a4, a3, a2, a1, a0])
    
        solutions = z * np.roots(p)
        g = solutions[np.argmax(np.imag(solutions))] #Find the solution with largest imaginary part

    # assert np.imag(g)/np.pi > -1e-5, "ERROR: Negative imaginary part of g: "+str(round(g,10))
    return g

def edges_rho(parameters, Delta):
    R1 = parameters["R1"]
    R2 = parameters["R2"]
    
    A1 = R2
    A2 = R1
    
    a0 = -27*(-1 + A1)**2 *(A1- A2)**2 *(-1 + A2)**2
    a1 = -216*A1*(1+A1 +A2)**2 *(1-(1+A1)*(1+A2)) - 324*A1*(1-(1+A1)*(1+A2))**2 -162*A1*(1+A1 +A2)*(A1*(1+A1)+A2*(1+A2)+A1*A2*(A1 +A2))
    a2 = -27*A1**2*(1+A1 +A2)**2 -324*A1**2*(1-(1+A1)*(1+A2))
    a3 = -108 * A1**3

    p = np.array([a3, a2, a1, a0])
    rts = np.roots(p)
    signal = np.max(rts) / np.sqrt(R1)
    minsig = np.min(rts[rts != 0]) * np.sqrt(R1)
    noise = Delta / np.sqrt(R1) * (1 + np.sqrt(R1))**2
    res = [signal, noise]
    return np.sqrt(res)

def find_rho(parameters, Delta):
    NB_POINTS = int(parameters["NB_POINTS"]) + 1
    regular_grid, zs, step = False, None, 0

    lmax_S, lmax_Z = edges_rho(parameters, Delta)
    lmin_S = -lmax_S
    lmin_Z = -lmax_Z

    #We find a rough estimate of the edges of the bulk since lmax(A+B) <= lmax(A) + lmax(B) and lmin(A+B) >=  lmin(A) + lmin(B)
    min_estimate, max_estimate = lmin_S + lmin_Z, lmax_S + lmax_Z 
    
    #To avoid some issues
    min_estimate -= 0.01 
    max_estimate += 0.01 

    #Now we compute the full gs
    zs, step = np.linspace(min_estimate, max_estimate, num = NB_POINTS-1, retstep=True)
    regular_grid = True

    #For small Delta > 0 and alpha > 1, we add many new points around Delta = 0 to increase the precision of the integrals
    if parameters['R2'] < 1 and Delta > 0 and Delta < 1e-2*lmax_Z: #Then we add a lot of new points around t = 0, but the grid is no longer regular
        zs = np.concatenate((zs, np.linspace(lmin_Z, lmax_Z, num = int(NB_POINTS/2))))
        zs = np.sort(zs)
        zs = np.unique(zs) #Sorted and unique elements
        regular_grid = False

    #Now we compute the Stieltjes transform
    gs = np.zeros_like(zs) + 1j*np.zeros_like(zs)
    for (i_z, z) in enumerate(zs):
        gs[i_z] = g_Y(z - 1j*parameters["EPSILON_IMAG"], parameters, Delta)

    #From g we extract rho and V
    rho = np.imag(gs) / np.pi

    return {'step':step, 'zs':zs, 'rho':rho, 'regular_grid':regular_grid}

def denoiser(z, parameters, Delta):
    g = g_Y(z - 1j*parameters["EPSILON_IMAG"], parameters, Delta)
    v = np.real(g) + (parameters["R1"]-1)/(2*z)
    return z - 2*Delta*v/np.sqrt(parameters["R1"])

def obtain_sample_parameters(parameters):
    M = parameters["M"]
    R1 = parameters["R1"]
    R2 = parameters["R2"]
    N = int(np.ceil(R2 * M))
    P = int(np.ceil(R1 * M))
    
    assert(M <= P)

    return M,N,P

def make_sample(parameters, Delta):
    M,N,P = obtain_sample_parameters(parameters)

    UL = np.random.normal(0,1,size=(M,N))
    UR = np.random.normal(0,1,size=(N,P))

    signal = UL@UR / np.sqrt(N*np.sqrt(M*P))
    noise = np.random.normal(0,1,size=(M,P)) / (P*M)**(.25)

    Y = signal + np.sqrt(Delta)*noise
    return signal, Y

def find_spectrum(Y):
    U,d,V = np.linalg.svd(Y)
    d_sym = np.concatenate([-d,d])
    return d_sym

def find_spectrum_nonsym(Y):
    U,d,V = np.linalg.svd(Y)
    return d

def denoise_sample(Y, parameters, Delta):
    U,d,V = np.linalg.svd(Y)

    d_clean = np.zeros_like(d)
    for (i_d, d_iter) in enumerate(d):
        d_clean[i_d] = denoiser(d_iter, parameters, Delta)

    Y_clean = np.dot(U * d_clean, V[:len(d), :])
    return Y_clean

def MSE(signal, Y_clean):
    return np.sum((signal-Y_clean)**2) / np.sqrt(signal.shape[0]*signal.shape[1])

def RIE_MSE(parameters):
    DELTAS = parameters["DELTAS"]

    #The function computes all MSES for all Deltas in Deltas
    y_MMSEs = np.zeros_like(DELTAS)
    for (i_D, Delta) in enumerate(DELTAS):
        S, Y = make_sample(parameters, Delta)
        Y_clean = denoise_sample(Y, parameters, Delta)
        y_MMSEs[i_D] = MSE(S, Y_clean)  
        if parameters['verbosity'] >= 1:
            print(f"Delta = {Delta}: {y_MMSEs[i_D]}")
    return {'DELTAS':DELTAS, 'y_MMSEs':y_MMSEs}

def compute_MMSE(parameters):
    DELTAS = parameters["DELTAS"]

    y_MMSEs, dlog_potentials_Y, drect_potentials_Y = np.zeros_like(DELTAS), np.zeros_like(DELTAS), np.zeros_like(DELTAS)

    for (i_D, Delta) in enumerate(DELTAS):
        #Compute the differential of the log potential
        Delta_step = min(DELTAS[i_D]*1e-2, 1e-1)
        Delta_next = DELTAS[i_D] + Delta_step
        Delta_previous = DELTAS[i_D] - Delta_step

        solution = find_rho(parameters, Delta_next)
        step, zs, rho, regular_grid = solution['step'], solution['zs'], solution['rho'], solution['regular_grid']
        if regular_grid:
            one_integral = np.array([integrate.romb(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), dx = step) for z in zs]) #Integral over the last dimension 
            log_potential_next = integrate.romb(rho*one_integral, dx = step)
            rect_potential_next = integrate.romb(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(zs))), dx = step)
        else:
            one_integral = np.array([integrate.simpson(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), zs) for z in zs]) #Integral over the last dimension 
            log_potential_next = integrate.simpson(rho*one_integral, zs)
            rect_potential_next = integrate.simpson(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(zs))), zs)

        solution = find_rho(parameters, Delta_previous)
        step, zs, rho, regular_grid = solution['step'], solution['zs'], solution['rho'], solution['regular_grid']
        if regular_grid:
            one_integral = np.array([integrate.romb(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), dx = step) for z in zs]) #Integral over the last dimension 
            log_potential_previous = integrate.romb(rho*one_integral, dx = step)
            rect_potential_previous = integrate.romb(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(zs))), dx = step)
        else:
            one_integral = np.array([integrate.simpson(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), zs) for z in zs]) #Integral over the last dimension 
            log_potential_previous = integrate.simpson(rho*one_integral, zs)
            rect_potential_previous = integrate.simpson(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(zs))), zs)

        dlog_potentials_Y[i_D] = (log_potential_next - log_potential_previous)/(2*Delta_step)
        drect_potentials_Y[i_D] = (rect_potential_next - rect_potential_previous)/(2*Delta_step)
        
        y_MMSEs[i_D] = Delta - 2 * Delta**2 * (dlog_potentials_Y[i_D] + (parameters['R1']-1)*drect_potentials_Y[i_D]) / parameters['R1']
        
        print(f"Delta = {Delta}: {y_MMSEs[i_D]}")    
    
    return {'DELTAS':DELTAS, 'y_MMSEs':y_MMSEs}

def compute_MMSE_MPI(parameters):
    DELTAS = parameters["DELTAS"]


    size = len(DELTAS)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pool_size = comm.Get_size()

    partial = []
    j = 0
    for i in range(rank,size,pool_size):
        partial.append([])

        #Compute the differential of the log potential
        Delta_step = min(DELTAS[i]*1e-2, 1e-1)
        Delta_next = DELTAS[i] + Delta_step
        Delta_previous = DELTAS[i] - Delta_step

        solution = find_rho(parameters, Delta_next)
        step, zs, rho, regular_grid = solution['step'], solution['zs'], solution['rho'], solution['regular_grid']
        if regular_grid:
            one_integral = np.array([integrate.romb(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), dx = step) for z in zs]) #Integral over the last dimension 
            log_potential_next = integrate.romb(rho*one_integral, dx = step)
            rect_potential_next = integrate.romb(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(zs))), dx = step)
        else:
            one_integral = np.array([integrate.simpson(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), zs) for z in zs]) #Integral over the last dimension 
            log_potential_next = integrate.simpson(rho*one_integral, zs)
            rect_potential_next = integrate.simpson(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(zs))), zs)

        solution = find_rho(parameters, Delta_previous)
        step, zs, rho, regular_grid = solution['step'], solution['zs'], solution['rho'], solution['regular_grid']
        if regular_grid:
            one_integral = np.array([integrate.romb(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), dx = step) for z in zs]) #Integral over the last dimension 
            log_potential_previous = integrate.romb(rho*one_integral, dx = step)
            rect_potential_previous = integrate.romb(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(zs))), dx = step)
        else:
            one_integral = np.array([integrate.simpson(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), zs) for z in zs]) #Integral over the last dimension 
            log_potential_previous = integrate.simpson(rho*one_integral, zs)
            rect_potential_previous = integrate.simpson(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(zs))), zs)

        dlog_potential_Y = (log_potential_next - log_potential_previous)/(2*Delta_step)
        drect_potential_Y = (rect_potential_next - rect_potential_previous)/(2*Delta_step)
        
        y_MMSE = DELTAS[i] - 2 * DELTAS[i]**2 * (dlog_potential_Y + (parameters['R1']-1)*drect_potential_Y) / parameters['R1']
        
        print(f"Delta = {DELTAS[i]}: {y_MMSE}")    

        partial[j].append(y_MMSE)
        j += 1

    if rank == 0:
        results = [0 for _ in range(size)]

        j = 0
        for i in range(0,size,pool_size):
            results[i] = partial[j][0]
            j += 1

        for i in range(1,size):
            if i % pool_size != 0:
                results[i] = comm.recv(source = i % pool_size)[0]
        
        return {'DELTAS':DELTAS, 'y_MMSEs':results}

    else:
        for i in range(len(partial)):
            comm.send(partial[i], dest = 0)

def post_processing_MMSE(result, parameters, MPI_used = False):
    NB_POINTS = parameters['NB_POINTS']
    R1 = parameters['R1']
    R2 = parameters['R2']

    if MPI_used == False:
        pd.DataFrame(result).to_csv(f"data/MMSE/NB{NB_POINTS}RA{R1}RB{R2}.csv")

        plt.plot(result['DELTAS'], result['y_MMSEs'], marker='.')

        df = pd.read_pickle('DownloadMatytsin_wishart_alpha_5.0_log_NB_points_x_15_NB_POINTS_t_3000_log_scale_t_1.pkl')
        plt.plot(df['Deltas'], df['y_MMSEs'], label='Maillard (gamma = 1/5)')

        delta_list = parameters["DELTAS"]
        plt.plot(delta_list, delta_list/(1+delta_list), color='black', label='Scalar denoising')


        plt.xscale('log')
        plt.show()
    
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            pd.DataFrame(result).to_csv(f"data/MMSE/NB{NB_POINTS}RA{R1}RB{R2}.csv")

            plt.plot(result['DELTAS'], result['y_MMSEs'], marker='.')

            df = pd.read_pickle('DownloadMatytsin_wishart_alpha_5.0_log_NB_points_x_15_NB_POINTS_t_3000_log_scale_t_1.pkl')
            plt.plot(df['Deltas'], df['y_MMSEs'], label='Maillard (gamma = 1/5)')

            delta_list = parameters["DELTAS"]
            plt.plot(delta_list, delta_list/(1+delta_list), color='black', label='Scalar denoising')


            plt.xscale('log')
            plt.show()
