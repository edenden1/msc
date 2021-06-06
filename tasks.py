import numpy as np
import matplotlib.pyplot as plt
from project import Model
import os
import pickle
from scipy.optimize import minimize
import json


def plot_Sigma2(w):
    """
    Plots the partial and informed entropy productions as in Figure 4 b (2017 Hierarchical bounds...)

    :param w: The rate matrix
    :return:
    """
    pps_list = []
    ips_list = []
    Sigma_list = []
    x_list = np.linspace(-6, 6, 120)
    real_to_observed = {i: i for i in range(w.shape[0])}
    for x in x_list:
        w_tmp = w.copy()
        np.fill_diagonal(w_tmp, 0)
        w_tmp[0, 1] = w[0, 1] * np.exp(x)
        w_tmp[1, 0] = w[1, 0] * np.exp(-x)
        np.fill_diagonal(w_tmp, (-np.sum(w_tmp, axis=0)).tolist())
        model_tmp = Model(real_to_observed, w=w_tmp, dt=0.0001)
        pps_list.append(model_tmp.passive_partial_Sigma)
        ips_list.append(model_tmp.informed_partial_Sigma)
        Sigma_list.append(model_tmp.steady_state_Sigma)

    plt.plot(x_list, pps_list, label='Passive')
    plt.plot(x_list, ips_list, label='Informed')
    plt.plot(x_list, Sigma_list, label='Total')
    plt.legend()
    plt.xlabel('x')
    plt.yscale('log')
    plt.show()


def plot_Sigma3(real_to_observed, w):
    """
    Plots the partial and informed entropy productions and Sigma_KLD.

    :param real_to_observed: A dictionary with real state as key and observed state as value.
    :param w: The rate matrix
    :return:
    """
    pps_list = []
    ips_list = []
    Sigma_list = []
    Sigma_KLD_list = []
    x_list = np.sort([x for x in range(-3, 5)] + [-0.67])
    N = 10 ** 6
    for x in x_list:
        w_tmp = w.copy()
        np.fill_diagonal(w_tmp, 0)
        w_tmp[0, 1] = w[0, 1] * np.exp(x)
        w_tmp[1, 0] = w[1, 0] * np.exp(-x)
        np.fill_diagonal(w_tmp, (-np.sum(w_tmp, axis=0)).tolist())
        model_tmp = Model(real_to_observed, w=w_tmp, dt=0.0001)
        model_tmp.sample_trajectory(N)
        # model_tmp = get_model(real_to_observed, w, x, N)
        pps_list.append(model_tmp.passive_partial_Sigma)
        ips_list.append(model_tmp.informed_partial_Sigma)
        Sigma_list.append(model_tmp.steady_state_Sigma)
        Sigma_aff_tmp = model_tmp.get_Sigma_aff()
        Sigma_WTD_tmp = model_tmp.get_Sigma_WTD()
        Sigma_KLD_list.append(Sigma_aff_tmp + Sigma_WTD_tmp)
        print(x, ' - ', 'Sigma:', Sigma_list[-1], 'KLD: ', Sigma_KLD_list[-1], 'WTD: ', Sigma_WTD_tmp, 'affinity: ',
              Sigma_aff_tmp, 'informed: ', ips_list[-1])

    plt.plot(x_list, pps_list, label='Passive')
    plt.plot(x_list, ips_list, label='Informed')
    plt.plot(x_list, Sigma_list, label='Total')
    plt.plot(x_list, Sigma_KLD_list, label='KLD')
    plt.legend()
    plt.xlabel('x')
    plt.yscale('log')
    plt.show()


def plot_dunkel(real_to_observed, w):
    pps_list = []
    ips_list = []
    Sigma_list = []
    Sigma_KLD_list = []
    Sigma2_list = []
    x_list = np.sort([x for x in range(-3, 5)] + [-0.67])
    N = 10 ** 7
    for x in x_list:
        print(x)
        w_tmp = w.copy()
        np.fill_diagonal(w_tmp, 0)
        w_tmp[0, 1] = w[0, 1] * np.exp(x)
        w_tmp[1, 0] = w[1, 0] * np.exp(-x)
        np.fill_diagonal(w_tmp, (-np.sum(w_tmp, axis=0)).tolist())
        model_tmp = Model(real_to_observed, w=w_tmp, dt=0.0001)
        model_tmp.sample_trajectory(N)
        # model_tmp = get_model(real_to_observed, w, x, N)
        # pps_list.append(model_tmp.passive_partial_Sigma)
        # ips_list.append(model_tmp.informed_partial_Sigma)
        # Sigma_list.append(model_tmp.steady_state_Sigma)
        # Sigma_aff_tmp = model_tmp.get_Sigma_aff()
        # Sigma_WTD_tmp = model_tmp.get_Sigma_WTD()
        # Sigma_KLD_list.append(Sigma_aff_tmp + Sigma_WTD_tmp)
        ep, ijk_dict = get_Sigma2(model_tmp.trajectory)
        with open(f'stats_{x}.json', 'w') as jsonFile:
            json.dump(ijk_dict, jsonFile)
        Sigma2_list.append(ep)
        # print(x, ' - ', 'Sigma: ', Sigma_list[-1], 'KLD: ', Sigma_KLD_list[-1], 'WTD: ', Sigma_WTD_tmp, 'affinity: ',
        #       Sigma_aff_tmp, 'informed: ', ips_list[-1], 'Sigma2: ', Sigma2_list[-1])

    # plt.plot(x_list, pps_list, label='Passive')
    # plt.plot(x_list, ips_list, label='Informed')
    # plt.plot(x_list, Sigma_list, label='Total')
    # plt.plot(x_list, Sigma_KLD_list, label='KLD')
    # plt.plot(x_list, Sigma2_list, label='Sigma2')
    # plt.legend()
    # plt.xlabel('x')
    # plt.yscale('log')
    # plt.show()


def task1():
    # Estimates the rate matrix and the steady state from a trajectory statistics
    n = 4
    real_to_observed = {i: i for i in range(n)}
    model = Model(real_to_observed)
    N = 1000000
    model.sample_trajectory(N)
    w, steady_state = model.trajectory.estimate_from_statistics()
    print('The rate matrix:')
    print(model.w)
    print('The rate matrix from trajectory statistics:')
    print(w)
    print('The steady state:')
    print(model.steady_state)
    print('The steady state from trajectory statistics:')
    print(steady_state)

    # Plots the rate matrix convergence
    model.plot_w_convergence()

    # Plots a trajectory
    model.sample_trajectory(N=10)
    model.plot_trajectory()

    # Plots the numeric process to get the steady state
    n = 10
    real_to_observed = {i: i for i in range(n)}
    model = Model(real_to_observed)
    model.numeric_steady_state(plot_flag=True)

    plt.show()


def task2():
    # Plots the passive and informed partial entropy productions
    w = np.array([[-8, 9, 0, 2],
                  [1, -20, 4, 6],
                  [0, 10, -12, 5],
                  [7, 1, 8, -13]], dtype=float)
    plot_Sigma2(w)


def task3():
    real_to_observed = {0: 0,
                        1: 1,
                        2: 2,
                        3: 2
                        }

    w = np.array([[-11, 2, 0, 1],
                  [3, -52.2, 2, 35],
                  [0, 50, -77, 0.7],
                  [8, 0.2, 75, -36.7]], dtype=float)

    # model = Model(real_to_observed,
    #               w=w)

    # N = 1000000
    # model.sample_trajectory(N)
    # print(model.get_Sigma_aff())
    # print(model.trajectory.Sigma_WTD)

    # return model
    plot_Sigma3(real_to_observed, w)


def task4():
    real_to_observed = {0: 0,
                        1: 1,
                        2: 2,
                        3: 2
                        }

    w = np.array([[-11, 2, 0, 1],
                  [3, -52.2, 2, 35],
                  [0, 50, -77, 0.7],
                  [8, 0.2, 75, -36.7]], dtype=float)

    plot_dunkel(real_to_observed, w)


def get_model(real_to_observed, w, x, N):
    """
    Sample a trajectory and save the model with force x if does not exist.

    :param real_to_observed: A dictionary with real state as key and observed state as value
    :param w: The rate matrix
    :param x: The force
    :param N: The size of the trajectory
    :return:
    """
    file_name = f'model_{x}'
    if not os.path.exists(file_name):
        with open(file_name, 'wb') as output:
            w_tmp = w.copy()
            np.fill_diagonal(w_tmp, 0)
            w_tmp[0, 1] = w[0, 1] * np.exp(x)
            w_tmp[1, 0] = w[1, 0] * np.exp(-x)
            np.fill_diagonal(w_tmp, (-np.sum(w_tmp, axis=0)).tolist())
            model = Model(real_to_observed, w=w_tmp, dt=0.0001)
            model.sample_trajectory(N)
            pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open(file_name, 'rb') as input:
            model = pickle.load(input)
    return model


def get_Sigma2(trj):
    observed_states = trj.observed_states
    n_observed = trj.n_observed
    w_est, p_est = trj.estimate_from_statistics()
    n_est = w_est.T * p_est
    cutoff = 1e-10
    ep = 0
    ijk_dict = {}
    for j in range(n_observed):
        J = observed_states[j]
        other_states = observed_states[:j]
        if j < n_observed - 1:
            other_states += observed_states[j + 1:]
        for i in range(n_observed - 1):
            I = other_states[i]
            for k in range(i+1, n_observed - 1):
                K = other_states[k]
                n_IJK = np.max([trj._get_n_IJK(I, J, K), cutoff])
                n_KJI = np.max([trj._get_n_IJK(K, J, I), cutoff])
                n_JI = np.max([n_est[J, I], cutoff])
                n_IJ = np.max([n_est[I, J], cutoff])
                n_JK = np.max([n_est[J, K], cutoff])
                # n_KJ = np.max([n_est[K, J], cutoff])
                ijk_dict[f'{i}{j}{k}'] = dict(n_JI=n_JI,
                                              n_IJ=n_IJ,
                                              n_JK=n_JK,
                                              n_IJK=n_IJK,
                                              n_KJI=n_KJI
                                              )
                ep += calc_Sigma2(n_JI, n_IJ, n_JK, n_IJK, n_KJI)
    # with open('stats.json', 'w') as jsonFile:
    #     json.dump(ijk_dict, jsonFile)
    return ep / 2.0, ijk_dict


def calc_Sigma2(n_JI, n_IJ, n_JK, n_IJK, n_KJI):
    _tol = 1e-10
    n_0 = np.array(4 * [n_JI] + 4 * [n_IJ] + 4 * [n_JK]) / 4.0
    # n_0 = np.random.rand(12)# n_00 * np.random.rand(12)

    # cons = [{'type': 'eq', 'fun': lambda x: np.sum(np.divide(x[4:8] * x[8:], x[:4] + x[8:], out=np.zeros(4), where=(x[:4]+x[8:]>2*cutoff))) - n_IJK},
    #         {'type': 'eq', 'fun': lambda x: np.sum(np.divide(x[:4] * (x[:4] - x[4:8] + x[8:]), x[:4] + x[8:], out=np.zeros(4), where=(x[:4]+x[8:]>2*cutoff))) - n_KJI},
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x[4:8] * x[8:] / (x[:4] + x[8:] + _tol)) - n_IJK},
            {'type': 'eq',
             'fun': lambda x: np.sum(x[:4] * (x[:4] - x[4:8] + x[8:]) / (x[:4] + x[8:] + _tol)) - n_KJI},
            {'type': 'ineq', 'fun': lambda x: n_JI - np.sum(x[:4])},
            {'type': 'ineq', 'fun': lambda x: n_IJ - np.sum(x[4:8])},
            {'type': 'ineq', 'fun': lambda x: n_JK - np.sum(x[8:])}
            # {'type': 'ineq', 'fun': lambda x: n_KJ - np.sum(x[:4] - x[4:8] + x[8:])},
            ] + [{'type': 'ineq', 'fun': lambda x: x[i] - x[4 + i] + x[8 + i]} for i in range(4)]
    bnds = 4 * [(0, n_JI)] + 4 * [(0, n_IJ)] + 4 * [(0, n_JK)]
    con_tol = 1e-6
    res = minimize(entropy_production, n_0, method='SLSQP', options={'maxiter': 1e4},
                   bounds=bnds,
                   constraints=cons, tol=con_tol)

    while res.status != 0:
        con_tol *= 4
        # n_0 = np.random.rand(12)  # n_00 * np.random.rand(12)
        res = minimize(entropy_production, n_0, method='SLSQP',
                       options={'maxiter': 1e4}, bounds=bnds,
                       constraints=cons, tol=con_tol)

    return entropy_production(res.x)


def dunkel():
    real_to_observed = {0: 0,
                        1: 1,
                        2: 2,
                        3: 2
                        }

    w = np.array([[-11, 2, 0, 1],
                  [3, -52.2, 2, 35],
                  [0, 50, -77, 0.7],
                  [8, 0.2, 75, -36.7]], dtype=float)

    x = -3

    w_tmp = w.copy()
    np.fill_diagonal(w_tmp, 0)
    w_tmp[0, 1] = w[0, 1] * np.exp(x)
    w_tmp[1, 0] = w[1, 0] * np.exp(-x)
    np.fill_diagonal(w_tmp, (-np.sum(w_tmp, axis=0)).tolist())

    model = Model(real_to_observed, w_tmp, 0.0001)
    print(model.w.T * model.steady_state)
    print(model.steady_state)
    model.sample_trajectory(N=10 ** 6)
    # trj = model.trajectory
    # w_est, p_est = trj.estimate_from_statistics()

    return model


def entropy_production(n):
    """

    :param n: n_jI, n_Ij, n_jJ, n_Kj
    :return:
    """
    m = 4

    n_jI = np.array(n[:m])
    n_Ij = np.array(n[m:2 * m])
    n_jK = np.array(n[2 * m:3 * m])
    n_Kj = n_jI - n_Ij + n_jK


    # res = 0
    # for i in range(m):
    #     if n_Ij[i] > cutoff and n_jI[i] > cutoff:
    #         res += (n_jI[i] - n_Ij[i]) * np.log(n_jI[i] / n_Ij[i])
    #     if n_Kj[i] > cutoff and n_jK[i] > cutoff:
    #         res += (n_jK[i] - n_Kj[i]) * np.log(n_jK[i] / n_Kj[i])
    _tol = 1e-10

    res = np.sum((n_jI-n_Ij)*np.real(np.log((n_jI+_tol)/(n_Ij+_tol))) + (n_jK-n_Kj)*np.real(np.log((n_jK+_tol)/(n_Kj+_tol))))

    return res


def ep2(x):
    a = x[0]
    b = x[1]
    _tol = 1e-10
    return 4 * (a - b) * np.log((a+_tol) / (b+_tol))


def dunkel_exmple():
    p_list = np.linspace(0, 0.99, 100)
    ep_list = []
    _tol = 1e-10
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(np.square(x)) / (np.sum(x)+_tol) - p},
            {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)}
            ]
    bnds = tuple([(0, 1)] * 2)
    x_0 = np.array([0.2, 0.4])
    con_tol = 1e-10
    for p in p_list:
        res = minimize(ep2, x_0, method='SLSQP', options={'maxiter': 1e4}, bounds=bnds, constraints=cons, tol=con_tol)
        while res.status != 0:
            con_tol *= 4
            res = minimize(ep2, x_0, method='SLSQP', options={'maxiter': 1e4}, bounds=bnds, constraints=cons, tol=con_tol)
        ep_list.append(ep2(res.x))
    plt.plot(p_list, ep_list)
    plt.show()


def main():
    # w = np.array([[-11, 2, 0, 1],
    #               [3, -52.2, 2, 35],
    #               [0, 50, -77, 0.7],
    #               [8, 0.2, 75, -36.7]], dtype=float)
    model = dunkel()
    trj = model.trajectory
    w_est, p_est = trj.estimate_from_statistics()
    n_est = w_est.T * p_est
    w, p = model.w, model.steady_state
    n = w.T * p
    n_IJK = trj._get_n_IJK(0, 2, 1)
    n_KJI = trj._get_n_IJK(1, 2, 0)
    n_00 = np.array(4 * [n_est[2, 0]] + 4 * [n_est[0, 2]] + 4 * [n_est[2, 1]])
    n_0 = n_00 / 4.0
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x[4:8] * x[8:] / (x[:4] + x[8:])) - n_IJK},
            {'type': 'eq', 'fun': lambda x: np.sum(x[:4] * (x[:4] + x[4:8] - x[8:]) / (x[:4] + x[8:])) - n_KJI},
            {'type': 'ineq', 'fun': lambda x: n_est[2, 0] - np.sum(x[:4])},
            {'type': 'ineq', 'fun': lambda x: n_est[0, 2] - np.sum(x[4:8])},
            {'type': 'ineq', 'fun': lambda x: n_est[2, 1] - np.sum(x[8:])},
            {'type': 'ineq', 'fun': lambda x: n_est[1, 2] - np.sum(x[:4] + x[4:8] - x[8:])},
            {'type': 'ineq', 'fun': lambda x: np.min(x[:4] + x[4:8] - x[8:])}
            ]
    cutoff = 1e-4
    bnds = tuple([(cutoff, None)] * 12)
    res = minimize(entropy_production(cutoff), n_0, method='SLSQP', options={'disp': True, 'maxiter': 1e5}, bounds=bnds,
                   constraints=cons)
    test = res.x
    test = np.append(test, res.x[:4] + res.x[4:8] - res.x[8:]).reshape(1, -1)
    test[test < cutoff * 1.1] = 0
    ep = entropy_production(res.x)
    return ep


if __name__ == '__main__':
    # task1()
    # task2()
    # task3()

    task4()

    pass
