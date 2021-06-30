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


def plot_dunkel():
    pps_list = [1.00196127014177, 0.452435294599866, 0.0396826503472705, 4.11814583773196e-07, 0.238340479269024, 2.16209063034565, 7.5012876279282, 16.4467696812019, 26.4274326150279]
    ips_list = [1.82243849911687, 0.805706287016173, 0.0696701901928799, 7.22232770092437e-07, 0.419813358949541, 3.87743978806599, 13.7386160984984, 30.6515988464724, 49.8919603932346]
    Sigma_list = [208.369826042737, 205.108904691876, 199.350857466418, 196.585390199776, 189.04154524386, 170.92808364218, 144.858509130704, 120.971535599399, 110.543562133818]
    Sigma_KLD_list = [1.91614527103623, 0.921161364260633, 0.233793282750832, 0.190248633152876, 0.687236608542188, 4.35390718665019, 14.5632862960081, 31.8716590274228, 51.3995346687822]
    Sigma2_list = []

    x_list = np.sort([x for x in range(-3, 5)] + [-0.67])

    # real_to_observed = {0: 0,
    #                     1: 1,
    #                     2: 2,
    #                     3: 2
    #                     }
    #
    # w = np.array([[-11, 2, 0, 1],
    #               [3, -52.2, 2, 35],
    #               [0, 50, -77, 0.7],
    #               [8, 0.2, 75, -36.7]], dtype=float)

    for x in x_list:
        # print(x)
        # w_tmp = w.copy()
        # np.fill_diagonal(w_tmp, 0)
        # w_tmp[0, 1] = w[0, 1] * np.exp(x)
        # w_tmp[1, 0] = w[1, 0] * np.exp(-x)
        # np.fill_diagonal(w_tmp, (-np.sum(w_tmp, axis=0)).tolist())
        # model_tmp = Model(real_to_observed, w=w_tmp, dt=0.0001)
        # ijk_dict = get_Sigma2_stats_from_model(model_tmp)

        with open(f'stats\stats8_{x}.json', 'r') as jsonFile:
            ijk_dict = json.load(jsonFile)
        ep = 0
        for ijk_stats in ijk_dict.values():
            ep += calc_Sigma2(**ijk_stats)/2.0
        Sigma2_list.append(ep)
    print(Sigma2_list)
    plt.plot(x_list, pps_list, label='Passive')
    plt.plot(x_list, ips_list, label='Informed')
    plt.plot(x_list, Sigma_list, label='Total')
    plt.plot(x_list, Sigma_KLD_list, label='KLD')
    plt.plot(x_list, Sigma2_list, label='Sigma2')
    plt.legend()
    plt.xlabel('x')
    plt.yscale('log')
    plt.show()


def save_dunkel_stats():
    real_to_observed = {0: 0,
                        1: 1,
                        2: 2,
                        3: 2
                        }

    w = np.array([[-11, 2, 0, 1],
                  [3, -52.2, 2, 35],
                  [0, 50, -77, 0.7],
                  [8, 0.2, 75, -36.7]], dtype=float)

    x_list = np.sort([x for x in range(-3, 5)] + [-0.67])

    N = 10 ** 7

    for x in x_list:
        print(x)
        name = f'stats\stats8_{float(x)}.json'
        w_tmp = w.copy()
        np.fill_diagonal(w_tmp, 0)
        w_tmp[0, 1] = w[0, 1] * np.exp(x)
        w_tmp[1, 0] = w[1, 0] * np.exp(-x)
        lamda_arr = np.sum(w_tmp, axis=0)
        np.fill_diagonal(w_tmp, (-lamda_arr).tolist())
        w_tmp /= lamda_arr
        model_tmp = Model(real_to_observed, w=w_tmp, dt=0.0001)
        # model_tmp.sample_trajectory(N)
        # save_statistics(trj=model_tmp.trajectory, name=f'stats\stats6_{float(x)}.json')
        # ijk_dict = get_Sigma2_stats_from_trajectory(trj)
        ijk_dict = get_Sigma2_stats_from_model(model_tmp)
        with open(name, 'w') as jsonFile:
            json.dump(ijk_dict, jsonFile)


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
    # real_to_observed = {0: 0,
    #                     1: 1,
    #                     2: 2,
    #                     3: 2
    #                     }
    #
    # w = np.array([[-11, 2, 0, 1],
    #               [3, -52.2, 2, 35],
    #               [0, 50, -77, 0.7],
    #               [8, 0.2, 75, -36.7]], dtype=float)

    plot_dunkel()


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


def get_Sigma2_stats_from_trajectory(trj):
    observed_states = trj.observed_states
    n_observed = trj.n_observed
    w_est, p_est = trj.estimate_from_statistics()
    n_est = w_est.T * p_est
    cutoff = 1e-10
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
                ijk_dict[f'{I}{J}{K}'] = dict(n_JI=n_JI,
                                              n_IJ=n_IJ,
                                              n_JK=n_JK,
                                              n_IJK=n_IJK,
                                              n_KJI=n_KJI
                                              )
    return ijk_dict


def get_Sigma2_stats_from_model(model):
    observed_states = model.observed_states
    n_observed = model.n_observed
    cutoff = 1e-10
    ijk_dict = {}
    for j in range(n_observed):
        J = observed_states[j]
        other_states = observed_states[:j]
        if j < n_observed - 1:
            other_states += observed_states[j + 1:]
        for i in range(n_observed - 1):
            I = other_states[i]
            for k in range(i + 1, n_observed - 1):
                K = other_states[k]
                n_IJK = np.max([model.get_n_ijk(I, J, K, obs=True), cutoff])
                n_KJI = np.max([model.get_n_ijk(K, J, I, obs=True), cutoff])
                n_JI = np.max([model.n_matrix_observed[J, I], cutoff])
                n_IJ = np.max([model.n_matrix_observed[I, J], cutoff])
                n_JK = np.max([model.n_matrix_observed[J, K], cutoff])
                # n_KJ = np.max([n_est[K, J], cutoff])
                ijk_dict[f'{I}{J}{K}'] = dict(n_JI=n_JI,
                                              n_IJ=n_IJ,
                                              n_JK=n_JK,
                                              n_IJK=n_IJK,
                                              n_KJI=n_KJI
                                              )
    return ijk_dict


def save_statistics(trj, name):
    ijk_dict = get_Sigma2_stats_from_trajectory(trj)
    with open(name, 'w') as jsonFile:
        json.dump(ijk_dict, jsonFile)


def calc_Sigma2(n_JI, n_IJ, n_JK, n_IJK, n_KJI):
    _tol = 1e-10
    n_0 = np.array(4 * [n_JI] + 4 * [n_IJ] + 4 * [n_JK]) / 4.0
    # n_0 = np.random.rand(12)# n_00 * np.random.rand(12)
    #
    # cons = [{'type': 'eq', 'fun': lambda x: np.sum(np.divide(x[4:8] * x[8:], x[:4] + x[8:], out=np.zeros(4), where=(x[:4]+x[8:]>2*_tol))) - n_IJK},
    #         {'type': 'eq', 'fun': lambda x: np.sum(np.divide(x[:4] * (x[:4] - x[4:8] + x[8:]), x[:4] + x[8:], out=np.zeros(4), where=(x[:4]+x[8:]>2*_tol))) - n_KJI},
    cons = [{'type': 'eq',
             'fun': lambda x: n_IJK - np.sum(x[4:8] * x[8:] / (x[:4] + x[8:] + _tol)),
             'jac': lambda x: np.concatenate([x[4:8] * x[8:] / (x[:4] + x[8:] + _tol)**2,
                                              -x[8:] / (x[:4] + x[8:] + _tol),
                                              -x[:4] * x[4:8] / (x[:4] + x[8:] + _tol)**2
                                              ])
             },
            {'type': 'eq',
             'fun': lambda x: n_KJI - np.sum(x[:4] - x[:4] * x[4:8] / (x[:4] + x[8:] + _tol)),
             'jac': lambda x: np.concatenate([x[4:8] * x[8:] / (x[:4] + x[8:] + _tol)**2 - 1,
                                              x[:4] / (x[:4] + x[8:] + _tol),
                                              -x[:4] * x[4:8] / (x[:4] + x[8:] + _tol)**2
                                              ])
             },
            {'type': 'ineq', 'fun': lambda x: n_JI - np.sum(x[:4])},
            {'type': 'ineq', 'fun': lambda x: n_IJ - np.sum(x[4:8])},
            {'type': 'ineq', 'fun': lambda x: n_JK - np.sum(x[8:])}
            ] + [{'type': 'ineq', 'fun': lambda x: x[i] - x[4 + i] + x[8 + i]} for i in range(4)]
    bnds = 4 * [(0, n_JI)] + 4 * [(0, n_IJ)] + 4 * [(0, n_JK)]
    con_tol = 1e-6
    res = minimize(entropy_production, n_0, jac=epr_jac, method='SLSQP',
                   options={'maxiter': 1e4, 'ftol': 1e-7}, bounds=bnds,
                   constraints=cons, tol=con_tol)
    ep_min = 0
    ep = entropy_production(res.x)
    if ep > 0:
        ep_min = ep
    while res.status != 0 and con_tol < 1:
        con_tol *= 4
        res = minimize(entropy_production, n_0, jac=epr_jac, method='SLSQP',
                       options={'maxiter': 1e4, 'ftol': 1e-5}, bounds=bnds,
                       constraints=cons, tol=con_tol)
        ep = entropy_production(res.x)
        if ep >= 0:
            ep_min = min(ep_min, ep)
    if res.status != 0:
        print("Didn't converge")
    else:
        ep_min = entropy_production(res.x)

    return ep_min


def epr_jac(x):
    _tol = 1e-10
    n_JI = x[:4]
    n_IJ = x[4:8]
    n_JK = x[8:]
    n_KJ = n_JK + n_JI - n_IJ
    tmp_jac = lambda a, b: (a - b) / (a + _tol) + np.real(np.log((a + _tol) / (b + _tol) + 0j))
    ret1 = np.concatenate([tmp_jac(n_JI, n_IJ), tmp_jac(n_IJ, n_JI), tmp_jac(n_JK, n_KJ)])
    # ret2 = np.concatenate([(x[:4] - x[4:8]) / (x[:4] + _tol) + np.real(np.log((x[:4] + _tol) / (x[4:8] + _tol) + 0j)),
    #                                 (x[4:8] - x[:4]) / (x[4:8] + _tol) + np.real(np.log((x[4:8] + _tol) / (x[:4] + _tol) + 0j)),
    #                                 (-x[:4] + x[4:8]) / (x[8:] + _tol) + np.real(np.log((x[8:] + _tol) / (x[:4] - x[4:8] + x[8:] + _tol) + 0j))
    #                                 ])
    # if not (ret1 == ret2).all():
    #     import pdb
    #     pdb.set_trace()
    return ret1


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

    x = 4

    w_tmp = w.copy()
    np.fill_diagonal(w_tmp, 0)
    w_tmp[0, 1] = w[0, 1] * np.exp(x)
    w_tmp[1, 0] = w[1, 0] * np.exp(-x)
    lamda_arr = np.sum(w_tmp, axis=0)
    np.fill_diagonal(w_tmp, (-lamda_arr).tolist())
    w_tmp = w_tmp/lamda_arr
    print(w_tmp)
    model = Model(real_to_observed, w_tmp, 0.0001)
    # print(model.w.T * model.steady_state)
    # print(model.steady_state)
    model.sample_trajectory(N=10 ** 7)
    # trj = model.trajectory
    # w_est, p_est = trj.estimate_from_statistics()

    return model


def entropy_production(n):
    """

    :param n: n_jI, n_Ij, n_jJ, n_Kj
    :return:
    """
    m = 4
    _tol = 1e-10

    n_jI = np.array(n[:m])
    n_Ij = np.array(n[m:2 * m])
    n_jK = np.array(n[2 * m:3 * m])
    n_Kj = n_jI - n_Ij + n_jK

    # res = 0
    # for i in range(m):
    #     if n_Ij[i] > 0 and n_jI[i] > 0:
    #         res += (n_jI[i] - n_Ij[i]) * np.log(n_jI[i] / n_Ij[i])
    #     if n_Kj[i] > 0 and n_jK[i] > 0:
    #         res += (n_jK[i] - n_Kj[i]) * np.log(n_jK[i] / n_Kj[i])

    res = np.sum((n_jI-n_Ij)*np.real(np.log((n_jI+_tol)/(n_Ij+_tol) + 0j)) + (n_jK-n_Kj)*np.real(np.log((n_jK+_tol)/(n_Kj+_tol) + 0j)))
    # res = np.sum((n_jI-n_Ij)*np.log(np.abs((n_jI+_tol)/(n_Ij+_tol))) + (n_jK-n_Kj)*np.log(np.abs((n_jK+_tol)/(n_Kj+_tol))))

    return res


def ep2(x):
    a = x[0]
    b = x[1]
    _tol = 1e-10
    return 4 * (a - b) * np.log(np.abs((a+_tol) / (b+_tol)))


def dunkel_exmple():
    p_list = np.linspace(0, 0.99, 100)
    ep_list = []
    _tol = 1e-10
    cons = [{'type': 'eq',
             'fun': lambda x: np.sum(x**2) / (np.sum(x)+_tol) - p,
             'jac': lambda x: np.array([1-2*(x[1]/(np.sum(x)+_tol))**2,
                                        1-2*(x[0]/(np.sum(x)+_tol))**2])
             },
            {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)}
            ]
    jac = lambda x: 4*np.array([(x[0]-x[1])/(x[0]+_tol) + np.log(np.abs((x[0]+_tol)/(x[1]+_tol))),
                                (x[1]-x[0])/(x[1]+_tol) + np.log(np.abs((x[1]+_tol)/(x[0]+_tol)))
                                ])
    bnds = [(0, 1)] * 2
    x_0 = np.array([0.2, 0.4])
    con_tol = 1e-10
    for p in p_list:
        res = minimize(ep2, x_0, jac=jac, method='SLSQP', options={'maxiter': 1e4}, bounds=bnds, constraints=cons, tol=con_tol)
        ep_min = ep2(res.x)
        while res.status != 0:
            con_tol *= 4
            res = minimize(ep2, x_0, jac=jac, method='SLSQP', options={'maxiter': 1e4}, bounds=bnds, constraints=cons, tol=con_tol)
            ep_min = min(ep_min, ep2(res.x))
        ep_list.append(ep_min)
    plt.plot(p_list, ep_list)
    plt.show()


def dunkel_example2():
    lamda = 1
    r = 0.05

    N = 10 ** 6

    # real_to_observed = {0: 0,
    #                     1: 1,
    #                     2: 2,
    #                     3: 3,
    #                     4: 0,
    #                     5: 1,
    #                     6: 2,
    #                     7: 3}

    real_to_observed = {0: 0,
                        1: 1,
                        2: 2,
                        3: 0,
                        4: 1,
                        5: 2}

    p_list = np.linspace(r/2, 1-r-0.05, 10)

    total_list = []
    # kld_list = []
    sigma2_list = []
    for p in p_list:
        print(p)
        p1 = p
        p2 = p
        q1 = lamda - r - p1
        q2 = lamda - r - p2

        # w = [[-lamda, p1, 0, q1, r, 0, 0, 0],
        #      [q1, -lamda, p1, 0, 0, r, 0, 0],
        #      [0, q1, -lamda, p1, 0, 0, r, 0],
        #      [p1, 0, q1, -lamda, 0, 0, 0, r],
        #      [r, 0, 0, 0, -lamda, q2, 0, p2],
        #      [0, r, 0, 0, p2, -lamda, q2, 0],
        #      [0, 0, r, 0, 0, p2, -lamda, q2],
        #      [0, 0, 0, r, q2, 0, p2, -lamda]]

        w = [[-lamda, p1, q1, r, 0, 0],
             [q1, -lamda, p1, 0, r, 0],
             [p1, q1, -lamda, 0, 0, r],
             [r, 0, 0, -lamda, q2, p2],
             [0, r, 0, p2, -lamda, q2],
             [0, 0, r, q2, p2, -lamda]]

        w = np.array(w, dtype=float)
        # print(w)
        model = Model(real_to_observed=real_to_observed, w=w, dt=0.0001)

        # model.sample_trajectory(N)

        total_list.append(model.steady_state_Sigma)
        # kld_list.append(model.get_Sigma_KLD())

        # ijk_dict = get_Sigma2_stats_from_trajectory(model.trajectory)
        ijk_dict = get_Sigma2_stats_from_model(model)
        # print(ijk_dict)
        sigma2 = 0
        for ijk, ijk_stats in ijk_dict.items():
            sigma2 += calc_Sigma2(**ijk_stats)/2.0
        sigma2_list.append(sigma2)
        print(total_list[-1], ' - ', sigma2_list[-1])

    print('total - ', total_list)
    print('sigma2 - ', sigma2_list)
    plt.plot(p_list, total_list, label='Total')
    # plt.plot(p_list, kld_list, label='KLD')
    plt.plot(p_list, sigma2_list, label='Sigma2')
    plt.legend()
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
    # w, p = model.w, model.steady_state
    n_obs = model.n_matrix_observed
    print(n_obs)
    print(n_est)
    print(trj._get_n_matrix())
    for i in range(3):
        for j in range(3):
            if j != i:
                for k in range(3):
                    if k not in [i, j]:
                        print(f'ijk = {i}{j}{k}')
                        print(trj._get_n_IJK(i, j, k), ' - ', model.get_n_ijk(i,j,k,obs=True))
                        print(trj._get_n_IJK(k, j, i), ' - ', model.get_n_ijk(k,j,i,obs=True))
    # n_0 = np.array(4 * [n_est[2, 0]] + 4 * [n_est[0, 2]] + 4 * [n_est[2, 1]])/4.0



if __name__ == '__main__':
    # task1()
    # task2()
    # task3()
    # main()
    # task4()
    dunkel_example2()
    # save_dunkel_stats()
    # real_to_observed = {0: 0,
    #                     1: 1,
    #                     2: 2,
    #                     3: 2
    #                     }
    #
    # w = np.array([[-11, 2, 0, 1],
    #               [3, -52.2, 2, 35],
    #               [0, 50, -77, 0.7],
    #               [8, 0.2, 75, -36.7]], dtype=float)
    #
    # model = Model(real_to_observed, w)
    # model.sample_trajectory(10**7)
    # trj = model.trajectory
    pass
