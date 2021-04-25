import numpy as np
import matplotlib.pyplot as plt
from project import Model
import os
import pickle


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
        w_tmp[0, 1] = w[0, 1]*np.exp(x)
        w_tmp[1, 0] = w[1, 0]*np.exp(-x)
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
    N = 10**7
    for x in x_list:
        w_tmp = w.copy()
        np.fill_diagonal(w_tmp, 0)
        w_tmp[0, 1] = w[0, 1]*np.exp(x)
        w_tmp[1, 0] = w[1, 0]*np.exp(-x)
        np.fill_diagonal(w_tmp, (-np.sum(w_tmp, axis=0)).tolist())
        model_tmp = Model(real_to_observed, w=w_tmp, dt=0.0001)
        model_tmp.sample_trajectory(N)
        # model_tmp = get_model(real_to_observed, w, x, N)
        pps_list.append(model_tmp.passive_partial_Sigma)
        ips_list.append(model_tmp.informed_partial_Sigma)
        Sigma_list.append(model_tmp.steady_state_Sigma)
        Sigma_aff_tmp = model_tmp.get_Sigma_aff()
        Sigma_WTD_tmp = model_tmp.get_Sigma_WTD()
        Sigma_KLD_list.append(Sigma_aff_tmp+Sigma_WTD_tmp)
        print(x, ' - ', 'Sigma:', Sigma_list[-1], 'KLD: ', Sigma_KLD_list[-1], 'WTD: ', Sigma_WTD_tmp, 'affinity: ', Sigma_aff_tmp, 'informed: ', ips_list[-1])

    plt.plot(x_list, pps_list, label='Passive')
    plt.plot(x_list, ips_list, label='Informed')
    plt.plot(x_list, Sigma_list, label='Total')
    plt.plot(x_list, Sigma_KLD_list, label='KLD')
    plt.legend()
    plt.xlabel('x')
    plt.yscale('log')
    plt.show()


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
                  [7, 1, 8, -13]], dtype=np.float)
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
                  [8, 0.2, 75, -36.7]], dtype=np.float)

    # model = Model(real_to_observed,
    #               w=w)

    # N = 1000000
    # model.sample_trajectory(N)
    # print(model.get_Sigma_aff())
    # print(model.trajectory.Sigma_WTD)

    # return model
    plot_Sigma3(real_to_observed, w)


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


if __name__ == '__main__':
    # task1()
    # task2()
    # task3()
    real_to_observed = {0: 0,
                        1: 1,
                        2: 2,
                        3: 2
                        }

    w = np.array([[-11, 2, 0, 1],
                  [3, -52.2, 2, 35],
                  [0, 50, -77, 0.7],
                  [8, 0.2, 75, -36.7]], dtype=np.float)

    model = Model(real_to_observed, w, 0.0001)
    model.sample_trajectory(N=10**6)
    trj = model.trajectory
    w_est, p_est = trj.estimate_from_statistics()
    model2 = Model(real_to_observed={i:i for i in range(3)}, w=w_est)
    pass
