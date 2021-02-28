import numpy as np
import matplotlib.pyplot as plt
from project import Model


def plot_Sigma(w):
    """
    Plots the partial and informed entropy productions as in Figure 4 b (2017 Hierarchical bounds...)

    :param w: The rate matrix
    :return:
    """
    pps_list = []
    ips_list = []
    Sigma_list = []
    x_list = np.linspace(-6, 6, 120)
    for x in x_list:
        w_tmp = w.copy()
        np.fill_diagonal(w_tmp, 0)
        w_tmp[0, 1] = w[0, 1]*np.exp(x)
        w_tmp[1, 0] = w[0, 1]*np.exp(-x)
        np.fill_diagonal(w_tmp, (-np.sum(w_tmp, axis=0)).tolist())
        model_tmp = Model(n=4, w=w_tmp, dt=0.0001)
        pps_list.append(model_tmp.passive_partial_Sigma)
        ips_list.append(model_tmp.infromed_partial_Sigma)
        Sigma_list.append(model_tmp.steady_state_Sigma)

    plt.plot(x_list, pps_list, label='Passive')
    plt.plot(x_list, ips_list, label='Informed')
    plt.plot(x_list, Sigma_list, label='Total')
    plt.legend()
    plt.xlabel('x')
    plt.yscale('log')
    plt.show()


def task1():
    # Estimates the rate matrix and the steady state from a trajectory statistics
    n = 4
    model = Model(n)
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
    model = Model(n)
    model.numeric_steady_state(plot_flag=True)

    plt.show()


def task2():
    # Plots the passive and informed partial entropy productions
    w = np.array([[-8, 9, 0, 2],
                  [1, -20, 4, 6],
                  [0, 10, -12, 5],
                  [7, 1, 8, -13]], dtype=np.float)
    plot_Sigma(w)


if __name__ == '__main__':
    # task1()
    # task2()
    pass
