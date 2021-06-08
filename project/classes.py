import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# import kalepy as kale

class Model:
    _w = None  # The rate matrix. (n, n) numpy array.
    _real_to_observed = None  # A dictionary with the real state as key and the observed state as value. dict.
    _observed_to_real = None  # A dictionary with the observed state as key and the possible real states as value. dict.
    _steady_state = None  # The steady state from the numeric calculations. (n, 1) numpy array.
    _steady_state_stalling = None  # The steady state at stalling. (n, 1) numpy array.
    _steady_state_J = None  # The steady state current. (n, n) numpy array.
    _steady_state_Sigma = None  # The steady state entropy production rate. (n, n) numpy array.
    _trajectory = None  # The latest sampled trajectory.
    _cache = None  #
    _dt = None  # The time delta for the numeric calculations

    @property
    def n(self):
        return len(self._real_to_observed.keys())

    @property
    def w(self):
        return self._w

    @property
    def n_observed(self):
        return len(set(self._real_to_observed.values()))

    @property
    def steady_state(self):
        if self._steady_state is None:
            self._steady_state = self.numeric_steady_state()
        return self._steady_state

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def steady_state_J(self):
        if self._steady_state_J is None:
            w_mul_pT = np.multiply(self.w, self.steady_state.T)
            self._steady_state_J = w_mul_pT-w_mul_pT.T
        return self._steady_state_J

    @property
    def steady_state_Sigma(self):
        if self._steady_state_Sigma is None:
            w_mul_pT = np.multiply(self.w, self.steady_state.T)
            np.fill_diagonal(w_mul_pT, 1)
            self._steady_state_Sigma = np.sum(np.multiply(self.steady_state_J, np.log(w_mul_pT, out=np.zeros_like(w_mul_pT), where=(w_mul_pT>0))))
        return self._steady_state_Sigma

    @property
    def passive_partial_Sigma(self):
        """
        The passive partial entropy production

        :return:
        """
        # w_mul_pT = np.multiply(self.w, self.steady_state.T)
        # np.fill_diagonal(w_mul_pT, 1)
        # return np.sum(np.multiply(self.steady_state_J, np.log(w_mul_pT)))
        return (self.steady_state_J[0, 1] * np.log(self.w[0, 1] * self.steady_state[1] / (self.w[1, 0] * self.steady_state[0])))[0]

    @property
    def steady_state_stalling(self):
        """
        The steady state at stalling

        :return:
        """
        if self._steady_state_stalling is None:
            self._steady_state_stalling = self.numeric_steady_state_stalling()
        return self._steady_state_stalling

    @property
    def informed_partial_Sigma(self):
        """
        The informed partial entropy production

        :return:
        """
        # w_mul_p_st_T = np.multiply(self.w, self.steady_state_stalling.T)
        # np.fill_diagonal(w_mul_p_st_T, 1)
        # return np.sum(np.multiply(self.steady_state_J, np.log(w_mul_p_st_T)))
        return (self.steady_state_J[0, 1] * np.log(self.w[0, 1] * self.steady_state_stalling[1] / (self.w[1, 0] * self.steady_state_stalling[0])))[0]

    def __init__(self, real_to_observed, w=None, dt=0.001):
        """

        :param real_to_observed: A dictionary with the real state as key and the observed state as value
        :param w: The rate matrix. If not provided, it will be initialized as a random (n, n) numpy array
        :param dt: The time delta for the numeric calculations
        """

        self._real_to_observed = real_to_observed
        self._create_observed_to_real()

        if w is not None:
            if w.shape[0] != w.shape[1]:
                raise Exception('w must be square')
            else:
                self._w = w
        else:
            self._w = self._initialize_w()


        self._dt = dt
        self._cache = {}

    def numeric_steady_state(self, dt=None, T=10.0, plot_flag=False):
        """
        Calculates the steady state numerically

        :param dt: The time delta
        :param T: The total time
        :param plot_flag: A flag to plot the probabilities over time
        :return:
        """
        dt = self._dt if dt is None else dt
        p = self._initialize_p()
        steps = int(T/dt)

        p_list = [p]

        for i in range(steps):
            p = p + np.dot(self.w, p)*dt
            p_list.append(p)

        if plot_flag:
            plt.figure()
            for i in range(self.n):
                plt.plot(dt*np.arange(steps+1), np.array(p_list)[:, i], label=f'p{i+1}')
                plt.xlim(0, T)
                # plt.ylim(0, 1)
            plt.legend(loc='upper right')
            plt.xlabel('Time')
            plt.ylabel('Probability')
            # plt.show()

        return p

    def numeric_steady_state_stalling(self, dt=None, T=10.0, plot_flag=False):
        """
        Calculates the steady state numerically

        :param dt: The time delta
        :param T: The total time
        :param plot_flag: A flag to plot the probabilities over time
        :return:
        """
        dt = self._dt if dt is None else dt
        p = self._initialize_p()
        steps = int(T/dt)

        p_list = [p]

        w_stalling = self.w.copy()
        w_stalling[0, 0] += w_stalling[1, 0]
        w_stalling[1, 0] = 0
        w_stalling[1, 1] += w_stalling[0, 1]
        w_stalling[0, 1] = 0

        for i in range(steps):
            p = p + np.dot(w_stalling, p) * dt
            p_list.append(p)

        if plot_flag:
            for i in range(self.n):
                plt.plot(dt*np.arange(steps+1), np.array(p_list)[:, i], label=f'p{i+1}')
                plt.xlim(0, T)
                # plt.ylim(0, 1)
            plt.legend(loc='upper right')
            plt.xlabel('Time')
            plt.ylabel('Probability')
            # plt.show()
        return p

    def sample_trajectory(self, N, initialState=0):
        """
        Samples a trajectory

        :param N: The trajectory size
        :param initialState: The initial state of the trajectory
        :param n_hidden: The number of hidden states
        :return:
        """
        self._trajectory = Trajectory(self._real_to_observed, self.w, self.steady_state, initialState)

        rates_convergence = []
        steady_state_convergence = []
        j = 1
        for i in range(1, N+1):
            self.trajectory.jump_state()
            if self.n == self.n_observed:
                if i == 10**j:
                    # print(i)
                    w, steady_state = self.trajectory.estimate_from_statistics()
                    rates_convergence.append(np.linalg.norm(w-self.w))
                    steady_state_convergence.append(np.linalg.norm(steady_state-self.steady_state))
                    j += 1
            # print(i)
        if self.n == self.n_observed:
            self._cache.update(dict(
                                    rates_convergence=rates_convergence,
                                    steady_state_convergence=steady_state_convergence
                                    )
                               )

    def plot_w_convergence(self):
        """
        Plots the rate matrix convergence

        :return:
        """
        if self._trajectory is None:
            raise Exception('No trajectory was sampled')

        if self.n != self.n_observed:
            raise Exception("Can't compute convergence with hidden layers")

        plt.figure()
        w_conv = self._cache['rates_convergence']
        plt.plot([10**(i + 1) for i in range(len(w_conv))], w_conv)
        plt.xscale('log')
        plt.title('The difference between the estimated rate matrix to the real one')
        plt.xlabel('Trajectory size (Log10 scale)')
        plt.ylabel('Difference')
        # plt.show()

    def plot_steady_state_convergence(self):
        """
        Plots the rate matrix convergence

        :return:
        """
        if self._trajectory is None:
            raise Exception('No trajectory was sampled')

        if self.n != self.n_observed:
            raise Exception("Can't compute convergence with hidden layers")

        plt.figure()
        steady_state_conv = self._cache['steady_state_convergence']
        plt.plot([10**(i + 1) for i in range(len(steady_state_conv))], steady_state_conv)
        plt.xscale('log')
        plt.title('The difference between the estimated steady state to the real one')
        plt.xlabel('Trajectory size (Log10 scale)')
        plt.ylabel('Difference')
        # plt.show()

    def plot_trajectory(self, real=False):
        """
        Plots the last sampled trajectory

        :param real: A flag to plot the real trajectory, ignoring the hidden status
        :return:
        """
        if self._trajectory is None:
            raise Exception('No trajectory was sampled')

        self.trajectory.plot(real)

    def get_Sigma_aff(self):
        return self.trajectory.Sigma_aff

    def get_Sigma_WTD(self):
        return self.trajectory.Sigma_WTD

    def get_Sigma_KLD(self):
        return self.get_Sigma_aff() + self.get_Sigma_WTD()

    def _create_observed_to_real(self):
        self._observed_to_real = {}
        for key, value in self._real_to_observed.items():
            self._observed_to_real.setdefault(value, []).append(key)

    def _initialize_w(self):
        """
        Initializes a random rate matrix

        :return:
        """
        w = np.random.rand(self.n, self.n)
        for i in range(self.n):
            w[i, i] = -np.sum(w[:, i])+w[i, i]
        return w

    def _initialize_p(self):
        """
        Initializes a random probabilities vector

        :return:
        """
        p = np.random.rand(self.n, 1)
        p /= np.sum(p)
        return p


class Trajectory:
    # n is the number of micro states
    # n_observed is the number of observed(coarse-grained) states
    _real_to_observed = None  # A dictionary with the real state as key and the observed state as value
    _w = None  # The rate matrix. (n, n) numpy array.
    _jumpProbabilities = None  # The probability to jump from state j to state i. (n, n) numpy array.
    _observed_to_real = None  # A dictionary with the observed state as key and the possible real states list as value.

    _real_trj = np.array([], dtype=int)
    _observed_trj = np.array([], dtype=int)

    _real_waiting_times = np.array([], dtype=float)
    _observed_waiting_times = np.array([], dtype=float)

    # _total_time_list = None  # The total time spent in each state. (n_observed, 1) numpy array
    # _counter_list = None  # The count of each observed state occurrence. (n_observed, 1) numpy array
    # _jump_counter = None  # The total observed jumps: j->i. (n_observed, n_observed) numpy array
    # _time_matrix = None  # A 3D matrix with a list of 2nd order waiting times in each element. (n_observed, n_observed, n_observed) numpy array, dtype=list
    _prev_observed = None  # Previous observed state
    _tmp_t = None  # Temporary time spent in the current observed state
    _p_ij_to_jk_matrix = None  # Cache of p_ij_to_jk for (i, j, k). (n_observed, n_observed, n_observed) numpy array

    @property
    def n(self):
        return self._w.shape[0]

    @property
    def w(self):
        return self._w

    @property
    def real_to_observed(self):
        return self._real_to_observed

    @property
    def observed_states(self):
        return np.unique(self.real_to_observed.values())

    @property
    def n_observed(self):
        """
        The number of the observed states

        :return:
        """
        return len(self.observed_states)

    @property
    def steady_state(self):
        return self._steady_state

    @property
    def real_total_time_list(self):
        return np.array([np.sum(self._real_waiting_times[self._real_trj == i]) for i in range(self.n)])

    @property
    def observed_total_time_list(self):
        return np.array([np.sum(self._observed_waiting_times[self._observed_trj == i]) for i in range(self.n_observed)])

    @property
    def real_counter_list(self):
        return np.array([np.sum(self._real_trj == i) for i in range(self.n)])

    @property
    def observed_counter_list(self):
        return np.array([np.sum(self._observed_trj == i) for i in range(self.n_observed)])

    @property
    def total_time(self):
        return np.sum(self._observed_waiting_times)

    @property
    def real_mean_time(self):
        _tol = 1e-10
        return self.real_total_time_list/(self.real_counter_list + _tol)
        # return np.divide(self.real_total_time_list, self.real_counter_list, out=np.zeros_like(self.real_total_time_list), where=(self.real_counter_list != 0))

    @property
    def observed_mean_time(self):
        _tol = 1e-10
        return self.observed_total_time_list / (self.observed_counter_list + _tol)

    @property
    def real_jump_counter(self):
        ret = np.zeros((self.n, self.n))
        i_list = self._real_trj[:-1]
        j_list = self._real_trj[1:]
        for i in range(self.n):
            for j in range(self.n):
                if j != i:
                    ret[j, i] = np.sum(i_list == i and j_list == j)
        return ret

    @property
    def observed_jump_counter(self):
        ret = np.zeros((self.n_observed, self.n_observed))
        i_list = self._observed_trj[:-1]
        j_list = self._observed_trj[1:]
        for i in range(self.n_observed):
            for j in range(self.n_observed):
                if j != i:
                    ret[j, i] = np.sum(i_list == i and j_list == j)
        return ret

    @property
    def real_time_matrix(self):
        ret = np.frompyfunc(list, 0, 1)(np.empty((self.n_observed,) * 3, dtype=object))

    @property
    def jumpProbabilities(self):
        return self._jumpProbabilities

    def __init__(self, real_to_observed, w, steady_state, initial_state):
        """

        :param real_to_observed: A dictionary with the real state as key and the observed state as value
        :param w: The rate matrix
        :param steady_state: The steady state probabilities
        :param initial_state: The initial state
        """
        super().__init__()
        self._w = w.copy()
        self._steady_state = steady_state
        self._real_to_observed = real_to_observed
        self._create_observed_to_real()

        self._jumpProbabilities = self.w/(-np.diagonal(self.w)).reshape(1, self.n)
        np.fill_diagonal(self._jumpProbabilities, 0)

        # self._time_matrix = [[[[] for k in range(self.n_observed)] for j in range(self.n_observed)] for i in range(self.n_observed)]
        # self._time_matrix = np.array(self._time_matrix, dtype=list)

        self._time_matrix = np.frompyfunc(list, 0, 1)(np.empty((self.n_observed,)*3, dtype=object))
        self._kde_matrix = np.empty((self.n_observed,) * 3, dtype=object)
        self._p_ij_to_jk_matrix = -np.ones((self.n_observed,)*3, dtype=float)

        # self._total_time_list = np.zeros((self.n_observed, 1))
        # self._counter_list = np.zeros((self.n_observed, 1))
        # self._jump_counter = np.zeros((self.n_observed, self.n_observed))

        t = np.random.exponential(1 / -self.w[initial_state, initial_state])
        self._tmp_t = t
        # initial_state_observed = self.get_observed_state(initial_state)
        # self._total_time_list[initial_state_observed] += t
        # self._counter_list[initial_state_observed] += 1

        # state = State(index=initial_state_index,
        #               t=t
        #               )
        #
        # self.append(state)
        self.append_state(initial_state, t)

    def get_observed_state(self, state):
        """

        :param state: The real state
        :return:
        """
        return self._real_to_observed[state]

    def append_state(self, state, t):
        """
        Append new state to trajectory

        :param state: The new state
        :param t: The waiting time
        :return:
        """
        self._real_trj = np.append(self._real_trj, state)
        self._real_waiting_times = np.append(self._real_waiting_times, t)
        observed_state = self.get_observed_state(state)
        if len(self._observed_trj) != 0 and self._observed_trj[-1] == observed_state:
            self._observed_waiting_times[-1] += t
        else:
            self._observed_trj = np.append(self._observed_trj, observed_state)
            self._observed_waiting_times = np.append(self._observed_waiting_times, t)

    def jump_state(self):
        """
        Performs a jump to a new state

        :return:
        """
        current_state = self._real_trj[-1]
        new_state = np.random.choice(np.arange(self.n), p=self.jumpProbabilities[:, current_state.index])
        t = np.random.exponential(1 / -self.w[new_state, new_state])
        # new_state = State(index=new_state_index,
        #                   t=t
        #                   )
        #
        # current_observed = self.real_to_observed[current_state.index]
        # new_observed = self.real_to_observed[new_state.index]
        #
        # self._total_time_list[new_observed] += t
        # if current_observed != new_observed:
        #     if self._prev_observed is not None:
        #         # import pdb
        #         # pdb.set_trace()
        #         self._time_matrix[self._prev_observed][current_observed][new_observed].append(self._tmp_t)
        #     self._tmp_t = t
        #     self._counter_list[new_observed] += 1
        #     self._jump_counter[new_observed, current_observed] += 1
        #     self._prev_observed = current_observed
        # else:
        #     if self._prev_observed is not None:
        #         self._tmp_t += t

        self.append_state(new_state, t)
        # return new_state

    def plot(self, real=False):
        """
        Plots the trajectory

        :param real: A flag. If True plots the real trajectory, if False plots the observed trajectory. Default False.
        :return:
        """
        N = len(self._real_trj)

        T = 0
        state = self._real_trj[0] if real else self._observed_trj[0]
        plt.figure()
        plt.hlines(state + 1, T, T + self._real_waiting_times[0])
        T += self._real_waiting_times[0]
        for i in range(1, N):
            prev_state = state
            state = self._real_trj[i] if real else self.get_observed_state(self._real_trj[i])

            plt.vlines(T, prev_state + 1, state + 1, linestyles=':')
            plt.hlines(state + 1, T, T + self._real_waiting_times[i])
            T += self._real_waiting_times[i]

        plt.xlim(0, T)
        plt.ylim(0, self.n + 1 if real else self.n_observed+1)

        yTicks = [y+1 for y in range(self.n)] if real else [y+1 for y in range(self.n_observed)]
        yLabels = [str(y) for y in yTicks] if real else [f'observed-{y}' for y in yTicks]

        plt.yticks(yTicks, yLabels)

        plt.xlabel('Time')
        plt.ylabel('State')
        # plt.show()

    def real_estimate_from_statistics(self):
        """
        Estimates the rate matrix and the steady state from the trajectory statistics.

        :return: w, steady_state
        """
        lamda = np.divide(1, self.real_mean_time, out=np.zeros_like(self.real_mean_time), where=(self.real_mean_time != 0))
        real_jump_counter = self.real_jump_counter
        total_jumps = np.sum(real_jump_counter, axis=0, keepdims=True)
        w = real_jump_counter * np.divide(lamda.T, total_jumps, out=np.zeros_like(total_jumps), where=(total_jumps != 0))
        np.fill_diagonal(w, -lamda)
        steady_state = self.real_total_time_list / self.total_time
        return w, steady_state

    def real_estimate_from_statistics(self):
        """
        Estimates the rate matrix and the steady state from the trajectory statistics.

        :return: w, steady_state
        """
        lamda = np.divide(1, self.observed_mean_time, out=np.zeros_like(self.observed_mean_time), where=(self.observed_mean_time != 0))
        observed_jump_counter = self.observed_jump_counter
        total_jumps = np.sum(observed_jump_counter, axis=0, keepdims=True)
        w = observed_jump_counter * np.divide(lamda.T, total_jumps, out=np.zeros_like(total_jumps), where=(total_jumps != 0))
        np.fill_diagonal(w, -lamda)
        steady_state = self.observed_total_time_list / self.total_time
        return w, steady_state

    def _create_observed_to_real(self):
        """
        Creates the dictionary self._observed_to_real

        :return:
        """
        self._observed_to_real = {}
        for key, value in self.real_to_observed.items():
            self._observed_to_real.setdefault(value, []).append(key)

    def _get_p_ijk(self, i, j, k):
        """
        The probability to observe the sequence i -> j -> k

        :param i: First observed state
        :param j: Second observed state
        :param k: Third observed state
        :return:
        """
        observed_jump_counter = self.observed_jump_counter
        return observed_jump_counter[j, i]/np.sum(observed_jump_counter) * self._get_p_ij_to_jk(i, j, k)

    def _get_p_ij_to_jk(self, i, j, k):
        """
        The probability of j -> k if we know the jump before was i -> j

        :param i: First observed state
        :param j: Second observed state
        :param k: Third observed state
        :return:
        """
        if self._p_ij_to_jk_matrix[i, j, k] == -1:
            N1 = len(self._time_matrix[i, j, k])
            N2 = len(self._time_matrix.sum(axis=2)[i, j])
            self._p_ij_to_jk_matrix[i, j, k] = N1/N2 if N2 != 0 else 0
        return self._p_ij_to_jk_matrix[i, j, k]

    def _get_kde_func(self, i, j, k):
        if self._kde_matrix[i, j, k] is None:
            # min_t = np.min(self._time_matrix[i][j][k])
            kde_func = gaussian_kde(self._time_matrix[i][j][k])
            # kde_func = kale.KDE(self._time_matrix[i][j][k])
            self._kde_matrix[i, j, k] = kde_func
        return self._kde_matrix[i, j, k]

    def _get_D(self, i1, j1, k1, i2, j2, k2):
        # min_t = np.min(self._time_matrix[i1][j1][k1]+self._time_matrix[i2][j2][k2])
        max_t = np.max(self._time_matrix[i1][j1][k1]+self._time_matrix[i2][j2][k2])
        kde_func_1 = self._get_kde_func(i1, j1, k1)
        kde_func_2 = self._get_kde_func(i2, j2, k2)
        t_arr = np.linspace(0, 1.5*max_t, 5000)
        # boundaries = [0.0, None]
        # _, kde_1 = kde_func_1.pdf(t_arr, reflect=boundaries)
        # _, kde_2 = kde_func_2.pdf(t_arr, reflect=boundaries)
        kde_1 = kde_func_1(t_arr) + kde_func_1(-t_arr)
        kde_2 = kde_func_2(t_arr) + kde_func_2(-t_arr)
        tmp = np.divide(kde_1, kde_2, out=np.ones_like(kde_2), where=(kde_2 > 1e-10))
        ret = (t_arr[1]-t_arr[0]) * np.sum(np.multiply(kde_1, np.log(tmp, out=np.zeros_like(kde_1), where=(kde_1 > 1e-10))))
        return ret

    @property
    def Sigma_aff(self):
        # print('in sigma affinity')
        # print(self._time_matrix)
        ret = 0
        mean_T = self.total_time/np.sum(self._jump_counter)
        for i in range(self.n_observed):
            for j in range(self.n_observed):
                for k in range(self.n_observed):
                    # print(i, ', ', j, ', ', k)
                    p_ij_to_jk = self._get_p_ij_to_jk(i, j, k)
                    p_kj_to_ji = self._get_p_ij_to_jk(k, j, i)
                    tmp = np.log(p_ij_to_jk / p_kj_to_ji if p_ij_to_jk != 0 and p_kj_to_ji != 0 else 1)
                    ret += self._get_p_ijk(i, j, k) * tmp / mean_T
        return ret

    @property
    def Sigma_WTD(self):
        ret = 0
        mean_T = self.total_time/np.sum(self._jump_counter)
        for i in range(self.n_observed):
            for j in range(self.n_observed):
                if i != j and len(self._observed_to_real[j]) > 1:
                    for k in range(self.n_observed):
                        if j != k and i != k:
                            print(i, ', ', j, ', ', k)
                            ret += self._get_p_ijk(i, j, k) * self._get_D(i, j, k, k, j, i) / mean_T
        return ret

    def _get_n_IJK(self, i, j, k):
        """

        :param i: first state
        :param j: second state
        :param k: third state
        :return:
        """
        w_est, p_est = self.observed_estimate_from_statistics()
        n_est = w_est.T*p_est
        return n_est[i, j]*self._get_p_ij_to_jk(i, j, k)


class State:
    _index = None
    _t = None

    @property
    def index(self):
        return self._index

    @property
    def t(self):
        return self._t

    def __init__(self, index, t):
        """

        :param index: The index representing the state
        :param t: The amount of time spent in the micro state
        """
        self._index = index
        self._t = t

    def add_time(self, t):
        """
        Adding time spent in the micro state

        :param t: The amount of time to add.
        :return:
        """
        self._t = self.t+t
