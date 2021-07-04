import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import sympy
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
    def observed_states(self):
        return list(set(self._real_to_observed.values()))

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

    def numeric_steady_state(self, dt=None, plot_flag=False):
        """
        Calculates the steady state numerically

        :param dt: The time delta
        :param plot_flag: A flag to plot the probabilities over time
        :return:
        """
        dt = self._dt if dt is None else dt
        p = self._initialize_p()
        T = 100.0 / np.min(np.abs(np.diagonal(self.w)))
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

    def numeric_steady_state_stalling(self, dt=None, plot_flag=False):
        """
        Calculates the steady state numerically

        :param dt: The time delta
        :param T: The total time
        :param plot_flag: A flag to plot the probabilities over time
        :return:
        """
        dt = self._dt if dt is None else dt
        p = self._initialize_p()
        T = 100.0 / np.min(np.abs(np.diagonal(self.w)))
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

    def observed_wtd_laplace_transform(self, H):
        states = [i for i in range(self.n)]
        inv_matrix = lambda s: np.linalg.inv(np.eye(self.n) - self.real_wtd_laplace_transform(s))
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                other_states = list(set(states) - {i, j})
                for k in range(self.n - 3):
                    K = other_states[k]
                    for l in range(k, self.n - 2):
                        L = other_states[l]
                        # ret +=

    def real_wtd_laplace_transform(self, s):
        real_wtd_laplace = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                real_wtd_laplace[j, i] = quad(lambda t: self.real_waiting_time_distribution(t)[j, i]*np.exp(-s*t), a=0, b=np.inf)
        return real_wtd_laplace

    def real_waiting_time_distribution(self, t):
        w_tmp = self.w.copy()
        np.fill_diagonal(w_tmp, 0)
        wtd = w_tmp*np.exp(-t/np.sum(w_tmp, axis=0))
        return wtd

    def observed_waiting_time_distribution(self, t):
        real_wtd_laplace = np.empty((self.n, self.n), dtype=object)
        observed_wtd_laplace = self.observed_wtd_laplace_transform()
        for i in range(self.n):
            for j in range(self.n):
                real_wtd_laplace[j, i] = quad(lambda s: observed_wtd_laplace(s)[j, i] * np.exp(s * t), a=0, b=np.inf)
        return real_wtd_laplace

    @property
    def Sigma_KLD(self):
        t, s = sympy.symbols('t, s')
        ret = 0
        for I in range(self.n_observed-1):
            i_states = self._observed_to_real[I]
            for J in range(I+1, self.n_observed):
                j_states = self._observed_to_real[J]
                for K in range(self.n_observed):
                    if K != I and K != J:
                        k_states = self._observed_to_real[K]
                        if len(k_states) > 1:
                            for i in i_states:
                                for j in j_states:
                                    for k in k_states:
                                        ret += self.get_Sigma_WTD(i, j, k)
                                        ret += self.get_Sigma_WTD(k, j, i)


        return ret

    def get_Sigma_WTD(self, i, j, k):
        t = sympy.symbols('t')
        laplace_ij = self.get_real_laplace_wtd(j, i)
        laplace_jk = self.get_real_laplace_wtd(j, k)
        self.get_p_ijk(i, j, k) * sympy.integrate(laplace_ij*sympy.log(laplace_ij/laplace_jk), (t, 0, sympy.oo))

    def get_real_wtd(self):
        t = sympy.symbols('t')
        w_tmp = self.w.copy()
        np.fill_diagonal(w_tmp, 0)
        return w_tmp * np.exp(-t / np.sum(w_tmp, axis=0))

    def get_real_laplace_wtd(self, i, j):
        return self.L(self.get_real_wtd(i, j))

    @staticmethod
    def L(f):
        t, s = sympy.symbols('t, s')
        return sympy.laplace_transform(f, t, s, noconds=True)

    @staticmethod
    def invL(F):
        t, s = sympy.symbols('t, s')
        return sympy.inverse_laplace_transform(F, s, t, noconds=True)

    @property
    def n_matrix(self):
        return self.w.T*self.steady_state

    @property
    def n_matrix_observed(self):
        n_matrix = self.n_matrix
        n_matrix_observed = np.zeros((self.n_observed, self.n_observed))
        tmp_matrix = np.zeros((self.n_observed, self.n))
        for obs in range(self.n_observed):
            real_list = self._observed_to_real[obs]
            row = n_matrix[real_list, :].sum(axis=0)
            tmp_matrix[obs, :] = row
        for obs in range(self.n_observed):
            real_list = self._observed_to_real[obs]
            col = tmp_matrix[:, real_list].sum(axis=1)
            n_matrix_observed[:, obs] = col
        return n_matrix_observed

    def get_p_ij(self, i, j):
        w_tmp = self.w.copy()
        np.fill_diagonal(w_tmp, 0)
        p_ij_matrix = w_tmp/w_tmp.sum(axis=0)
        return p_ij_matrix[j, i]

    def get_n_ijk(self, i, j, k, obs=False):
        if obs:
            ret = 0
            i_states = self._observed_to_real[i]
            j_states = self._observed_to_real[j]
            k_states = self._observed_to_real[k]
            for I in i_states:
                for J in j_states:
                    for K in k_states:
                        ret += self.get_n_ijk(I, J, K)
        else:
            n_tmp = self.n_matrix
            np.fill_diagonal(n_tmp, 0)
            ret = n_tmp[i, j]*self.get_p_ij(j, k)
        return ret


class Trajectory(list):
    # n is the number of micro states
    # n_observed is the number of observed(coarse-grained) states
    _real_to_observed = None  # A dictionary with the real state as key and the observed state as value
    _w = None  # The rate matrix. (n, n) numpy array.
    _jumpProbabilities = None  # The probability to jump from state j to state i. (n, n) numpy array.
    _observed_to_real = None  # A dictionary with the observed state as key and the possible real states list as value.
    _total_time_list = None  # THe total time spent in each state. (n_observed, n_observed) numpy array
    _counter_list = None  # The count of each observed state occurrence. (n_observed, 1) numpy array
    _jump_counter = None  # The total observed jumps: i->j. (n_observed, n_observed) numpy array
    _time_matrix = None  # A 3D matrix with a list of 2nd order waiting times in each element. (n_observed, n_observed, n_observed) numpy array, dtype=list
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
        return list(set(self.real_to_observed.values()))

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
    def total_time(self):
        return np.sum(self._total_time_list)

    @property
    def mean_time(self):
        return np.divide(self._total_time_list, self._counter_list, out=np.zeros_like(self._total_time_list), where=(self._counter_list != 0))

    @property
    def jumpProbabilities(self):
        return self._jumpProbabilities

    @property
    def sigma(self):
        """
        Calculates the fluctuating steady state entropy

        :return:
        """
        ret = np.log(self.steady_state[self[0][0]]/self.steady_state[self[-1][0]])[0]
        tmp = np.multiply((self._jump_counter-self._jump_counter.T), np.log(np.divide(self.w, self.w.T)))
        ret += np.sum(tmp)/2.0
        return ret

    def __init__(self, real_to_observed, w, steady_state, initial_state_index):
        """

        :param real_to_observed: A dictionary with the real state as key and the observed state as value
        :param w: The rate matrix
        :param steady_state: The steady state probabilities
        :param initial_state_index: The index of the initial state
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

        self._total_time_list = np.zeros((self.n_observed, 1))
        self._counter_list = np.zeros((self.n_observed, 1))
        self._jump_counter = np.zeros((self.n_observed, self.n_observed))

        t = np.random.exponential(1 / -self.w[initial_state_index, initial_state_index])
        self._tmp_t = t
        initial_state_observed = self._real_to_observed[initial_state_index]
        self._total_time_list[initial_state_observed] += t
        self._counter_list[initial_state_observed] += 1

        state = State(index=initial_state_index,
                      t=t
                      )

        self.append(state)

    def jump_state(self):
        """
        Performs a jump to a new state

        :return:
        """
        current_state = self[-1]
        new_state_index = np.random.choice(np.arange(self.n), p=self.jumpProbabilities[:, current_state.index])
        t = np.random.exponential(1 / -self.w[new_state_index, new_state_index])
        new_state = State(index=new_state_index,
                          t=t
                          )

        current_observed = self.real_to_observed[current_state.index]
        new_observed = self.real_to_observed[new_state.index]

        self._total_time_list[new_observed] += t
        if current_observed != new_observed:
            if self._prev_observed is not None:
                # import pdb
                # pdb.set_trace()
                self._time_matrix[self._prev_observed][current_observed][new_observed].append(self._tmp_t)
            self._tmp_t = t
            self._counter_list[new_observed] += 1
            self._jump_counter[new_observed, current_observed] += 1
            self._prev_observed = current_observed
        else:
            if self._prev_observed is not None:
                self._tmp_t += t

        self.append(new_state)
        # return new_state

    def plot(self, real=False):
        """
        Plots the trajectory

        :param real: A flag. If True plots the real trajectory, if False plots the observed trajectory. Default False.
        :return:
        """
        N = len(self)

        T = 0
        state = self[0]
        index = state.index if real else self.real_to_observed[state.index]
        plt.figure()
        plt.hlines(index + 1, T, T + state.t)
        T += state.t
        for i in range(1, N):
            prev_index = index
            state = self[i]
            index = state.index if real else self.real_to_observed[state.index]

            plt.vlines(T, prev_index + 1, index + 1, linestyles=':')
            plt.hlines(index + 1, T, T + state.t)
            T += state.t

        plt.xlim(0, T)
        plt.ylim(0, self.n + 1 if real else self.n_observed+1)

        yTicks = [y+1 for y in range(self.n)] if real else [y+1 for y in range(self.n_observed)]
        yLabels = [str(y) for y in yTicks] if real else [f'observed-{y}' for y in yTicks]

        plt.yticks(yTicks, yLabels)

        plt.xlabel('Time')
        plt.ylabel('State')
        # plt.show()

    def estimate_from_statistics(self):
        """
        Estimates the rate matrix and the steady state from the trajectory statistics.

        :return: w, steady_state
        """
        lamda = np.divide(1, self.mean_time, out=np.zeros_like(self.mean_time), where=(self.mean_time != 0))
        total_jumps = np.sum(self._jump_counter, axis=0, keepdims=True)
        w = self._jump_counter * np.divide(lamda.T, total_jumps, out=np.zeros_like(total_jumps), where=(total_jumps != 0))
        np.fill_diagonal(w, -lamda)
        steady_state = self._total_time_list / self.total_time
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
        return self._jump_counter[j, i]/np.sum(self._jump_counter) * self._get_p_ij_to_jk(i, j, k)

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
        # w_est, p_est = self.estimate_from_statistics()
        # n_est = w_est.T*p_est
        # return n_est[i, j]*self._get_p_ij_to_jk(i, j, k)
        return len(self._time_matrix[i, j, k])/self.total_time

    def _get_n_matrix(self):
        return self._jump_counter.T/self.total_time


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
