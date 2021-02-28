import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class Model:
    _n = None  # The number of states. A scalar.
    _w = None  # The rate matrix. (n, n) numpy array.
    _steady_state = None  # The steady state from the numeric calculations. (n, 1) numpy array.
    _steady_state_stalling = None  # The steady state at stalling. (n, 1) numpy array.
    _steady_state_J = None  # The steady state current. (n, n) numpy array.
    _steady_state_Sigma = None  # The steady state entropy production rate. (n, n) numpy array.
    _trajectory = None  # The latest sampled trajectory.
    _cache = None  #
    _dt = None  # The time delta for the numeric calculations

    @property
    def n(self):
        return self._n

    @property
    def w(self):
        if self._w is None:
            self._w = self._initialize_w()
        return self._w

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
    def infromed_partial_Sigma(self):
        """
        The informed partial entropy production

        :return:
        """
        # w_mul_p_st_T = np.multiply(self.w, self.steady_state_stalling.T)
        # np.fill_diagonal(w_mul_p_st_T, 1)
        # return np.sum(np.multiply(self.steady_state_J, np.log(w_mul_p_st_T)))
        return (self.steady_state_J[0, 1] * np.log(self.w[0, 1] * self.steady_state_stalling[1] / (self.w[1, 0] * self.steady_state_stalling[0])))[0]

    def __init__(self, n, w=None, dt=0.001):
        """

        :param n: The number of states
        :param w: The rate matrix. If not provided, it will be initialized when needed
        :param dt: The time delta for the numeric calculations
        """
        self._n = n
        if w is not None:
            if w.shape[0] != w.shape[1] or w.shape[0] != n:
                raise Exception("w must be in shape (n, n)")
            else:
                self._w = w
        self._dt = dt
        self._cache = {}

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

    def _numeric_one_step(self, p_prev, dt):
        """
        Executes one time step

        :param p_prev: The previous probabilities vector
        :param dt: The time delta
        :return: The new probabilities vector
        """
        p_new = p_prev + np.dot(self.w, p_prev)*dt
        return p_new

    def _numeric_one_step_stalling(self, p_prev, dt):
        """
        Executes one time step at stalling

        :param p_prev: The previous probabilities vector
        :param dt: The time delta
        :return: The new probabilities vector
        """
        w = self.w.copy()
        w[0, 0] += w[1, 0]
        w[1, 0] = 0
        w[1, 1] += w[0, 1]
        w[0, 1] = 0
        p_new = p_prev + np.dot(w, p_prev)*dt
        return p_new

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
            p = self._numeric_one_step(p, dt)
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

        for i in range(steps):
            p = self._numeric_one_step_stalling(p, dt)
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

    def sample_trajectory(self, N, initialState=0, n_hidden=0):
        """
        Samples a trajectory

        :param N: The trajectory size
        :param initialState: The initial state of the trajectory
        :param n_hidden: The number of hidden states
        :return:
        """
        self._trajectory = Trajectory(self.n, self.w, self.steady_state, initialState, n_hidden)

        rates_convergence = []
        steady_state_convergence = []

        j = 1
        for i in range(1, N+1):
            self.trajectory.jump_state()
            if n_hidden == 0:
                if i == 10**j:
                    w, steady_state = self.trajectory.estimate_from_statistics()
                    rates_convergence.append(np.linalg.norm(w-self.w))
                    steady_state_convergence.append(np.linalg.norm(steady_state-self.steady_state))
                    j += 1
        if n_hidden == 0:
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

        if self._trajectory.n_hidden > 1:
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

        if self._trajectory.n_hidden > 1:
            raise Exception("Can't compute convergence with hidden layers")

        plt.figure()
        steady_state_conv = self._cache['steady_state_convergence']
        plt.plot([10**(i + 1) for i in range(len(steady_state_conv))], steady_state_conv)
        plt.xscale('log')
        plt.title('The difference between the estimated steady state to the real one')
        plt.xlabel('Trajectory size (Log10 scale)')
        plt.ylabel('Difference')
        # plt.show()

    def plot_trajectory(self, raw=False):
        """
        Plots the last sampled trajectory

        :param raw: A flag to plot the real trajectory, ignoring the hidden status
        :return:
        """
        if self._trajectory is None:
            raise Exception('No trajectory was sampled')

        self.trajectory.plot(raw)


class Trajectory(list):
    _n = None  # The number of states. int.
    _w = None  # The rate matrix. (n, n) numpy array.
    _n_hidden = None  # The number of the hidden states. int.
    _jumpProbabilities = None  # The probability to jump from state j to state i. (n, n) numpy array.
    _counterList = None  # The count of each state. (n-n_hidden+1, 1) or (n, 1) numpy array
    _counter_matrix = None  # The count of each jump: prev_state -> new_state. (n-n_hidden+1, n-n_hidden+1) or (n, n) numpy array
    _timeList = None  # The count of total time in each state. (n-n_hidden+1, 1) or (n, 1) numpy array.
    _jumpCounter = None  # The count of the jumps from each state to each state. (n-n_hidden+1, n-n_hidden+1) or (n, n) numpy array.

    @property
    def n(self):
        return self._n

    @property
    def w(self):
        return self._w

    @property
    def n_hidden(self):
        return self._n_hidden

    @property
    def n_observed(self):
        return self.n-self.n_hidden

    @property
    def steady_state(self):
        return self._steady_state

    @property
    def totalTime(self):
        return np.sum(self._timeList)

    @property
    def meanTime(self):
        return np.divide(self._timeList, self._counterList, out=np.zeros_like(self._timeList), where=(self._counterList != 0))

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
        tmp = np.multiply((self._jumpCounter-self._jumpCounter.T), np.log(np.divide(self.w, self.w.T)))
        ret += np.sum(tmp)/2.0
        return ret

    def __init__(self, n, w, steady_state, initialStateIndex, n_hidden=0):
        """

        :param n: The number of states
        :param w: The rate matrix
        :param steady_state: The steady state probabilities
        :param initialStateIndex: The index of the initial state
        :param hidden: The number of hidden layers
        """
        super().__init__()
        self._n = n
        self._w = w.copy()
        self._steady_state = steady_state.copy()
        self._n_hidden = n_hidden

        self._jumpProbabilities = self.w/(-np.diagonal(self.w)).reshape(1, self.n)
        np.fill_diagonal(self._jumpProbabilities, 0)

        arrays_size = n if self.n_hidden == 0 else self.n_observed+1
        self._timeList = np.zeros((arrays_size, 1))
        self._counterList = np.zeros((arrays_size, 1))
        self._jumpCounter = np.zeros((arrays_size, arrays_size))

        t = np.random.exponential(1 / -self.w[initialStateIndex, initialStateIndex])
        self._timeList[initialStateIndex] += t
        self._counterList[initialStateIndex] += 1

        state = State(index=initialStateIndex,
                      t=t,
                      hidden=(initialStateIndex >= self.n_observed)
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
                          t=t,
                          hidden=new_state_index >= self.n_observed,
                          prev_state=current_state
                          )
        current_index = self.n_observed if current_state.hidden else current_state.index
        new_index = self.n_observed if new_state.hidden else new_state.index

        self._timeList[new_index] += t
        if not(current_state.hidden and new_state.hidden):
            self._counterList[new_index] += 1
            self._jumpCounter[new_index, current_index] += 1

        self.append(new_state)
        # return new_state

    def plot(self, raw=False):
        """
        Plots the trajectory

        :param raw: A flag to plot the real trajectory, ignoring the hidden status
        :return:
        """
        N = len(self)

        T = 0
        state = self[0]
        index = self.n_observed if state.hidden and not raw else state.index
        plt.figure()
        plt.hlines(index + 1, T, T + state.t)
        T += state.t
        for i in range(1, N):
            prev_index = index
            state = self[i]
            index = self.n_observed if state.hidden and not raw else state.index

            plt.vlines(T, prev_index + 1, index + 1, linestyles=':')
            plt.hlines(index + 1, T, T + state.t)
            T += state.t

        plt.xlim(0, T)
        plt.ylim(0, self.n + 1 if self.n_hidden == 0 or raw else self.n_observed+2)

        yTicks = [y+1 for y in range(self.n)] if self.n_hidden == 0 or raw else [y+1 for y in range(self.n_observed+1)]
        yLabels = [str(y) for y in yTicks] if self.n_hidden == 0 or raw else [str(y) for y in yTicks[:-1]]+['H']

        plt.yticks(yTicks, yLabels)

        plt.xlabel('Time')
        plt.ylabel('State')
        # plt.show()

    def estimate_from_statistics(self):
        """
        Estimates the rate matrix and the steady state from the trajectory statistics.

        :return: w, steady_state
        """
        lamda = np.divide(1, self.meanTime, out=np.zeros_like(self.meanTime), where=(self.meanTime != 0))
        total_jumps = np.sum(self._jumpCounter, axis=0, keepdims=True)
        w = self._jumpCounter * np.divide(lamda.T, total_jumps, out=np.zeros_like(total_jumps), where=(total_jumps != 0))
        np.fill_diagonal(w, -lamda)
        steady_state = self._timeList / self.totalTime
        return w, steady_state


class State:
    _index = None
    _t = None
    _hidden = None
    _prev_state = None

    @property
    def index(self):
        return self._index

    @property
    def t(self):
        return self._t

    @property
    def hidden(self):
        return self._hidden

    @property
    def prev_state(self):
        return self._prev_state

    @property
    def prev_index(self):
        return None if self.prev_state is None else self._prev_state.index

    def __init__(self, index, t, hidden=False, prev_state=None):
        """

        :param index: The index of representing the state
        :param t: The amount of time spent in the micro state
        :param hidden: A flag. True if the state is hidden
        :param prev_state: The previous state
        """
        self._index = index
        self._t = t
        self._hidden = hidden
        self._prev_state = prev_state

    def add_time(self, t):
        """
        Adding time spent in the micro state

        :param t: The amount of time to add.
        :return:
        """
        self._t = self.t+t
