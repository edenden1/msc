import numpy as np
import matplotlib.pyplot as plt


class Model:
    _n = None  # The number of states. A scalar.
    _w = None  # The rate matrix. (n, n) numpy array.
    _steady_state = None  # The steady state from the numeric calculations. (n, 1) numpy array.
    _steady_state_J = None  # The steady state current. (n, n) numpy array.
    _steady_state_Sigma = None  # The steady state entropy production rate. (n, n) numpy array.
    _trajectory = None  # The latest sampled trajectory.
    _cache = None  #

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
            self._steady_state = self.numeric_steady_state(T=10.0 / self.n)
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
            self._steady_state_Sigma = np.sum(np.multiply(self.steady_state_J, np.log(w_mul_pT)))
        return self._steady_state_Sigma

    def __init__(self, n):
        """

        :param n: The number of states
        """
        self._n = n
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

    def _initialize_trajectory(self):
        """
        Initializes a trajectory.

        :return:
        """

    def _numeric_one_step(self, p_prev, dt):
        """
        Executes one time step

        :param p_prev: The previous probabilities vector
        :param dt: The time delta
        :return: The new probabilities vector
        """
        p_new = p_prev + np.dot(self.w, p_prev)*dt
        return p_new

    def numeric_steady_state(self, dt=0.001, T=10.0, plot_flag=False):
        """
        Calculates the steady state numerically

        :param dt: The time delta
        :param T: The total time
        :param plot_flag: A flag to plot the probabilities over time
        :return:
        """
        p = self._initialize_p()
        steps = int(T/dt)

        p_list = [p]

        for i in range(steps):
            p = self._numeric_one_step(p, dt)
            p_list.append(p)

        if plot_flag:
            for i in range(self.n):
                plt.plot(dt*np.arange(steps+1), np.array(p_list)[:, i], label=f'p{i+1}')
                plt.xlim(0, T)
                # plt.ylim(0, 1)
            plt.legend(loc='upper right')
            plt.xlabel('Time')
            plt.ylabel('Probability')
            plt.show()

        return p

    def sample_trajectory(self, N, initialState=0):
        """
        Samples a trajectory

        :param N: The trajectory size
        :param initialState: The initial state of the trajectory
        :return:
        """
        self._trajectory = Trajectory(self.n, self.w, self.steady_state, initialState)

        rates_convergence = []
        steady_state_convergence = []

        j = 1
        for i in range(1, N+1):
            self.trajectory.jump_state()
            if i == 10**j:
                w, steady_state = self.trajectory.estimate_from_statistics()
                rates_convergence.append(np.linalg.norm(w-self.w))
                steady_state_convergence.append(np.linalg.norm(steady_state-self.steady_state))
                j += 1

        self._cache.update(dict(
                                rates_convergence = rates_convergence,
                                steady_state_convergence = steady_state_convergence
                                )
                           )

    def plot_w_convergence(self):
        """
        Plots the rate matrix convergence

        :return:
        """
        if self._trajectory is None:
            raise Exception('No trajectory was sampled')

        plt.figure()
        w_conv = self._cache['rates_convergence']
        plt.plot([10**(i + 1) for i in range(len(w_conv))], w_conv)
        plt.xscale('log')
        plt.title('The distance between the estimated rate matrix to the real one')
        plt.xlabel('Trajectory size (Log10 scale)')
        plt.ylabel('Distance')
        plt.show()

    def plot_steady_state_convergence(self):
        """
        Plots the rate matrix convergence

        :return:
        """
        if self._trajectory is None:
            raise Exception('No trajectory was sampled')

        plt.figure()
        steady_state_conv = self._cache['steady_state_convergence']
        plt.plot([10**(i + 1) for i in range(len(steady_state_conv))], steady_state_conv)
        plt.xscale('log')
        plt.title('The distance between the estimated steady state to the real one')
        plt.xlabel('Trajectory size (Log10 scale)')
        plt.ylabel('Distance')
        plt.show()

    def plot_trajectory(self):
        """
        Plots the last sampled trajectory

        :return:
        """
        self.trajectory.plot()


class Trajectory(list):
    _n = None  # The number of states. int.
    _w = None  # The rate matrix. (n, n) numpy array.
    _hidden = None  # The number of the hidden states. int.
    _jumpProbabilities = None  # The probability to jump from state j to state i. (n, n) numpy array.
    _counterList = None  # The count of each state. (n, 1) numpy array.
    _timeList = None  # The count of total time in each state. (n, 1) numpy array.
    _jumpCounter = None  # The count of jump from each state to each state. (n, n) numpy array.

    @property
    def n(self):
        return self._n

    @property
    def w(self):
        return self._w

    @property
    def steady_state(self):
        return self._steady_state

    @property
    def totalTime(self):
        return np.sum(self._timeList)

    @property
    def meanTime(self):
        return np.divide(self._timeList, self._counterList, out=np.zeros((self.n, 1)), where=(self._counterList != 0))

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

    def __init__(self, n, w, steady_state, initialState, hidden=0):
        """

        :param n: The number of states
        :param w: The rate matrix
        :param steady_state: The steady state probabilities
        :param initialState: The initial state
        :param hidden: The number of hidden layers
        """
        super().__init__()
        self._n = n
        self._w = w
        self._steady_state = steady_state

        self._jumpProbabilities = self.w/(-np.diagonal(self.w)).reshape(1, self.n)
        np.fill_diagonal(self._jumpProbabilities, 0)

        self._timeList = np.zeros((self.n, 1))
        self._counterList = np.zeros((self.n, 1))
        self._jumpCounter = np.zeros((self.n, self.n))

        t = np.random.exponential(1 / -self.w[initialState, initialState])
        self._timeList[initialState] += t
        self._counterList[initialState] += 1

        self.append((initialState, t))

    def jump_state(self):
        current_state = self[-1][0]
        new_state = np.random.choice(np.arange(self.n), p=self.jumpProbabilities[:, current_state])
        t = np.random.exponential(1 / -self.w[new_state, new_state])
        self._timeList[new_state] += t
        self._counterList[new_state] += 1
        self._jumpCounter[new_state, current_state] += 1

        self.append((new_state, t))
        return new_state, t

    def plot(self):
        """
        Plots the trajectory

        :return:
        """
        N = len(self)

        T = 0
        state, t = self[0]
        plt.figure()
        plt.hlines(state + 1, T, T + t)
        T += t
        for i in range(1, N):
            state, t = self[i]
            plt.vlines(T, self[i - 1][0] + 1, state + 1, linestyles=':')
            plt.hlines(state + 1, T, T + t)
            T += t

        plt.xlim(0, T)
        plt.ylim(0, self.n + 1)

        plt.xlabel('Time')
        plt.ylabel('State')
        plt.show()

    def estimate_from_statistics(self):
        """
        Estimates the rate matrix and the steady state from the trajectory statistics.

        :return: w, steady_state
        """

        lamda = np.divide(1, self.meanTime, out=np.zeros((self.n, 1)), where=(self.meanTime != 0))
        total_jumps = np.sum(self._jumpCounter, axis=0)
        w = self._jumpCounter * np.divide(lamda.T, total_jumps, out=np.zeros((1, self.n)), where=(total_jumps != 0))
        np.fill_diagonal(w, -lamda)
        steady_state = self._timeList / self.totalTime
        return w, steady_state


if __name__ == '__main__':
    n = 10
    model = Model(n)

    N = 1000000
    model.sample_trajectory(N)
    w, steady_state = model.trajectory.estimate_from_statistics()
