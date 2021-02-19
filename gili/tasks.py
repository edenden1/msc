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

    def sample_trajectory(self, N, initialState=0, n_hidden=0):
        """
        Samples a trajectory

        :param N: The trajectory size
        :param initialState: The initial state of the trajectory
        :param n_hidden: The number of hidden states
        :return:
        """
        self._trajectory = Trajectory(self.n, self.w, self.steady_state, initialState, n_hidden)

        if n_hidden == 0:
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

        if self._trajectory.n_hidden > 1:
            raise Exception("Can't compute convergence with hidden layers")

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
    _n_hidden = None  # The number of the hidden states. int.
    _jumpProbabilities = None  # The probability to jump from state j to state i. (n, n) numpy array.
    _counterList = None  # The count of each state. (n-n_hidden+1, 1) or (n, 1) numpy array
    _timeList = None  # The count of total time in each state. (n-n_hidden+1, 1) or (n, 1) numpy array.
    _jumpCounter = None  # The count of the jumps from each state to each state. (n-n_hidden+1, n-n_hidden+1) or (n, n,) numpy array.

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
        self._w = w
        self._steady_state = steady_state
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
                          hidden=new_state_index >= self.n_observed
                          )
        if current_state.hidden and new_state.hidden:
            new_state.add_time(current_state.t)
            self._timeList[self.n_observed] += t
            self[-1] = new_state
        else:
            current_index = self.n_observed if current_state.hidden else current_state.index
            new_index = self.n_observed if new_state.hidden else new_state.index

            self._timeList[new_index] += t
            self._counterList[new_index] += 1
            self._jumpCounter[new_index, current_index] += 1

            self.append(new_state)
        # return new_state

    def plot(self):
        """
        Plots the trajectory

        :return:
        """
        N = len(self)

        T = 0
        state = self[0]
        index = self.n_observed if state.hidden else state.index
        plt.figure()
        plt.hlines(index + 1, T, T + state.t)
        T += state.t
        for i in range(1, N):
            prev_index = index
            state = self[i]
            index = self.n_observed if state.hidden else state.index

            plt.vlines(T, prev_index + 1, index + 1, linestyles=':')
            plt.hlines(index + 1, T, T + state.t)
            T += state.t

        plt.xlim(0, T)
        plt.ylim(0, self.n + 1 if self.n_hidden == 0 else self.n_observed+2)

        plt.xlabel('Time')
        plt.ylabel('State')
        plt.show()

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

    @property
    def index(self):
        return self._index

    @property
    def t(self):
        return self._t

    @property
    def hidden(self):
        return self._hidden

    def __init__(self, index, t, hidden=False):
        """

        :param index: The index of representing the state
        :param t: The amount of time spent in the micro state
        :param hidden: A flag. True if the state is hidden
        """
        self._index = index
        self._t = t
        self._hidden = hidden

    def add_time(self, t):
        """
        Adding time spent in the micro state

        :param t: The amount of time to add.
        :return:
        """
        self._t = self.t+t


if __name__ == '__main__':
    n = 10
    model = Model(n)

    N = 1000000
    model.sample_trajectory(N)
    w, steady_state = model.trajectory.estimate_from_statistics()
