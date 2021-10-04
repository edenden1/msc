from project import *
from scipy.optimize import minimize

"""
Getting my estimator for the 2-hidden states system
"""


def get_stats(trj):
    pass


def epr1(w12, w13, w14, w21, w23, w24, w31, w32, w34, w41, w42, w43, p3, p4):
    w = np.array([[0, w12, w13, w14],
                  [w21, 0, w23, w24],
                  [w31, w32, 0, w34],
                  [w41, w42, w43, 0]
                  ], dtype=float
                 )


def epr2(n):
    _tol = 1e-10
    ret = 0
    for i in range(n.shape[0]):
        for j in range(i+1, n.shape[0]):
            ret += (n[i, j] - n[j, i]) * np.log((n[i, j] + _tol) / (n[j, i] + _tol))
    return ret


def calc_bound(trj):
    pass


if __name__ == '__main__':
    real_to_observed = {0: 0,
                        1: 1,
                        2: 2,
                        3: 2
                        }

    w = np.array([[0, 2, 0, 1],
                  [3, 0, 2, 35],
                  [0, 50, 0, 0.7],
                  [8, 0.2, 75, 0]], dtype=float)

    N = 10**6

    trj = TrajectorySigma2(real_to_observed, w, N)
