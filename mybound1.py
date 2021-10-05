import numpy as np

from project import *
from scipy.optimize import minimize

"""
Getting my estimator for the 2-hidden states system
"""


def get_conf(trj):
    conf = {}
    conf['n_IJ'], n_IJK = trj.get_sigma2_stats()
    conf['n1H2'], conf['n2H1'] = n_IJK[0, 2, 1], n_IJK[1, 2, 0]
    conf['p1'], conf['p2'], conf['pH'] = trj.get_steady_state()
    conf['t1'], conf['t2'], conf['tH'] = trj.get_mean_waiting_times()
    return conf


def get_epr(n):
    _tol = 1e-10
    ret = 0
    for i in range(n.shape[0]):
        for j in range(i+1, n.shape[0]):
            ret += (n[i, j] - n[j, i]) * np.real(np.log((n[i, j] + _tol) / (n[j, i] + _tol) + 0j))
    return ret


def calc_bound(conf):
    n_IJ = conf['n_IJ']
    n12 = n_IJ[0, 1]
    n21 = n_IJ[1, 0]
    n1H = n_IJ[0, 2]  # = n13 + n14 = (w31 + w41) * p1
    n2H = n_IJ[1, 2]  # = n23 + n24 = (w32 + w42) * p2
    nH1 = n_IJ[2, 0]  # = n31 + n41 = w13*p3 + w14*p4
    nH2 = n_IJ[2, 1]  # = n32 + n42 = w23*p3 + w24*p4
    n1H2 = conf['n1H2']
    n2H1 = conf['n2H1']
    t1 = conf['t1']
    t2 = conf['t2']
    tH = conf['tH']
    p1 = conf['p1']
    p2 = conf['p2']
    pH = conf['pH']
    # import pdb
    # pdb.set_trace()
    assert(1 - (p1 + p2 + pH) < 1e-10)
    w12 = n21 / p2
    w21 = n12 / p1

    _tol = 1e-15

    def get_w_and_p(x):
        w13, w23, w31, w32, w34, w43, p3 = x
        p4 = pH - p3
        w14 = (nH1 - w13 * p3) / (p4 + _tol)
        w24 = (nH2 - w23 * p3) / (p4 + _tol)
        w41 = n1H / p1 - w31
        w42 = n2H / p2 - w32

        p = np.array([p1, p2, p3, p4])
        w = np.array([[0, w12, w13, w14],
                      [w21, 0, w23, w24],
                      [w31, w32, 0, w34],
                      [w41, w42, w43, 0]
                      ], dtype=float)
        np.fill_diagonal(w, -np.sum(w, axis=0).squeeze())
        return w, p

    def epr(x):
        w, p = get_w_and_p(x)
        p = p.reshape((1, -1))
        n = (w*p).T
        return get_epr(n)

    def epr_jac(x):
        w, p = get_w_and_p(x)

        def f_jac_w(i, j):
            wij = w[i-1, j-1]
            wji = w[j-1, i-1]
            pi = p[i-1]
            pj = p[j-1]
            return (wij*pj - wji*pi) / (wij + _tol) + pj*np.real(np.log((wij*pj + _tol) / (wji*pi + _tol) + 0j))

        def f_jac_pj(j):
            ret = 0
            for i in range(1, 5):
                if i != j:
                    wij = w[i - 1, j - 1]
                    wji = w[j - 1, i - 1]
                    pi = p[i - 1]
                    pj = p[j - 1]
                    ret += (wij*pj - wji*pi) / (pj + _tol) + wij*np.real(np.log((wij*pj + _tol) / (wji*pi + _tol) + 0j))

        jac = [f_jac_w(1, 3), f_jac_w(2, 3), f_jac_w(3, 1), f_jac_w(3, 2), f_jac_w(3, 4), f_jac_w(4, 3), f_jac_pj(3)]
        return jac

    def cons(x, name):
        w, p = get_w_and_p(x)
        t3 = 1 / (w[0, 2] + w[1, 2] + w[3, 2] + _tol)
        t4 = 1 / (w[0, 3] + w[1, 3] + w[2, 3] + _tol)
        denominator = 1 - w[2, 3] * w[3, 2] * t3 * t4 + _tol
        if name == 't1':
            con = t1 - 1 / (w[1, 0] + w[2, 0] + w[3, 0] + _tol)
        elif name == 't2':
            con = t2 - 1 / (w[0, 1] + w[2, 1] + w[3, 1] + _tol)
        elif name == 'tH':
            t3_p = t3 * (1 + w[3, 2] * t4) / denominator
            t4_p = t4 * (1 + w[2, 3] * t3) / denominator
            con = tH - (p[2] * t3_p + p[3] * t4_p) / (p[2] + p[3] + _tol)
        elif name == 'n1H2':
            con = n1H2 - p[0] * (w[2, 0]*t3*(w[1, 2] + w[3, 2]*w[1, 3]*t4) + w[3, 0]*t4*(w[1, 3] + w[2, 3]*w[1, 2]*t3)) / denominator
        elif name == 'n2H1':
            con = n2H1 - p[1] * (w[2, 1]*t3*(w[0, 2] + w[3, 2]*w[0, 3]*t4) + w[3, 1]*t4*(w[0, 3] + w[2, 3]*w[0, 2]*t3)) / denominator
        else:
            raise NameError('Not a valid name argument')
        return con

    cons_names = ['t1', 't2', 'tH', 'n1H2', 'n2H1']
    cons_list = []
    for con_name in cons_names:
        cons_list.append({'type': 'eq',
                          'fun': lambda x: cons(x, con_name)
                          }
                         )

    bnds = [(0, None), (0, None), (0, n1H/p1), (0, n2H/p2), (0, None), (0, None), (0, pH)]

    def calc():
        rand_mul = np.random.rand(7)
        x0 = np.array([1, 1, n1H/p1, n2H/p2, 1, 1, pH])
        x0 *= rand_mul
        con_tol = 1e-10
        res = minimize(epr, x0, jac=epr_jac, method='SLSQP',
                       options={'maxiter': 1e5, 'ftol': 1e-10}, bounds=bnds,
                       constraints=cons_list, tol=con_tol)
        return epr(res.x), res.status

    ep, status = calc()
    if status != 0:
        ep, status = calc()

    return ep


if __name__ == '__main__':
    real_to_observed = {0: 0,
                        1: 1,
                        2: 2,
                        3: 2
                        }

    W = np.array([[0, 2, 0, 1],
                  [3, 0, 2, 35],
                  [0, 50, 0, 0.7],
                  [8, 0.2, 75, 0]], dtype=float)

    N = 10**6
    ep_list = []
    x_list = np.sort(np.append(np.arange(-3, 5), -0.67))
    for x in x_list:
        w_tmp = W.copy()
        w_tmp[0, 1] *= np.exp(x)
        w_tmp[1, 0] *= np.exp(-x)

        trj = TrajectorySigma2(real_to_observed, w_tmp, N)

        config = get_conf(trj)
        ep = calc_bound(config)
        ep_list.append(ep)
        print(ep)

    pps_list = []
    ips_list = []
    Sigma_list = []
    x_list_analytic = np.sort(np.append(np.linspace(-3, 4, 100), -0.67))
    for x in x_list_analytic:
        print(x)
        w_tmp = W.copy()
        w_tmp[0, 1] *= np.exp(x)
        w_tmp[1, 0] *= np.exp(-x)

        model_tmp = Model(real_to_observed, w=w_tmp, dt=0.0001)
        pps_list.append(model_tmp.passive_partial_Sigma)
        ips_list.append(model_tmp.informed_partial_Sigma)
        Sigma_list.append(model_tmp.steady_state_Sigma)

    print(f'x - {x_list}')
    print(f'Sigma - {Sigma_list}')
    print(f'Mine - {ep_list}')
    print(f'Informed - {ips_list}')

    plt.plot(x_list_analytic, pps_list, label='Passive', c='b')
    plt.plot(x_list_analytic, ips_list, label='Informed', c='g')
    plt.plot(x_list_analytic, Sigma_list, label='Total', c='y')
    plt.scatter(x_list, ep_list, label='My estimator', marker='x', c='r')
    plt.legend()
    plt.xlabel('x')
    plt.yscale('log')
    plt.show()
