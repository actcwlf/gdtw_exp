# Author: Mathieu Blondel
# License: Simplified BSD

import numpy as np

from scipy.optimize import minimize

from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean
from gsfdtw import FSDTW, GFSDTW
import matplotlib.pyplot as plt


def sdtw_barycenter(X, barycenter_init, gamma=1.0, weights=None,
                    method="L-BFGS-B", tol=1e-3, max_iter=50):
    """
    Compute barycenter (time series averaging) under the soft-DTW geometry.

    Parameters
    ----------
    X: list
        List of time series, numpy arrays of shape [len(X[i]), d].

    barycenter_init: array, shape = [length, d]
        Initialization.

    gamma: float
        Regularization parameter.
        Lower is less smoothed (closer to true DTW).

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).

    method: string
        Optimization method, passed to `scipy.optimize.minimize`.
        Default: L-BFGS.

    tol: float
        Tolerance of the method used.

    max_iter: int
        Maximum number of iterations.
    """
    if weights is None:
        weights = np.ones(len(X))

    weights = np.array(weights)

    def _func(Z):
        # Compute objective value and grad at Z.

        Z = Z.reshape(*barycenter_init.shape)

        m = Z.shape[0]
        G = np.zeros_like(Z)

        obj = 0

        for i in range(len(X)):
            D = SquaredEuclidean(Z, X[i])
            sdtw = SoftDTW(D, gamma=gamma)
            value = sdtw.compute()
            # R = sdtw.R_
            # R[R > 1e300] = np.inf
            # R[R < -1e30] = -np.inf
            # t = R[-10:, -10:]
            E = sdtw.grad()
            G_tmp = D.jacobian_product(E)

            # _gamma = 0.5
            # _q = 2
            # x = Z.squeeze()
            # y = X[i].squeeze()
            # d_mat = qsdtw.d_matrix(x, y)
            # value2, r_mat, z_mat, g0_mat, g1_mat, g2_mat = gsdtw.compute_gumbel_softdtw(d_mat, _gamma, _q)
            # e_mat = gsdtw.compute_gumbel_softdtw_backward(d_mat, r_mat, z_mat, g0_mat, g1_mat, g2_mat, _gamma, _q)
            # G_tmp2 = gsdtw.jacobian_product_sq_euc(x, y, e_mat)

            # plt.imshow(E)
            # plt.colorbar()
            # plt.title('soft e_mat')
            # plt.show()

            # plt.imshow(e_mat)
            # plt.colorbar()
            # plt.title('gumbel e_mat')
            # plt.show()
            # plt.plot(G_tmp, label='softdtw')
            # plt.plot(G_tmp2, label='gumbel')
            # plt.legend()
            # plt.show()
            # print('show')

            G += weights[i] * G_tmp
            obj += weights[i] * value
        return obj, G.ravel()

    # The function works with vectors so we need to vectorize barycenter_init.
    res = minimize(_func, barycenter_init.ravel(), method=method, jac=True,
                   tol=tol, options=dict(maxiter=max_iter, disp=False))

    return res.x.reshape(*barycenter_init.shape)



def gfsdtw_barycenter(X, barycenter_init, gamma=1.0, q=100, weights=None, radius=1,
                    method="L-BFGS-B", tol=1e-3, max_iter=50):

    if weights is None:
        weights = np.ones(len(X))

    weights = np.array(weights)

    def _func(Z):
        # Compute objective value and grad at Z.

        Z = Z.reshape(*barycenter_init.shape)

        m = Z.shape[0]
        G = 0 # np.zeros_like(Z)

        obj = 0
        fsdtw_i = GFSDTW(gamma=gamma, q=q, radius=radius)
        # fsdtw_i.
        for i in range(len(X)):

            v = fsdtw_i.forward(Z.squeeze(), X[i].squeeze())
            g = fsdtw_i.backward()

            # x = Z.squeeze()
            # y = X[i].squeeze()
            # d_mat = qsdtw.d_matrix(x, y)
            # value, r_mat, z_mat, g0_mat, g1_mat, g2_mat = gsdtw.compute_gumbel_softdtw(d_mat, gamma, q)
            # e_mat = gsdtw.compute_gumbel_softdtw_backward(d_mat, r_mat, z_mat, g0_mat, g1_mat, g2_mat, gamma, q)
            # G_tmp = gsdtw.jacobian_product_sq_euc(x, y, e_mat)


            G += g
            obj += v
        return obj, G.ravel()

    # The function works with vectors so we need to vectorize barycenter_init.
    res = minimize(_func, barycenter_init.ravel(), method=method, jac=True,
                   tol=tol, options=dict(maxiter=max_iter, disp=False))

    return res.x.reshape(*barycenter_init.shape)