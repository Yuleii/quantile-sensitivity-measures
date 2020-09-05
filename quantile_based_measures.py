"""Calculate quantile based global sensitivity measures.

This module contains functions to calculate global sensitivity measures based on
quantiles of the output introduced by Kucherenko et al.(2019).

TODO:
    - Correct the sampling methods for exponential distribution and multivariate distribution.

"""
import numpy as np
import chaospy as cp

from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import expon


def MCS_quantile(objfun, N, M, dim, loc, scale, dist_type, skip=0):
    """Compute Monte Carlo estimates of quantile based global sensitivity measures.

    This function implements the Double loop reordering(DLR) approach described in
    [Section 4.2] of Kucherenko et al.(2019).

    Parameters
    ----------
    objfun: callable
        Objective function to calculate the quantile based measures. Must be broadcastable.

    N: int
        Number of sampled points. This will later turn into the number of Monte Carlo draws.
        To preserve their uniformity properties N should always be equal to N = 2^p, where p
        is an integer.

    M: int
        Number of conditional samples.

    dim: int
        Number of parameters of objective function.

    loc: np.ndarray or float
        Shift of "standardized" distribution which corresponds to the location(loc)
        keyword in Scipy Package.

    scale: np.ndarray or float
        Scale of "standardized" distribution which corresponds to the scale(scale)
        keyword in Scipy Package. Specifically, for normal distribution it specifies
        the covariance matrix.

    dist_type: str
        The distribution type of input. Options are "Normal", "Exponential" and "Uniform".

    skip: int
        Number of values to skip of Sobol sequence. Default is 0.

    Returns
    -------
    q1_alp: np.ndarray
        Quantile based measure.

    q2_alp: np.ndarray
        Quantile based measure.

    Q1_alp: np.ndarray
        Nomalized quantile based measure.

    Q2_alp: np.ndarray
        Nomalized quantile based measure.
    """
    # range of alpha
    dalp = (0.98 - 0.02) / 30
    alp = np.arange(0.02, 0.98 + dalp, dalp)

    # Get quantile based measure
    q1_alp, q2_alp = _quantile_based_measures(
        objfun, N, M, loc, scale, dim, dist_type, alp, skip=0)

    # Get nomalizeduantile based measure
    Q1_alp, Q2_alp = _nomalized_quantile_based_measures(
        objfun, N, M, loc, scale, dim, dist_type, alp, skip=0)

    return q1_alp, q2_alp, Q1_alp, Q2_alp


def _get_unconditional_sample(N, M, loc, scale, dim, dist_type, skip=0):
    # Generate uniform distributed sample
    A = np.zeros((N, dim))
    X001 = cp.generate_samples(order=N + skip, domain=dim, rule="S").T
    X01 = X001[skip:, :dim]

    # Transform uniform draw into assigned joint PDF
    if dist_type == "Normal":
        X1 = norm.ppf(X01)
        cholesky = np.linalg.cholesky(scale)
        A = loc + cholesky.dot(X1.T).T
    elif dist_type == "Exponential":
        A = expon.ppf(X01, loc, scale)
    elif dist_type == "Uniform":
        A = uniform.ppf(X01, loc, scale)
    else:
        raise NotImplementedError

    return A


def _get_conditional_sample(N, M, loc, scale, dim, skip, dist_type):
    # conditional sample matrix C,with shape the of (64, 4, 8192, 4)
    A = _get_unconditional_sample(N, M, loc, scale, dim, dist_type, skip=0)
    B = A[:M]
    C = np.array([[np.zeros((N, dim)) for x in range(dim)]
                  for z in range(M)], dtype=np.float64)
    for i in range(dim):
        for j in range(M):
            C[j, i] = A
            C[j, i, :, i] = B[j, i]

    return C


def _unconditional_q_Y(objfun, N, M, loc, scale, dim, dist_type, alp, skip=0):
    A = _get_unconditional_sample(N, M, loc, scale, dim, dist_type, skip=0)
    # values of outputs
    Y1 = objfun(A)
    # reorder in ascending order
    y1 = np.sort(Y1)

    # q_Y(alp)
    q_index = (np.floor(alp * N) - 1).astype(int)
    qy_alp1 = y1[q_index]

    return qy_alp1


def _conditional_q_Y(objfun, N, M, loc, scale, dim, dist_type, alp, skip=0):
    C = _get_conditional_sample(
        N, M, loc, scale, dim, dist_type, skip=0)  # shape(64, 4, 8192, 4)

    # initialize values of conditional outputs.
    Y2 = np.array([[np.zeros((N, 1)) for x in range(dim)]
                   for z in range(M)], dtype=np.float64)  # shape(8192, 4, 8192, 1)
    y2 = np.array([[np.zeros((N, 1)) for x in range(dim)]
                   for z in range(M)], dtype=np.float64)

    # initialize quantile of conditional outputs.
    qy_alp2 = np.array([[np.zeros((len(alp), M)) for x in range(dim)]
                        for z in range(1)], dtype=np.float64)  # shape(1, 4, 31, 64)

    for i in range(dim):
        for j in range(M):
            # values of conditional outputs
            Y2[j, i] = np.vstack(objfun(C[j, i]))
            Y2[j, i].sort(axis=0)
            y2[j, i] = Y2[j, i]  # reorder in ascending order
            for pp in range(len(alp)):
                # conditioanl q_Y(alp)
                qy_alp2[0, i, pp, j] = y2[j, i][(
                    np.floor(alp[pp] * N) - 1).astype(int)]

    return qy_alp2


def _quantile_based_measures(objfun, N, M, loc, scale, dim, dist_type, alp, skip=0):

    qy_alp1 = _unconditional_q_Y(
        objfun, N, M, loc, scale, dim, dist_type, alp, skip=0)
    qy_alp2 = _conditional_q_Y(
        objfun, N, M, loc, scale, dim, dist_type, alp, skip=0)

    # initialization
    q1_alp = np.zeros((len(alp), dim))
    q2_alp = np.zeros((len(alp), dim))
    delt = np.array([[np.zeros((1, M)) for x in range(dim)]
                     for z in range(1)], dtype=np.float64)

    for i in range(dim):
        for pp in range(len(alp)):
            delt[0, i] = qy_alp2[0, i, pp, :] - qy_alp1[pp]  # delt
            q1_alp[pp, i] = np.mean(np.absolute(delt[0, i]))  # |delt|
            q2_alp[pp, i] = np.mean(delt[0, i] ** 2)  # (delt)^2

    return q1_alp, q2_alp


def _nomalized_quantile_based_measures(objfun, N, M, loc, scale, dim, dist_type, alp, skip=0):

    q1_alp, q2_alp = _quantile_based_measures(
        objfun, N, M, loc, scale, dim, dist_type, alp, skip=0)

    # initialize quantile measures arrays.
    q1 = np.zeros(len(alp))
    q2 = np.zeros(len(alp))
    Q1_alp = np.zeros((len(alp), dim))
    Q2_alp = np.zeros((len(alp), dim))

    for pp in range(len(alp)):
        q1[pp] = np.sum(q1_alp[pp, :])
        q2[pp] = np.sum(q2_alp[pp, :])
        for i in range(dim):
            Q1_alp[pp, i] = q1_alp[pp, i] / q1[pp]
            Q2_alp[pp, i] = q2_alp[pp, i] / q2[pp]

    return Q1_alp, Q2_alp
