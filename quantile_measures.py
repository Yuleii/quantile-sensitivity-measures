"""The (Quasi) Monte Carlo estimators of quantile based global sensitivity measures.

This module contains functions to calculate global sensitivity measures based on
quantiles of the output introduced by Kucherenko et al.(2019). Both the brute force
estimator and double loop reordering approach are included.

"""
import chaospy as cp
import numpy as np
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import uniform


def bf_mcs_quantile(
    func,
    n_params,
    loc,
    scale,
    dist_type,
    n_draws,
    sampling_scheme="sobol",
    seed=0,
    skip=0,
):
    r"""Compute the brute force MC/QMC estimators of quantile based global sensitivity measures.

    This function compute the brute force estimator described in Section 4.1 of [K2019]_.

    Parameters
    ----------
    func : callable
        Objective function to estimate the quantile-based measures. Must be broadcastable.

    n_params : int
        Number of parameters of objective function.

    loc : float or np.ndarray
        The location(`loc`) keyword passed to `scipy.stats.norm`_ function to shift the
        location of "standardized" distribution. Specifically, for normal distribution
        it specifies the mean with the length of `n_params`.

        .. _scipy.stats.norm: https://docs.scipy.org/doc/scipy/reference/generated/
            _scipy.stats.norm.html

    scale : float or np.ndarray
        The `scale` keyword passed to `scipy.stats.norm`_ function to adjust the scale of
        "standardized" distribution. Specifically, for normal distribution it specifies
        the covariance matrix of shape (n_params, n_params).

    dist_type : str
        The distribution type of inputs. Options are "Normal", "Exponential" and "Uniform".

    n_draws : int
        Number of brute force Monte Carlo draws for the estimation of the sensitivity measures.

    sampling_scheme : str, optional
        One of ["random", "sobol"], default "sobol".

    seed : int, optional
        Random number generator seed.

    skip : int, optional
        Number of values to skip of Sobol sequence. Default is `0`.

    Returns
    -------
    q_1 : np.ndarray
        The brute force estimator of quantile based measure :math:`\tilde{q_i}^{(1)}(\alpha)`.
        Shape has the form (31, n_params), where 31 denotes the length of `alp`, a sequence
        of evenly spaced values on the interval (0, 1).

    q_2 : np.ndarray
        The brute force estimator of quantile based measure :math:`\tilde{q_i}^{(2)}(\alpha)`.
        Shape has the form (31, n_params).

    norm_q_1 : np.ndarray
        The brute force estimator of normalized quantile based measure :math:`Q_i^{(1)}(\alpha)`.
        Shape has the form (31, n_params).

    norm_q_2 : np.ndarray
        The brute force estimator of normalized quantile based measure :math:`Q_i^{(2)}(\alpha)`.
        Shape has the form (31, n_params).

    References
    ----------
    .. [K2019] S. Kucherenko, S. Song, L. Wang. Quantile based global
        sensitivity measures, Reliab. Eng. Syst. Saf. 185 (2019) 35–48.
    """
    # range of alpha
    dalp = (0.98 - 0.02) / 30
    alp = np.arange(0.02, 0.98 + dalp, dalp)  # len(alp) = 31

    # get the two independent groups of sample points
    x, x_prime = _unconditional_samples(
        n_draws,
        n_params,
        dist_type,
        loc,
        scale,
        sampling_scheme="sobol",
        seed=0,
        skip=0,
    )

    # get the conditional sample sets
    x_mix = _bf_conditional_samples(x, x_prime)

    # quantiles of output with unconditional input
    quantile_y_x = _unconditional_quantile_y(x, alp, func)

    # quantiles of output with conditional input
    quantile_y_x_mix = _conditional_quantile_y(x_mix, func, alp)

    # Get quantile based measures
    q_1, q_2 = _quantile_measures(quantile_y_x, quantile_y_x_mix)

    # Get normalized quantile based measures
    norm_q_1, norm_q_2 = _normalized_quantile_measures(q_1, q_2)

    return q_1, q_2, norm_q_1, norm_q_2


def dlr_mcs_quantile(
    func,
    n_params,
    loc,
    scale,
    dist_type,
    n_draws,
    sampling_scheme="sobol",
    seed=0,
    skip=0,
):
    r"""Compute (Quasi) Monte Carlo estimators of quantile based global sensitivity measures.

    This function implements the Double loop reordering
    (DLR) approach described in Section 4.2 of [K2019]_.

    Parameters
    ----------
    func : callable
        Objective function to calculate the quantile-based measures. Must be broadcastable.

    n_params : int
        Number of parameters of objective function.

    loc : float or np.ndarray
        The location(`loc`) keyword passed to `scipy.stats.norm`_ function to shift the
        location of "standardized" distribution. Specifically, for normal distribution
        it specifies the mean with the length of `n_params`.

        .. _scipy.stats.norm: https://docs.scipy.org/doc/scipy/reference/generated/
            _scipy.stats.norm.html

    scale : float or np.ndarray
        The `scale` keyword passed to `scipy.stats.norm`_ function to adjust the scale of
        "standardized" distribution. Specifically, for normal distribution it specifies
        the covariance matrix of shape (n_params, n_params).

    dist_type : str
        The distribution type of input. Options are "Normal", "Exponential" and "Uniform".

    n_draws : int
        Number of brute force Monte Carlo draws for the estimation of the sensitivity measures.
        Accroding to [K2017]_, to preserve the uniformity properties `n_draws` should always be
        equal to :math:`n_draws = 2^p`, where :math:`p` is an integer.

    sampling_scheme : str, optional
        One of ["random", "sobol"], default "sobol".

    seed : int, optional
        Random number generator seed.

    skip : int, optional
        Number of values to skip of Sobol sequence. Default is `0`.

    Returns
    -------
    q_1 : np.ndarray
        The DLR estimator of quantile based measure :math:`\tilde{q_i}^{(1)}(\alpha)`.
        Shape has the form (31, n_params), where 31 denotes the length of `alpha`, a
        sequence of evenly spaced values on the interval (0, 1).
    q_2 : np.ndarray
        The DLR estimator of quantile based measure :math:`\tilde{q_i}^{(2)}(\alpha)`.
        Shape has the form (31, n_params).

    norm_q_1 : np.ndarray
        The DLR estimator of normalized quantile based measure :math:`Q_i^{(1)}(\alpha)`.
        Shape has the form (31, n_params).

    norm_q_2 : np.ndarray
        The DLR estimator of normalized quantile based measure :math:`Q_i^{(2)}(\alpha)`.
        Shape has the form (31, n_params).


    References
    ----------
    .. [K2019] S. Kucherenko, S. Song, L. Wang. Quantile based global
        sensitivity measures, Reliab. Eng. Syst. Saf. 185 (2019) 35–48.

    .. [K2017] Kucherenko S, Song S. Different numerical estimators
        for main effect global sensitivity indices. Reliab Eng Syst
        Saf 2017;165:222–38.
    """
    # range of alpha
    dalp = (0.98 - 0.02) / 30
    alp = np.arange(0.02, 0.98 + dalp, dalp)  # len(alp) = 31

    # get the base draws from a joint PDF
    x = _unconditional_samples(
        n_draws,
        n_params,
        dist_type,
        loc,
        scale,
        sampling_scheme="sobol",
        seed=0,
        skip=0,
    )[0]

    # get the conditional draws
    x_mix = _dlr_conditional_samples(x)

    # quantile of output calculated with base draws
    quantile_y_x = _unconditional_quantile_y(x, alp, func)

    # quantile of output calculated with conditional draws
    quantile_y_x_mix = _conditional_quantile_y(x_mix, func, alp)

    # Get quantile based measures
    q_1, q_2 = _quantile_measures(quantile_y_x, quantile_y_x_mix)

    # Get normalized quantile based measures
    norm_q_1, norm_q_2 = _normalized_quantile_measures(q_1, q_2)

    return q_1, q_2, norm_q_1, norm_q_2


def _unconditional_samples(
    n_draws,
    n_params,
    dist_type,
    loc,
    scale,
    sampling_scheme="sobol",
    seed=0,
    skip=0,
):
    """Generate two independent groups of sample points.

    Parameters
    ----------
    n_draws : int
        Number of DLR Monte Carlo draws.
    n_params : int
        Number of parameters of objective function.
    dist_type : str
        The distribution type of input. Options are "Normal", "Exponential" and "Uniform".
    loc : float or np.ndarray
        The location(`loc`) keyword passed to `scipy.stats.norm`_ function to shift the
        location of "standardized" distribution.
    scale : float or np.ndarray
        The `scale` keyword passed to `scipy.stats.norm`_ function to adjust the scale of
        "standardized" distribution.
    sampling_scheme : str, optional
        One of ["sobol", "random"]
    seed : int, optional
        Random number generator seed. Default is 0.
    skip : int, optional
        Number of values to skip of Sobol sequence. Default is `0`.

    Returns
    -------
    x, x_prime : np.ndarray
        Two arrays of shape (n_draws, n_params) with i.i.d draws from a joint distribution.
    """
    # Generate uniform distributed samples
    np.random.seed(seed)
    if sampling_scheme == "sobol":
        u = cp.generate_samples(
            order=n_draws + skip,
            domain=n_params,
            rule="S",
        ).T
    elif sampling_scheme == "random":
        u = np.random.uniform(size=(n_draws, n_params))
    else:
        raise ValueError("Argument 'sampling_scheme' is not in {'sobol', 'random'}.")

    skip = skip if sampling_scheme == "sobol" else 0

    u = cp.generate_samples(order=n_draws, domain=2 * n_params, rule="S").T
    u_1 = u[skip:, :n_params]
    u_2 = u[skip:, n_params:]

    # Transform uniform draws into a joint PDF
    if dist_type == "Normal":
        z = norm.ppf(u_1)
        z_prime = norm.ppf(u_2)
        cholesky = np.linalg.cholesky(scale)
        x = loc + cholesky.dot(z.T).T
        x_prime = loc + cholesky.dot(z_prime.T).T
    elif dist_type == "Exponential":
        x = expon.ppf(u_1, loc, scale)
        x_prime = expon.ppf(u_2, loc, scale)
    elif dist_type == "Uniform":
        x = uniform.ppf(u_1, loc, scale)
        x_prime = uniform.ppf(u_2, loc, scale)
    else:
        raise NotImplementedError

    return x, x_prime


def _bf_conditional_samples(x, x_prime):
    """Generate mixed sample sets distributed accroding to a conditional PDF.

    Parameters
    ----------
    x, x_prime : np.ndarray
        Two arrays of shape (n_draws, n_params) with i.i.d draws from a joint distribution.

    Returns
    -------
    x_mix :  np.ndarray
        Mixed sample sets. Shape has the form (n_draws, n_params, n_draws, n_params).

    """
    n_draws, n_params = x.shape
    x_mix = np.zeros((n_draws, n_params, n_draws, n_params))

    for i in range(n_params):
        for j in range(n_draws):
            x_mix[j, i] = x
            x_mix[j, i, :, i] = x_prime[j, i]

    return x_mix


def _dlr_conditional_samples(x):
    """Generate conditional sample sets from the base draws.

    Parameters
    ----------
    x : np.ndarray
        Draws from a joint distribution. Shape has the form (n_draws, n_params).

    Returns
    -------
    x_mix :  np.ndarray
        Mixed sample sets. Shape has the form (m, n_params, n_draws, n_params),  where m
        is the number of conditional bins.
    """
    n_draws, n_params = x.shape

    # The dependence of m versus n_draws accroding to [K2017] fig.1
    if n_draws == 2 ** 6:
        m = 2 ** 3
    elif n_draws <= 2 ** 9:
        m = 2 ** 4
    elif n_draws == 2 ** 10:
        m = 2 ** 5
    elif n_draws <= 2 ** 13:
        m = 2 ** 6
    elif n_draws <= 2 ** 15:
        m = 2 ** 7
    else:
        raise NotImplementedError

    conditional_bin = x[:m]
    x_mix = np.zeros((m, n_params, n_draws, n_params))

    # subdivide unconditional samples into M eaually bins, within each bin x_i being fixed.
    for i in range(n_params):
        for j in range(m):
            x_mix[j, i] = x
            x_mix[j, i, :, i] = conditional_bin[j, i]

    return x_mix


def _unconditional_quantile_y(x, alp, func):
    """Return quantiles of outputs with unconditional draws as input.

    Parameters
    ----------
    x : np.ndarray
        Draws from a joint distribution. Shape has the form (n_draws, n_params).
    alp : np.ndarray
        A sequence of evenly spaced values on the interval (0, 1).
    func : callable
        Objective function to calculate the quantile-based measures. Must be broadcastable.

    Returns
    -------
    quantile_y_x :  np.ndarray
        Quantiles of outputs corresponding to alpha for unconditional inputs.
        Shape has the form (len(alp),).

    """
    n_draws = x.shape[0]

    # Equation 21a
    y_x = func(x)
    y_x_asc = np.sort(y_x)
    q_index = (np.floor(alp * n_draws)).astype(int)
    quantile_y_x = y_x_asc[q_index]

    return quantile_y_x


def _conditional_quantile_y(x_mix, func, alp):
    """Return quantiles of outputs with conditional draws as input.

    Parameters
    ----------
    x_mix : np.ndarray
        Mixed draws. Shape has the form (m, n_params, n_draws, n_params).
    func : callable
        Objective function to calculate the quantile-based measures. Must be broadcastable.
    alp : np.ndarray
        A sequence of evenly spaced values on the interval (0, 1).

    Returns
    -------
    quantile_y_x_mix  :  np.ndarray
        Quantiles of output corresponding to alpha for conditional inputs. Shape has the form
        (m, n_params, len(alp), 1), where m is the number of conditional bins.
    """
    m, n_params, n_draws = x_mix.shape[:3]

    y_x_mix = np.zeros((m, n_params, n_draws, 1))
    y_x_mix_asc = np.zeros((m, n_params, n_draws, 1))
    quantile_y_x_mix = np.zeros((m, n_params, len(alp), 1))

    # Equation 21b/26. Get quantiles within each bin.
    for i in range(n_params):
        for j in range(m):
            # values of conditional outputs
            y_x_mix[j, i] = np.vstack(func(x_mix[j, i]))
            y_x_mix_asc[j, i] = np.sort(y_x_mix[j, i], axis=0)
            for pp, a in enumerate(alp):
                quantile_y_x_mix[j, i, pp] = y_x_mix_asc[j, i][
                    (np.floor(a * n_draws)).astype(int)
                ]  # quantiles corresponding to alpha
    return quantile_y_x_mix


def _quantile_measures(quantile_y_x, quantile_y_x_mix):
    """Estimate the values of quantile based measures."""
    m, n_params, len_alp = quantile_y_x_mix.shape[:3]

    # initialization
    q_1 = np.zeros((len_alp, n_params))
    q_2 = np.zeros((len_alp, n_params))
    delt = np.zeros((m, n_params, len_alp, 1))

    # Equation 24 & 25 / 27 & 28
    for j in range(m):
        for i in range(n_params):
            for pp in range(len_alp):
                delt[j, i, pp] = quantile_y_x_mix[j, i, pp] - quantile_y_x[pp]
                q_1[pp, i] = np.mean(np.absolute(delt[:, i, pp]))
                q_2[pp, i] = np.mean(delt[:, i, pp] ** 2)

    return q_1, q_2


def _normalized_quantile_measures(q_1, q_2):
    """Estimate the values of normalized quantile based measures."""
    len_alp, n_params = q_1.shape

    # initialization
    sum_q_1 = np.zeros(len_alp)
    sum_q_2 = np.zeros(len_alp)
    norm_q_1 = np.zeros((len_alp, n_params))
    norm_q_2 = np.zeros((len_alp, n_params))

    # Equation 13 & 14
    for pp in range(len_alp):
        sum_q_1[pp] = np.sum(q_1[pp, :])
        sum_q_2[pp] = np.sum(q_2[pp, :])
        for i in range(n_params):
            norm_q_1[pp, i] = q_1[pp, i] / sum_q_1[pp]
            norm_q_2[pp, i] = q_2[pp, i] / sum_q_2[pp]

    return norm_q_1, norm_q_2
