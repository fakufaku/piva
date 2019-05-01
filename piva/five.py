#  Fast Independent Vector Extraction
#  Copyright (C) 2020  Robin Scheibler
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
FIVE: Fast Independent Vector Extraction
========================================

This algorithm extracts one source independent from a minimum energy background
using an iterative algorithm that attempts to maximize the SINR at every step.

References
----------

.. [1] R. Scheibler and N. Ono, Fast Independent Vector Extraction by Iterative
    SINR Maximization, arXiv, 2019.
"""
import numpy as np
from scipy import linalg

from .projection_back import project_back
from .core import five_laplace_core, five_gauss_core
from . import defaults

try:
    import mkl

    has_mkl = True
except ImportError:
    has_mkl = False


def demix(X, W):
    """ Apply demixing matrices W to X """
    return (W @ X.transpose([1, 2, 0])).transpose([2, 0, 1])


def five(X, backend="cpp", **kwargs):

    if backend == "cpp":
        return five_cpp(X, **kwargs)
    elif backend == "py":
        return five_py(X, **kwargs)
    else:
        raise ValueError("Only backends " "py" " and " "cpp" " are supported")


def five_cpp(
    X,
    n_iter=3,
    proj_back=True,
    W0=None,
    model=defaults.model,
    return_filters=False,
    callback=None,
    callback_checkpoints=[],
    cost_callback=None,
    **kwargs,
):
    assert (
        X.dtype == np.complex128
    ), "FIVE only supports complex double precision arrays"

    n_frames, n_freq, n_chan = X.shape

    if has_mkl:
        # We need to deactivate parallelization in mkl
        mkl_num_threads = mkl.get_max_threads()
        mkl.set_num_threads(1)

    # Make a copy of the input with efficient axis order
    X_T = X.transpose([1, 2, 0]).copy()

    # Create arrays to receive the output
    W = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    Y = np.zeros((n_freq, 1, n_frames), dtype=X.dtype)

    if W0 is not None:
        X_T = W0 @ X_T

    if model == "laplace":
        five_laplace_core(X_T, Y, W, n_iter)
    elif model == "gauss":
        five_gauss_core(X_T, Y, W, n_iter)
    else:
        raise ValueError(f"No such model {model}")

    if has_mkl:
        mkl.set_num_threads(mkl_num_threads)

    Y = Y.transpose([2, 0, 1]).copy()

    if proj_back:
        Y = project_back(Y, X[:, :, 0])

    if return_filters:
        return Y, W
    else:
        return Y


def five_py(
    X,
    n_iter=3,
    proj_back=True,
    W0=None,
    model=defaults.model,
    return_filters=False,
    callback=None,
    **kwargs,
):

    """
    This algorithm extracts one source independent from a minimum energy background.
    The separation is done in the time-frequency domain and the FFT length
    should be approximately equal to the reverberation time. The residual
    energy in the background is minimized.

    Two different statistical models (Laplace or time-varying Gauss) can
    be used by specifying the keyword argument `model`. The performance of Gauss
    model is higher in good conditions (few sources, low noise), but Laplace
    (the default) is more robust in general.

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_iter: int, optional
        The number of iterations (default 3)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    W0: ndarray (nfrequencies, nsrc, nchannels), optional
        Initial value for demixing matrix
    model: str
        The model of source distribution 'gauss' or 'laplace' (default)
    return_filters: bool
        If true, the function will return the demixing matrix too
    callback: func
        A callback function called every 10 iterations, allows to monitor
        convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, 1) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """

    n_frames, n_freq, n_chan = X.shape

    def tensor_H(A):
        return np.conj(A.swapaxes(1, 2))

    def cost(Y, X, W):
        _, logdet = np.linalg.slogdet(W)
        target = np.sum(np.linalg.norm(Y[:, 0, :], axis=0)) / n_frames
        background = np.sum(np.linalg.norm(W[:, 1:, :] @ X, axis=1) ** 2) / n_frames

        return -2 * np.sum(logdet) + target + background

    # default to determined case
    n_src = 1

    if model not in ["laplace", "gauss"]:
        raise ValueError("Model should be either " "laplace" " or " "gauss" ".")

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = np.mean(X[:, :, :, None] * np.conj(X[:, :, None, :]), axis=0)

    # We will need the inverse square root of Cx
    e_val, e_vec = np.linalg.eigh(Cx)
    Q_H_inv = tensor_H(e_vec) * (1.0 / np.sqrt(e_val[:, :, None]))

    W_hat = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    W = W_hat[:, :n_src, :]

    # initialize A and W
    if W0 is None:
        # We use the principal component from PCA as initialization
        W_hat[:, 0, -1] = 1.0
    else:
        W_hat[:, :, :] = W0

    eps = 1e-10
    V = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    r_inv = np.zeros((n_src, n_frames))

    # Things are more efficient when the frequencies are over the first axis
    Y = np.zeros((n_freq, n_src, n_frames), dtype=X.dtype)
    X_original = X.copy()

    # decorrelate the input signal
    X = Q_H_inv @ X.transpose([1, 2, 0])

    # Compute the demixed output
    def demix(Y, X, W):
        Y[:, :, :] = np.matmul(W, X)

    for epoch in range(n_iter):

        demix(Y, X, W)

        if callback is not None:
            Y_tmp = Y.transpose([2, 0, 1])
            callback(Y_tmp, epoch)

        # Update now the demixing matrix

        # shape: (n_frames, n_src)
        if model == "laplace":
            r_inv = 1.0 / np.maximum(eps, 2.0 * np.linalg.norm(Y[:, 0, :], axis=0))
        elif model == "gauss":
            r_inv = 1.0 / np.maximum(
                eps, (np.linalg.norm(Y[:, 0, :], axis=0) ** 2) / n_freq
            )

        # Compute Auxiliary Variable
        # shape: (n_freq, n_chan, n_chan)
        V[:, :, :] = np.matmul(
            (X * r_inv[None, None, :]), np.conj(X.swapaxes(1, 2)) / n_frames
        )

        # Solve the Eigenvalue problem
        # We only need the smallest eigenvector and eigenvalue,
        # so we could solve this more efficiently, but it is faster to
        # just solve everything rather than wrap this in a for loop
        lambda_, R = np.linalg.eigh(V)
        # Eigenvalues are in ascending order
        W_hat[:, 0, :] = np.conj(R[:, :, 0]) / np.sqrt(lambda_[:, 0, None])

        # Save the whole demxing matrix in the last iteration
        if epoch == n_iter - 1:
            W_hat[:, 1:, :] = tensor_H(R[:, :, 1:])

    demix(Y, X, W)

    Y = Y.transpose([2, 0, 1]).copy()

    if proj_back:
        Y = project_back(Y, X_original[:, :, 0])

    if return_filters:
        return Y, W_hat @ Q_H_inv
    else:
        return Y
