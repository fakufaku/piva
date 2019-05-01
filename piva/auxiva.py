#  Auxiliary function based independent vector analysis
#  with iterative projection 2 algorithm.
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

import numpy as np
from .projection_back import project_back
from .utils import tensor_H
from .core import (
    auxiva_gauss_core,
    auxiva_laplace_core,
    overiva_gauss_core,
    overiva_laplace_core,
)
from . import defaults

try:
    import mkl

    has_mkl = True
except ImportError:
    has_mkl = False


def auxiva(X, **kwargs):

    _, __, n_chan = X.shape

    # for auxiva, we ignore the number of sources
    if "n_src" in kwargs:
        kwargs.pop("n_src")

    return overiva(X, **kwargs)


def overiva(
    X, backend="cpp", **kwargs,
):
    if backend == "cpp":
        return overiva_cpp(X, **kwargs)
    elif backend == "py":
        return overiva_py(X, **kwargs)
    else:
        raise ValueError("Only backends " "py" " and " "cpp" " are supported")


def overiva_py(
    X,
    n_src=None,
    n_iter=20,
    proj_back=True,
    model=defaults.model,
    return_filters=False,
    callback=None,
):
    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = n_chan

    # Things are more efficient when the frequencies are over the first axis
    Y = np.zeros((n_freq, n_src, n_frames), dtype=X.dtype)
    X_original = X
    X = X.transpose([1, 2, 0]).copy()

    # We keep a copy of the identity matrix around
    eyes = np.eye(n_chan)[None, :, :]

    # Init. demixing matrix
    W = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    W[:, :, :] = eyes

    # prepare the overdetermined case
    if n_src < n_chan:
        # covariance of input signal
        Cx = (X @ tensor_H(X)) / n_frames

        # view to background part of demixing matrix
        J = W[:, n_src:, :n_src]
        W[:, n_src:, n_src:] *= -1

    r_inv = np.zeros((n_src, n_frames))

    # because we initialize demixing matrix with identity
    # the first output is a copy of the input signal
    Y[:, :, :] = X[:, :n_src, :]

    for epoch in range(n_iter):

        if callback is not None:
            Y_tmp = Y.transpose([2, 0, 1])
            callback(Y_tmp, epoch)

        # simple loop as a start
        # shape: (n_frames, n_src)
        eps = 1e-10
        if model == "laplace":
            r_inv[:, :] = 1.0 / np.maximum(eps, 2.0 * np.linalg.norm(Y, axis=0))
        elif model == "gauss":
            r_inv[:, :] = 1.0 / np.maximum(
                eps, (np.linalg.norm(Y, axis=0) ** 2) / n_freq
            )

        # Update now the demixing matrix
        for s in range(n_src):

            # Update the mixing matrix according to orthogonal constraints
            if n_src < n_chan:
                tmp = W[:, :n_src, :] @ Cx
                J[:, :, :] = tensor_H(
                    np.linalg.solve(tmp[:, :, :n_src], tmp[:, :, n_src:])
                )

            # Compute Auxiliary Variable
            # shape: (n_freq, n_chan, n_chan)
            V = (X * r_inv[None, s, None, :]) @ tensor_H(X) / n_frames

            WV = W @ V
            W[:, s, :] = np.conj(np.linalg.solve(WV, eyes[:, :, s]))

            # normalize
            denom = np.real(W[:, None, s, :] @ V[:, :, :] @ np.conj(W[:, s, :, None]))
            W[:, s, :] /= np.sqrt(denom[:, :, 0])

        # demixing
        Y[:, :, :] = W[:, :n_src, :] @ X

    # reorder
    Y = Y.transpose([2, 0, 1]).copy()

    if proj_back:
        Y = project_back(Y, X_original[:, :, 0])

    if return_filters:
        return Y, W
    else:
        return Y


def overiva_cpp(
    X,
    n_src=None,
    n_iter=20,
    proj_back=True,
    model=defaults.model,
    return_filters=False,
    **kwargs,
):
    """
    Wrapper for the C++ implementation of AuxIVA

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal
    n_src: int, optional
        The number of sources or independent components
    n_iter: int, optional
        The number of iterations (default 20)
    proj_back: bool, optional
        Scaling on first mic by back projection (default True)
    model: str
        The model of source distribution 'gauss' or 'laplace' (default)
    return_filters: bool
        If true, the function will return the demixing matrix too

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nchannels)
    if ``return_values`` keyword is True.
    """

    if model not in ["laplace", "gauss"]:
        raise ValueError(f"No such model {model}")

    n_frames, n_freq, n_chan = X.shape

    if n_src is None:
        n_src = n_chan

    # new shape: (nfrequencies, nchannels, nframes)
    X_T = X.transpose([1, 2, 0]).copy()

    if has_mkl:
        # We need to deactivate parallelization in mkl
        mkl_num_threads = mkl.get_max_threads()
        mkl.set_num_threads(1)

    # Initialize the demixing matrix
    W = np.zeros((n_freq, n_chan, n_chan), dtype=np.complex128)
    W[:, :, :] = np.eye(n_chan)[None, :, :]

    if n_src == n_chan:
        Y_T = X_T.copy()

        if model == "laplace":
            auxiva_laplace_core(X_T, Y_T, W, n_iter)
        elif model == "gauss":
            auxiva_gauss_core(X_T, Y_T, W, n_iter)

    else:
        Y_T = X_T[:, :n_src, :].copy()
        W_loc = W[:, :n_src, :].copy()

        if model == "laplace":
            overiva_laplace_core(X_T, Y_T, W_loc, n_iter)
        elif model == "gauss":
            overiva_gauss_core(X_T, Y_T, W_loc, n_iter)

        if return_filters:
            # copy demixing matrix to return to user
            W[:, :n_src, :] = W_loc
            W[:, n_src:, n_src:] *= -1

            # covariance of input signal
            Cx = (X_T @ tensor_H(X_T)) / n_frames

            # build missing part of demixing matrix
            tmp = W[:, :n_src, :] @ Cx
            W[:, n_src:, :n_src] = tensor_H(
                np.linalg.solve(tmp[:, :, :n_src], tmp[:, :, n_src:])
            )

    if has_mkl:
        mkl.set_num_threads(mkl_num_threads)

    Y = Y_T.transpose([2, 0, 1]).copy()

    if proj_back:
        Y = project_back(Y, X[:, :, 0])

    if return_filters:
        return Y, W
    else:
        return Y
