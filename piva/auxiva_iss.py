#  Auxiliary function based independent vector analysis
#  with iterative source steering algorithm.
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
Blind Source Separation using Independent Vector Analysis with Auxiliary Function
"""
import numpy as np

from .projection_back import project_back
from . import defaults
from .core import auxiva_iss_gauss_core, auxiva_iss_laplace_core

try:
    import mkl

    has_mkl = True
except ImportError:
    has_mkl = False


def auxiva_iss(
    X, backend="cpp", **kwargs,
):
    if backend == "cpp":
        return auxiva_iss_cpp(X, **kwargs)
    elif backend == "py":
        return auxiva_iss_py(X, **kwargs)
    else:
        raise ValueError("Only backends " "py" " and " "cpp" " are supported")


def auxiva_iss_cpp(
    X,
    n_src=None,
    n_iter=20,
    proj_back=True,
    model=defaults.model,
    return_filters=False,
    callback=None,
):
    """
    Wrapper for the C++ implementation of MixIVA

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
    callback: func
        A callback function called every 10 iterations, allows to monitor
        convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """
    n_frames, n_freq, n_chan = X.shape

    X_T_original = X.transpose([1, 2, 0])
    X_T = X_T_original.copy()

    if has_mkl:
        # We need to deactivate parallelization in mkl
        mkl_num_threads = mkl.get_max_threads()
        mkl.set_num_threads(1)

    if model == "laplace":
        auxiva_iss_laplace_core(X_T, n_iter)
    elif model == "gauss":
        auxiva_iss_gauss_core(X_T, n_iter)
    else:
        raise ValueError(f"No such model {model}")

    if has_mkl:
        mkl.set_num_threads(mkl_num_threads)

    if return_filters is not None:
        # Demixing matrices were not computed explicitely so far,
        # do it here, if necessary
        W = X_T[:, :, :n_chan] @ np.linalg.inv(X_T_original[:, :, :n_chan])

    Y = X_T.transpose([2, 0, 1]).copy()

    if proj_back:
        Y = project_back(Y, X[:, :, 0])

    if return_filters:
        return Y, W
    else:
        return Y


def auxiva_iss_py(
    X,
    n_src=None,
    n_iter=20,
    proj_back=True,
    model=defaults.model,
    return_filters=False,
    callback=None,
):

    """
    Blind source separation based on independent vector analysis with
    alternating updates of the mixing vectors

    Robin Scheibler, Nobutaka Ono, Unpublished, 2019

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
    callback: func
        A callback function called every 10 iterations, allows to monitor convergence

    Returns
    -------
    Returns an (nframes, nfrequencies, nsources) array. Also returns
    the demixing matrix (nfrequencies, nchannels, nsources)
    if ``return_values`` keyword is True.
    """

    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = X.shape[2]

    # for now, only supports determined case
    assert n_chan == n_src

    # pre-allocate arrays
    r_inv = np.zeros((n_src, n_frames))
    v_num = np.zeros((n_freq, n_src), dtype=X.dtype)
    v_denom = np.zeros((n_freq, n_src), dtype=np.float64)
    v = np.zeros((n_freq, n_src), dtype=X.dtype)

    # Things are more efficient when the frequencies are over the first axis
    X = X.transpose([1, 2, 0]).copy()

    # Initialize the demixed outputs
    Y = X.copy()

    for epoch in range(n_iter):

        if callback is not None:
            Y_tmp = Y.transpose([2, 0, 1])
            callback(Y_tmp, epoch)

        # shape: (n_src, n_frames)
        # OP: n_frames * n_src
        eps = 1e-10
        if model == "laplace":
            r_inv[:, :] = 1.0 / np.maximum(eps, 2.0 * np.linalg.norm(Y, axis=0))
        elif model == "gauss":
            r_inv[:, :] = 1.0 / np.maximum(
                eps, (np.linalg.norm(Y, axis=0) ** 2) / n_freq
            )

        # Update now the demixing matrix
        for s in range(n_src):

            # OP: n_frames * n_src
            v_num = (Y * r_inv[None, :, :]) @ np.conj(
                Y[:, s, :, None]
            )  # (n_freq, n_src, 1)
            # OP: n_frames * n_src
            v_denom = r_inv[None, :, :] @ np.abs(Y[:, s, :, None]) ** 2
            # (n_freq, n_src, 1)

            # OP: n_src
            v[:, :] = v_num[:, :, 0] / v_denom[:, :, 0]
            # OP: 1
            v[:, s] -= 1 / np.sqrt(v_denom[:, s, 0])

            # update demixed signals
            # OP: n_frames * n_src
            Y[:, :, :] -= v[:, :, None] * Y[:, s, None, :]

    if return_filters is not None:
        # Demixing matrices were not computed explicitely so far,
        # do it here, if necessary
        W = Y[:, :, :n_chan] @ np.linalg.inv(X[:, :, :n_chan])

    Y = Y.transpose([2, 0, 1]).copy()
    X = X.transpose([2, 0, 1])

    if proj_back:
        Y = project_back(Y, X[:, :, 0])

    if return_filters:
        return Y, W
    else:
        return Y
