# Copyright (c) 2019 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Decorrelation via Principal Component Analysis
==============================================

This is a simple implemenation of decorrelation via PCA.
"""
import numpy as np
from scipy import linalg

from .utils import tensor_H
from .core import pca as pca_core

try:
    import mkl

    has_mkl = True
except ImportError:
    has_mkl = False


def pca(X, backend="cpp"):
    """
    Whiten the input signal

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the signal

    Returns
    -------
    ndarray (nframes, nfrequencies, nchannels)
        The decorrelated input with the channels orderer from least dominant
        to most dominant
    ndarray (nfrequencies, nchannels, nchannels)
        The decorrelation matrices for all frequencies
    """

    if backend == "cpp":
        return pca_cpp(X)
    elif backend == "py":
        return pca_py(X)
    else:
        raise ValueError("Only backends " "py" " and " "cpp" " are supported")


def pca_py(X):
    """
    Python implementation of signal decorrelation
    """
    n_frames, n_freq, n_chan = X.shape

    # compute the cov mat (n_freq, n_chan, n_chan)
    X_T = X.transpose([1, 2, 0])
    covmat = (X_T @ tensor_H(X_T)) * (1.0 / n_frames)

    # Compute EVD
    # v.shape == (n_freq, n_chan), w.shape == (n_freq, n_chan, n_chan)
    eig_val, eig_vec = np.linalg.eigh(covmat)

    # The decorrelation matrices
    Q = (1.0 / np.sqrt(eig_val[:, :, None])) * tensor_H(eig_vec)

    return (Q @ X_T).transpose([2, 0, 1]).copy(), Q


def pca_cpp(X):
    """
    Wrapper for cpp implementation of signal decorrelation
    """
    assert (
        X.dtype == np.complex128
    ), "cpp backend only supports double precision complex arrays"

    n_frames, n_freq, n_chan = X.shape

    if has_mkl:
        # We need to deactivate parallelization in mkl
        mkl_num_threads = mkl.get_max_threads()
        mkl.set_num_threads(1)

    # Make a copy of the input with efficient axis order
    X_T = X.transpose([1, 2, 0]).copy()

    # Create arrays to receive the output
    W = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)

    pca_core(X_T, W)

    if has_mkl:
        mkl.set_num_threads(mkl_num_threads)

    return X_T.transpose([2, 0, 1]), W
