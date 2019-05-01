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
A few useful routines
=====================
"""
import numpy as np


# The threshold for small values in activations
_eps = 1e-15


def tensor_H(X):
    return np.conj(X).swapaxes(-1, -2)


def crandn(*shape, **kwargs):
    return np.random.randn(*shape, **kwargs) + 1j * np.random.randn(*shape, **kwargs)


def demix(Y, X, W):
    Y[:, :, :] = W @ X


def compute_activations(Y, subset, model):

    # shape: (n_frames, n_src)
    if model == "laplace":
        r = 2.0 * np.linalg.norm(Y[:, subset, :], axis=0)
    elif model == "gauss":
        r = (np.linalg.norm(Y[:, subset, :], axis=0) ** 2) / Y.shape[0]

    return 1.0 / np.maximum(_eps, r)


def compute_weighted_covmat(X, weights):
    return (X * weights[:, None, :]) @ tensor_H(X) / X.shape[2]


def compute_weighted_xcovmat(X1, X2, weights):
    return (X1 * weights[:, None, :]) @ tensor_H(X2) / X1.shape[2]


def iva_cost(Y, W, model):

    if model == "laplace":
        r = np.sum(np.linalg.norm(Y, axis=0))
    elif model == "gauss":
        r = np.sum(np.log(np.linalg.norm(Y, axis=0)))

    logdet = np.sum(np.linalg.slogdet(W)[1])

    return r - 2 * Y.shape[2] * logdet


class TwoStepsIterator(object):
    """
    Iterates two elements at a time between 0 and m - 1
    """

    def __init__(self, m):
        self.m = m

    def _inc(self):
        self.next = (self.next + 1) % self.m
        self.count += 1

    def __iter__(self):
        self.count = 0
        self.next = 0
        return self

    def __next__(self):

        if self.count < 2 * self.m:

            m = self.next
            self._inc()
            n = self.next
            self._inc()

            return m, n

        else:
            raise StopIteration
