"""
Test the whiten python and matlab implementations
"""
import time
import numpy as np
from piva import crandn, tensor_H, pca


def covmat(X):
    X_T = X.transpose([1, 2, 0])
    return X_T @ tensor_H(X_T) / X.shape[0]


def test_err_time(backend, rep=100):

    # Python
    t1 = time.perf_counter()
    for i in range(rep):
        Y, W = pca(X, backend=backend)
    t2 = time.perf_counter()

    eps = np.max(np.abs(covmat(Y) - np.eye(n_chan)[None, :, :]))

    return eps, (t2 - t1) / rep


if __name__ == "__main__":

    # Dimensions
    n_frames, n_freq, n_chan = (20, 100, 10)

    # Create fake data
    S = crandn(n_frames, n_freq, n_chan)
    W0 = crandn(n_freq, n_chan, n_chan)
    X = (W0 @ S.transpose([1, 2, 0])).transpose([2, 0, 1])

    # Test both backends
    for backend in ["py", "cpp"]:
        eps, runtime = test_err_time(backend)
        print(f"backend={backend} err={eps} time={runtime:.6f}")
