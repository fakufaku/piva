import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra

import piva

if __name__ == "__main__":

    # No padding needed in this test

    fs = 16000
    T = 1  # + 30 / 16000
    dt = 1 / fs
    t = np.arange(0, T, dt)
    f = 666.0  # Hertz
    s = np.sin(2.0 * np.pi * t * f)

    n_win = 512
    n_zeropad_back = 128
    n_zeropad_front = 512 - 128
    win = np.ones(n_win)
    shift = n_win // 4  # half-overlap
    win_s = pra.transform.compute_synthesis_window(win, shift)

    X = piva.stft(s[:, None], win, shift, n_zeropad_front, n_zeropad_back)

    s2 = piva.istft(X, win_s, shift, n_zeropad_front, n_zeropad_back)[:, 0]

    plt.figure()
    plt.imshow(
        np.abs(X[:, 0, :]),
        aspect="auto",
        extent=[0, T, 0, fs / 2],
        origin="lower",
        interpolation="none",
    )

    m_len = np.minimum(len(s), len(s2))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t[:m_len], s[:m_len], label="Original")
    ax1.plot(t[:m_len], s2[:m_len], label="Reconstructed")
    ax1.legend()
    ax2.plot(t[:m_len], s[:m_len] - s2[:m_len], label="Error")
    ax2.legend()
    plt.show()
