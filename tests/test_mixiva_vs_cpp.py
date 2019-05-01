import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile

import piva

# We use several sound samples for each source to have a long enough length
wav_files = [
    [
        "examples/input_samples/cmu_arctic_us_axb_a0004.wav",
        "examples/input_samples/cmu_arctic_us_axb_a0005.wav",
        "examples/input_samples/cmu_arctic_us_axb_a0006.wav",
    ],
    [
        "examples/input_samples/cmu_arctic_us_aew_a0001.wav",
        "examples/input_samples/cmu_arctic_us_aew_a0002.wav",
        "examples/input_samples/cmu_arctic_us_aew_a0003.wav",
    ],
]


def create_mix(filename=None):

    # Room 4m by 6m
    room_dim = [8, 9]

    # source location
    source = np.array([1, 4.5])

    # create an anechoic room with sources and mics
    room = pra.ShoeBox(room_dim, fs=16000, max_order=0)

    # get signals
    signals = [
        np.concatenate([wavfile.read(f)[1].astype(np.float32) for f in source_files])
        for source_files in wav_files
    ]
    delays = [1.0, 0.0]
    locations = [[2.5, 3], [2.5, 6]]

    # add mic and good source to room
    # Add silent signals to all sources
    for sig, d, loc in zip(signals, delays, locations):
        room.add_source(loc, signal=sig, delay=d)

    # add microphone array
    room.add_microphone_array(
        pra.MicrophoneArray(np.c_[[6.5, 4.47], [6.5, 4.53]], fs=room.fs)
    )

    # compute RIRs
    room.compute_rir()

    def callback_mix(premix):

        sigma_s = np.std(premix[:, premix.shape[1] // 2, :], axis=1)
        premix /= sigma_s[:, None, None]

        mix = np.sum(premix, axis=0)

        scale = np.maximum(np.max(np.abs(premix)), np.max(np.abs(mix)))

        mix *= 0.95 / scale
        premix *= 0.95 / scale

        return mix

    # Run the simulation
    separate_recordings = room.simulate(callback_mix=callback_mix, return_premix=True)
    mics_signals = room.mic_array.signals

    if filename is not None:
        wavfile.write(filename, room.fs, (mics_signals.T * (2 ** 15)).astype(np.int16))

    return mics_signals, separate_recordings, len(locations)


if __name__ == "__main__":

    mix, references, n_src = create_mix()

    # STFT parameters
    framesize = 512
    hop = framesize // 2
    win_a = pra.hann(framesize)
    win_s = pra.transform.compute_synthesis_window(win_a, framesize // 2)

    # shape == (n_chan, n_frames, n_freq)
    X = pra.transform.analysis(mix.T, framesize, hop, win=win_a)
    n_frames, n_freq, n_chan = X.shape

    n_iter = 10
    n_iter_inner = 1

    y_diff = []
    y_error_py = []
    y_error_cpp = []

    Y_py = X.copy()
    Y_cpp = X.copy()

    for n in range(n_iter):
        # Run OverIVA (python)
        Y_py = piva.auxiva(Y_py.copy(), n_src=n_src, n_iter=n_iter_inner, proj_back=False)

        z = pra.bss.projection_back(Y_py, X[:, :, 0])
        y_py = pra.transform.synthesis(Y_py * np.conj(z), framesize, hop, win=win_s).T

        y_py = y_py[:, framesize - hop :]
        m = np.minimum(y_py.shape[1], references.shape[2])
        e_py = np.minimum(
            np.mean((references[:, 0, :m] - y_py[:, :m]) ** 2),
            np.mean((references[::-1, 0, :m] - y_py[:, :m]) ** 2),
        )
        y_error_py.append(e_py / np.mean(np.abs(references[:, 0, :m]) ** 2))

        # Run OverIVA (cpp)
        Y_cpp = piva.auxiva_cpp(
            Y_cpp.copy(), n_src=n_src, n_iter=n_iter_inner, proj_back=False
        )

        z = pra.bss.projection_back(Y_cpp, X[:, :, 0])
        y_cpp = pra.transform.synthesis(Y_cpp * np.conj(z), framesize, hop, win=win_s).T

        y_cpp = y_cpp[:, framesize - hop :]
        m = np.minimum(y_cpp.shape[1], references.shape[2])
        e_cpp = np.minimum(
            np.mean((references[:, 0, :m] - y_cpp[:, :m]) ** 2),
            np.mean((references[::-1, 0, :m] - y_cpp[:, :m]) ** 2),
        )
        y_error_cpp.append(e_cpp / np.var(references[:, 0, :m]))

        y_diff.append(np.mean((y_py - y_cpp) ** 2))

    plt.semilogy(y_diff, label="Difference between py and cpp")
    plt.semilogy(y_error_py, label="Separation error Python")
    plt.semilogy(y_error_cpp, label="Separation error C++")
    plt.legend()
    plt.show()
