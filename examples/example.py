"""
Example of using the piva library
=================================
"""

import sys
import time

import matplotlib
import numpy as np
import pyroomacoustics as pra

from utils import make_room, PlaySoundGUI, samples

import piva


# We concatenate a few samples to make them long enough
if __name__ == "__main__":

    algo_choices = list(piva.algorithms.keys())
    model_choices = piva.models

    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstration of blind source extraction using FIVE."
    )
    parser.add_argument("-m", "--mics", type=int, default=5, help="Number of mics")
    parser.add_argument("-s", "--srcs", type=int, default=2, help="Number of sources")
    parser.add_argument(
        "-a",
        "--algo",
        type=str,
        default=algo_choices[0],
        choices=algo_choices,
        help="Chooses BSS method to run",
    )
    parser.add_argument(
        "-n", "--n_iter", type=int, default=51, help="Number of iterations"
    )
    parser.add_argument(
        "-d",
        "--dist",
        type=str,
        default=model_choices[0],
        choices=model_choices,
        help="IVA model distribution",
    )
    parser.add_argument(
        "--init_pca",
        action="store_true",
        help="Initialization by PCA",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Creates a small GUI for easy playback of the sound samples",
    )
    parser.add_argument("--snr", type=float, default=15, help="Signal to noise ratio")
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed to get different results"
    )
    args = parser.parse_args()

    assert args.srcs <= args.mics, "More sources than microphones is not supported"
    assert args.srcs <= len(
        samples
    ), f"This example only supports up to {len(samples)} sources"

    """
    if args.algo == "five":
        n_sources = 1
        print("Using single source for algorithm 'five'")
    else:
        n_sources = args.srcs
    """
    n_sources = args.srcs

    if args.gui:
        print("setting tkagg backend")
        # avoids a bug with tkinter and matplotlib
        import matplotlib

        matplotlib.use("TkAgg")

    # Simulation parameters
    room_dim = [10.0, 7.5, 3.2]
    fs = 16000
    absorption, max_order = 0.35, 17  # RT60 == 0.3
    # absorption, max_order = 0.45, 12  # RT60 == 0.2
    array_radius = 0.05  # cm
    n_mics = args.mics

    # fix the randomness for repeatability
    np.random.seed(args.seed)

    # STFT parameters
    framesize = 4096
    hop = framesize // 2
    win_a = pra.hamming(framesize)
    win_s = pra.transform.compute_synthesis_window(win_a, hop)

    # algorithm parameters
    n_iter = args.n_iter

    # Prepare the signals
    source_signals = [samples[i][1] for i in range(n_sources)]

    # Create the room itself
    room = make_room(
        room_dim,
        args.mics,
        source_signals,
        fs=fs,
        max_order=max_order,
        absorption=absorption,
        array_radius=array_radius,
    )

    # Run the simulation
    # premix.shape == (n_src, n_mics, n_samples)
    premix = room.simulate(return_premix=True,)

    # first normalize all separate recording to have unit power at microphone one
    p_mic_ref = np.std(premix[:, 0, :], axis=1)  # reference mic is 0
    premix /= p_mic_ref[:, None, None]

    # Total variance of noise components
    sigma_n = 10 ** (-args.snr / 10) * premix.shape[0]

    # Mix down the recorded signals
    mix = np.sum(premix, axis=0) + sigma_n * np.random.randn(*premix.shape[1:])

    print("Simulation done.")

    # START BSS
    ###########

    # shape: (n_frames, n_freq, n_mics)
    X_mics = pra.transform.analysis(mix.T, framesize, hop, win=win_a).astype(
        np.complex128
    )

    if args.init_pca:
        X_init, _ = piva.pca(X_mics)
    else:
        X_init = X_mics

    tic = time.perf_counter()

    # Run BSS
    Y = piva.algorithms[args.algo](
        X_init, n_src=n_sources, n_iter=n_iter, proj_back=False, model=args.dist,
    )

    # projection back
    Y = piva.project_back(Y, X_mics[:, :, 0])

    toc = time.perf_counter()

    print(f"Processing time: {toc - tic:.3f} s")

    # Run iSTFT
    if Y.shape[2] == 1:
        y = pra.transform.synthesis(Y[:, :, 0], framesize, hop, win=win_s)[:, None]
    else:
        y = pra.transform.synthesis(Y, framesize, hop, win=win_s)
    y = y[framesize - hop :, :].astype(np.float64)

    # when we separated more channels, pick the most energetic ones
    if y.shape[1] > n_sources:
        pwrs = np.linalg.norm(y, axis=0)
        new_order = np.argsort(pwrs)[-n_sources:]
        y = y[:, new_order]
    elif y.shape[1] == 1 and n_sources > 1:
        y = np.broadcast_to(y, (y.shape[0], n_sources))

    # Compare SIR
    #############
    m = np.minimum(y.shape[0], premix.shape[2])
    sdr, sir, sar, perm = piva.metrics.si_bss_eval(
        premix[: n_sources, 0, :m].T, y[:m, :]
    )

    # reorder the vector of reconstructed signals
    y = y[:, perm]

    print("SDR:", sdr)
    print("SIR:", sir)

    if args.gui:

        from tkinter import Tk

        # Make a simple GUI to listen to the separated samples
        root = Tk()
        my_gui = PlaySoundGUI(
            root, room.fs, mix[0, :], y.T, references=premix[:, 0, :]
        )
        root.mainloop()
