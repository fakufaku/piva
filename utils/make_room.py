#  Few routines to place microphones in simulation and playback sound in GUI.
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

import math
import numpy as np
import pyroomacoustics as pra
from tkinter import Tk, Button, Label


def make_room(
    room_dim,
    n_mics,
    source_signals,
    fs=16000,
    max_order=17,
    absorption=0.35,
    array_radius=0.05,
):

    room_dim = np.array(room_dim)
    n_sources = len(source_signals)

    room = pra.ShoeBox(
        room_dim, fs=fs, absorption=absorption, max_order=max_order
    )

    # mic array centered close, but slightly off the center of the room
    mic_center = room_dim / 2 * (0.995 + 0.01 * np.random.rand(*room_dim.shape))

    # circular microphone array
    theta = 2 * np.pi * np.arange(n_mics) / n_mics
    R = mic_center[:, None] + array_radius * np.array(
        [np.cos(theta), np.sin(theta), np.zeros(n_mics)]
    )
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    # sources at random in the room
    S = np.random.rand(len(source_signals), *room_dim.shape) * room_dim[None, :]
    for loc, signal in zip(S, source_signals):
        room.add_source(loc, signal=signal)

    return room
