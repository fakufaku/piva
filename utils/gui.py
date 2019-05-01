#  A simple GUI to listen to sound samples
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
from tkinter import Tk, Button, Label

try:
    import sounddevice as sd

    has_sounddevice = True
except ImportError:
    has_sounddevice = False

# Now come the GUI part
class PlaySoundGUI(object):
    def __init__(self, master, fs, mix, sources, references=None):
        assert has_sounddevice, (
            "Soundevice needs to be installed to use the Player GUI."
            "Please do 'pip install sounddevice'"
        )

        self.master = master
        self.fs = fs
        self.mix = mix
        self.sources = sources
        self.sources_max = np.max(np.abs(sources))
        self.references = references.copy()
        master.title("Comparator")

        if self.references is not None:
            self.references *= 0.75 / np.max(np.abs(self.references))

        nrow = 0

        self.label = Label(master, text="Listen to the output.")
        self.label.grid(row=nrow, columnspan=2)
        nrow += 1

        self.mix_button = Button(
            master, text="Mix", command=lambda: self.play(self.mix)
        )
        self.mix_button.grid(row=nrow, columnspan=2)
        nrow += 1

        self.buttons = []
        for i, source in enumerate(self.sources):
            self.buttons.append(
                Button(
                    master,
                    text="Source " + str(i + 1),
                    command=lambda src=source: self.play(src),
                )
            )

            if self.references is not None:
                self.buttons[-1].grid(row=nrow, column=1)
                self.buttons.append(
                    Button(
                        master,
                        text="Ref " + str(i + 1),
                        command=lambda rs=self.references[i, :]: self.play(rs),
                    )
                )
                self.buttons[-1].grid(row=nrow, column=0)

            else:
                self.buttons[-1].grid(row=nrow, columnspan=2)

            nrow += 1

        self.stop_button = Button(master, text="Stop", command=sd.stop)
        self.stop_button.grid(row=nrow, columnspan=2)
        nrow += 1

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.grid(row=nrow, columnspan=2)
        nrow += 1

    def play(self, src):
        sd.play(0.75 * src / self.sources_max, samplerate=self.fs, blocking=False)
