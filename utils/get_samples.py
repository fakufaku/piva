import os
import numpy as np
from urllib.request import urlretrieve
from scipy.io import wavfile

URL = "https://github.com/LCAV/pyroomacoustics/raw/master/examples/input_samples"
FOLDER = "samples"
FILES = [
    "cmu_arctic_us_aew_a0002.wav",
    "cmu_arctic_us_axb_a0004.wav",
    "cmu_arctic_us_axb_a0006.wav",
    "cmu_arctic_us_aew_a0001.wav",
    "cmu_arctic_us_aew_a0003.wav",
    "cmu_arctic_us_axb_a0005.wav",
]


class SampleList(object):
    def __init__(self, files):

        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fs, audio = wavfile.read(self.files[index])

        if audio.dtype == np.int16:
            audio = audio.astype(np.float64) / (2 ** 15)

        return fs, audio


def download_files():

    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)

    samples = []

    for filename in FILES:

        path = os.path.join(FOLDER, filename)
        url = os.path.join(URL, filename)

        samples.append(path)

        if not os.path.exists(path):
            urlretrieve(url, filename=path)

    return SampleList(samples)


samples = download_files()

if __name__ == "__main__":
    pass
