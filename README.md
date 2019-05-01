PIVA: Performance Independent Vector Analysis Library
=====================================================

**/!\\ This repository is a work in progress /!\\**

[![Actions Status](https://github.com/fakufaku/piva/workflows/piva/badge.svg)](https://github.com/fakufaku/piva/actions)

Blind source separation allows to separates multiple sound sources that have been mixed together in a multi-channel audio file.
This is a performance-oriented implementation based on independent vector analysis.
The implementations are for batch processing with fully parallel processing.
At the moment, we implement two algorithms.

1. AuxIVA with iterative projection (IP) [\[1\]](#auxiva_ip)
2. AuxIVA with iterative source steering (AuxIVA-ISS) [\[2\]](#auxiva_iss)
3. OverIVA for when there are more microphones than sources [\[3\]](#overiva)
4. Fast independent vector exctraction (FIVE) that extracts only one source [\[4\]](#five)

ISS is faster than IP, especially for larger number of channels.
Separation performance is about the same.
OverIVA is suitable to extract few sources from a larger number of channels.
FIVE extracts a single source, but is very fast.
The library is a C++ header-only library with Python bindings.

Author
------

[Robin Scheibler](fakufaku[at]gmail[dot]com)

Use the library
---------------

### C++

The library is header only with the headers in `include/piva`.

### Python

At the moment, the package is only supported through Anaconda.

#### Quick Start

    conda install piva -c fakufaku

#### Manual Instalation

First, clone this repository.

    git clone https://github.com/fakufaku/piva

Make sure you have the dependencies installed

    cd piva
    conda env create -f environment

Build the module

    conda activate piva
    pip install .

#### Windows runtime requirements

On Windows, the Visual C++ 2015 redistributable packages are a runtime
requirement for this project. It can be found [here](https://www.microsoft.com/en-us/download/details.aspx?id=48145).

If you use the Anaconda python distribution, you may require the Visual Studio
runtime as a platform-dependent runtime requirement for you package:

    requirements:
      build:
        - python
        - setuptools
        - pybind11

      run:
       - python
       - vs2015_runtime  # [win]

### Dependencies

- XTensor
- Intel Threading Building Blocks (Intel TBB)
- FFTW (for the STFT)
- SndFile (for C++ examples, to read audio files)

Building the documentation
--------------------------

**Documentation to be added later**

Documentation for the example project is generated using Sphinx. Sphinx has the
ability to automatically inspect the signatures and documentation strings in
the extension module to generate beautiful documentation in a variety formats.
The following command generates HTML-based reference documentation; for other
formats please refer to the Sphinx manual:

 - `piva/docs`
 - `make html`


Running the tests
-----------------

**Tests to be added later**

Running the tests requires `pytest`.

    py.test .

License and Attribution
-----------------------

Copyright 2020 Robin Scheibler. GPL3 License (see LICENSE).

If you use this library for an academic publication, please cite [2].
Please also cite the paper for the relevant algorithm used.

References
----------

<a name="auxiva_ip">[1]</a> Nobutaka Ono, "Stable and fast update rules for independent vector analysis based on auxiliary function technique," Proc. WASPAA, 2011. </br>
<a name="auxiva_iss">[2]</a> Robin Scheibler, Nobutaka Ono, "Fast and stable blind source separation with rank-1 updates," Proc. IEEE ICASSP, 2020. </br>
<a name="overiva">[3]</a> Robin Scheibler, Nobutaka Ono, "Independent vector analysis with more microphones than sources," Proc. WASPAA, 2019. </br>
<a name="five">[4]</a> Robin Scheibler, Nobutaka Ono, "Fast independent vector extraction by iterative SINR maximization," Proc. IEEE ICASSP, 2020. </br>
