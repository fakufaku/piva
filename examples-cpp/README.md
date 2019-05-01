PIVA C++ Bindings Examples
==========================

These examples will create stand alone executables that can separate multichannel sound files.

Install Dependencies
--------------------

On macOS and Linux, the dependency on libsndfile can be satsified via conda.

    conda install libsndfile -c conda-forge

On windows, I still need to find out.

Compile
-------

    mkdir build
    cd build
    cmake ..
    make all
