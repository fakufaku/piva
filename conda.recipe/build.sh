#!/bin/bash

unset MACOSX_DEPLOYMENT_TARGET

# By default conda-build uses the "-dead_strip_dylibs"
# flags which for some reason causes libmkl_sequential and libfftw
# to not be properly linked on macOS
# see: https://github.com/AnacondaRecipes/intel_repack-feedstock/issues/8
LDFLAGS=""

# Now build the package
${PYTHON} ./setup.py install

# Add more build steps here, if they are necessary.

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
