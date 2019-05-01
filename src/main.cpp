/*
   Python bindings for PIVA library.
   Copyright (C) 2020  Robin Scheibler

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
   */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#undef min
#undef max
#endif

#include "pybind11/pybind11.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>

// #if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
// #undef VOID
// #undef FLOAT
// #undef DOUBLE
// #undef CHAR
// #undef BOOL
// #endif

#define FORCE_IMPORT_ARRAY
#include <xtensor/xarray.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

#include "piva/core.hpp"
#include "piva/auxiva.hpp"
#include "piva/five.hpp"
#include "piva/auxiva_iss.hpp"
#include "piva/overiva.hpp"
#include "piva/pca.hpp"
#include "piva/stft.hpp"

namespace py = pybind11;

// shortcut for pytensor 3D with complex double precision
using piva::Distribution;
using pytensor1_dp = xt::pytensor<double, 1>;
using pytensor2_dp = xt::pytensor<double, 2>;
using pytensor3_cpx_dp = xt::pytensor<complex<double>, 3>;

// Python Module and Docstrings
PYBIND11_MODULE(core, m) {
  xt::import_numpy();

  m.doc() = R"pbdoc(
        Source separation in C++

        .. currentmodule:: piva (core)

        .. autosummary::
           :toctree: _generate

           auxiva_laplace_core
           auxiva_gauss_core
           overiva_laplace_core
           auxiva_laplace_core
           auxiva_iss_laplace_core
           auxiva_iss_gauss_core
    )pbdoc";

  m.def("auxiva_laplace_core",
      piva::impl::auxiva<pytensor3_cpx_dp, double, Distribution::Laplace>,
      "Do AuxIVA (Laplace)");
  m.def("auxiva_gauss_core",
      piva::impl::auxiva<pytensor3_cpx_dp, double, Distribution::Gauss>,
      "Do AuxIVA (Gauss)");
  m.def("overiva_laplace_core",
      piva::impl::overiva<pytensor3_cpx_dp, double, Distribution::Laplace>,
      "Do OverIVA (Laplace)");
  m.def("overiva_gauss_core",
      piva::impl::overiva<pytensor3_cpx_dp, double, Distribution::Gauss>,
      "Do OverIVA (Gauss)");
  m.def("auxiva_iss_laplace_core",
      piva::impl::auxiva_iss<pytensor3_cpx_dp, double, Distribution::Laplace>,
      "Do MixIVA (Laplace)");
  m.def("auxiva_iss_gauss_core",
      piva::impl::auxiva_iss<pytensor3_cpx_dp, double, Distribution::Gauss>,
      "Do MixIVA (Gauss)");
  m.def("five_laplace_core",
      piva::impl::five<pytensor3_cpx_dp, double, Distribution::Laplace>,
      "Do FIVE (Laplace)");
  m.def("five_gauss_core",
      piva::impl::five<pytensor3_cpx_dp, double, Distribution::Gauss>,
      "Do FIVE (Gauss)");
  m.def("pca",
      piva::pca<double>,
      "Whiten the input data");

  // The cast to the function type is necessary here due to a bug in the GCC with
  // auto return value
  m.def("stft", (xt::xtensor<complex<double>, 3>(*)(pytensor2_dp, pytensor1_dp, size_t, size_t, size_t)) piva::impl::stft<pytensor2_dp, pytensor1_dp>, "STFT");
  m.def("istft", (xt::xtensor<double, 2>(*)(pytensor3_cpx_dp, pytensor1_dp, size_t, size_t, size_t)) piva::impl::istft<pytensor3_cpx_dp, pytensor1_dp>, "iSTFT");
}
