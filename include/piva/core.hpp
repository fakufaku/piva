/*
   Core sub-routines for blind source separation.
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
#ifndef __CORE_HPP__
#define __CORE_HPP__

#define PIVA_VERSION_MAJOR 0
#define PIVA_VERSION_MINOR 0
#define PIVA_VERSION_PATCH 1-a

#include <complex>
#include <iostream>
#include <string>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xview.hpp>

#include <tbb/tbb.h>

// PIVA: Parallel Independent Vector Analysis
// (also in Czeck: Piva == Beer)

namespace piva {

using std::complex;
using xt::all;
using xt::range;
using xt::newaxis;
using xt::placeholders::_;

enum class Distribution { Laplace, Gauss };

template <class T>
void print_debug(const char *msg, T value) {
  std::cout << msg << " " << value << std::endl << std::flush;
}


template <class T>
void print_shape(const xt::xexpression<T> &expr) {
  auto shape = expr.derived_cast().shape();
  std::cout << "Shape == ";
  for (size_t d = 0; d < shape.size(); d++) std::cout << shape[d] << " ";
  std::cout << std::endl << std::flush;
}


template <class E, class T, Distribution D>
void compute_activations(
    const E &input, xt::xtensor<T, 2> &output) {
  /*
  Reduces along the first dimension of the input tensor and stores the
  result in the output tensor
  */

  // Run checks on the dimensions
  auto shape_in = input.shape();
  size_t n_freq = shape_in[0];
  size_t n_frames = shape_in[2];

  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, n_frames),
    [&](const tbb::blocked_range<size_t> &r) {
      auto Y = xt::view(input, all(), all(), range(r.begin(), r.end()));
      auto ia = xt::view(output, all(), range(r.begin(), r.end()));

      // average squared magnitude along frequencies, shape
      // (n_chan, n_frames)
      if (D == Distribution::Gauss) {
        // auto act = xt::mean(xt::square(xt::abs(Y)), 0);  //
        // time-varying Gaussian activations
        auto act = xt::norm_sq(Y, {0}) * T(1. / n_freq);
        ia = 1. / xt::maximum(act, 1e-10);
      } else if (D == Distribution::Laplace) {
        // auto act = xt::sqrt(xt::sum(xt::square(xt::abs(Y)),
        // 0));  // Laplace activations
        auto act = 2. * xt::norm_l2(Y, {0});
        ia = 1. / xt::maximum(act, 1e-10);
      }
    });
}


// Compute a weighted covariance matrix
template <class E1, class E2>
xt::xtensor<complex<double>, 2> compute_covmat(const E1 &data, const E2 &weights)
{
  /*
  Efficient computation of the covariance matrix of complex data

  data:  shape = (ndim, nframes)
    data matrix containing one vector of data in each column
  weights: shape = (1, nframes)
    a vector of weights that correspond to each data frame
  */
  auto shape_data = data.shape();
  size_t ndim = shape_data[0];
  size_t nframes = shape_data[1];

  double norm = 1. / nframes;

  // This should be a 2D tensor
  xt::xtensor<complex<double>, 2> covmat({ndim, ndim});

  auto data_H = xt::conj(xt::transpose(data));

  // compute first half
  for (size_t r = 0 ; r < ndim ; r++)
  {
    auto weighted_row = xt::view(data, r, newaxis(), all()) * weights;
    auto data_H_loc = xt::view(data_H, all(), range(r, _));

    auto covmat_row = xt::view(covmat, r, newaxis(), range(r, _));
    covmat_row = norm * xt::linalg::dot(weighted_row, data_H_loc);
  }

  // copy to second half
  for (size_t r = 1 ; r < ndim ; r++)
    for (size_t c = 0 ; c < r ; c++)
      covmat(r, c) = std::conj(covmat(c, r));

  return covmat;
}


} // namespace piva
#endif  // __CORE_HPP__
