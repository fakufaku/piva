/*
   Auxiliary function based independent vector analysis for blind
   source separation using iterative source steering algorithm.
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
#ifndef __MIXIVA_HPP__
#define __MIXIVA_HPP__

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xview.hpp>

#include <tbb/tbb.h>

#include "core.hpp"

using std::array;
using std::complex;

namespace piva {

namespace impl {

template <class E, class T, Distribution D>
void auxiva_iss(E &mic_signals, size_t n_iter) {
  /*
  Implementation of overdetermined IVA algorithm for BSS

  Orthogonal constraints only

  Parameters
  ----------
  mic_signals: xtensor 3D (nframes, nfrequencies, nchannels)
      STFT representation of the signal
  n_iter: int, optional
      The number of iterations (default 20)
  */
  using xt::all;
  using xt::newaxis;
  using xt::range;

  // infer the size of each dimensions
  auto shape = mic_signals.shape();
  size_t n_freq = shape[0];
  size_t n_chan = shape[1];
  size_t n_frames = shape[2];

  // Array of activations (real valued)
  xt::xtensor<T, 2> inv_act({n_chan, n_frames});

  // No need to demix at the beginning since we initialize the filters with
  // identity

  for (size_t epoch = 0; epoch < n_iter; epoch++) {

    compute_activations<E, T, D>(mic_signals, inv_act);

      // We use a parallel for loop over the frequency-wise processing
      tbb::parallel_for(tbb::blocked_range<size_t>(0, n_freq),
        [&](const tbb::blocked_range<size_t> &r) {

          auto v = xt::xtensor<complex<T>, 1>::from_shape({n_chan});
          auto v_denom = xt::xtensor<T, 1>::from_shape({n_chan});

          for (size_t src = 0; src < n_chan; src++) {
            for (size_t freq = r.begin(); freq != r.end(); ++freq) {
              auto Yf = xt::view(mic_signals, freq, all(), all());

              auto Yfs = xt::view(mic_signals, freq, src, all());
              auto conjYfs = xt::conj(Yfs);

              // compute the auxilliary variable (weighted
              // covariance matrix)
              auto v_num = xt::linalg::dot(Yf * inv_act, conjYfs);
              auto Yfs_mag_sq = xt::square(xt::abs(Yfs));
              v_denom = xt::linalg::dot(inv_act, Yfs_mag_sq);

              // The matrix multiplication W * V
              v = v_num / v_denom;
              v(src) -= 1. / sqrt(v_denom(src));

              // Update the output signals
              Yf -= xt::linalg::outer(v, xt::view(mic_signals, freq, src, all()));
            }
          }
        });
  }

  return;
}

}  // namespace impl


// C++ API
template <class T, Distribution D = Distribution::Laplace>
void auxiva_iss(xt::xtensor<complex<T>, 3> &mic_signals, size_t n_iter) {
  impl::auxiva_iss<xt::xtensor<complex<T>, 3>, T, D>(mic_signals, n_iter);
}


} // namespace piva
#endif  // __MIXIVA_HPP__
