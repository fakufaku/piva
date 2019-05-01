/*
   Fast Independent Vector Extraction.
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
#ifndef __FIVE_HPP__
#define __FIVE_HPP__

#include <array>
#include <cassert>
#include <cmath>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xview.hpp>

#include <tbb/tbb.h>

#include "core.hpp"
#include "pca.hpp"

using std::array;
using std::complex;

namespace piva {

namespace impl {

template <class E, class T, Distribution D>
void five(E &&mic_signals,
          E &&sources_out,
          E &&demixing_matrices,
          size_t n_iter) {
  /*
  Implementation of Fast Independent Vector Extraction

  Parameters
  ----------
  mic_signals: ndarray (nfrequencies, nchannels, nframes)
      STFT representation of the signal
  source_out: (nfrequencies, 1, nframes)
      The output sources
  demixing_matrices: (nfrequencies, nchannels, nchannels)
      The demixing matrices
  n_iter: int, optional
      The number of iterations (default 20)
  */

  using tensor2 = xt::xtensor<complex<T>, 2>;
  using xt::all;
  using xt::newaxis;
  using xt::range;

  // Run checks on the dimensions
  auto shape_in = mic_signals.shape();
  size_t n_freq = shape_in[0];
  size_t n_chan = shape_in[1];
  size_t n_frames = shape_in[2];

  // Normalizing constant for the covmat computation
  auto covmat_norm = T(1. / n_frames);

  // Whiten the input signal, the dominant signal is placed in
  // the last channel. We'll use it as a starting point
  pca(mic_signals, demixing_matrices);

  // Array of inverse activations (real valued)
  xt::xtensor<T, 2> inv_act({1, n_frames});
  auto inv_act_s = xt::view(inv_act, 0, newaxis(), all());

  // Pick the initial source as the principal component (last channel)
  xt::view(sources_out, all(), 0, all()) = xt::view(mic_signals, all(), n_chan - 1, all());

  for (size_t epoch = 0; epoch < n_iter; epoch++) {

    compute_activations<E, T, D>(sources_out, inv_act);

    // We use a parallel for loop over the frequency-wise processing
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, n_freq),
      [&](const tbb::blocked_range<size_t> &r) {

        for (size_t freq = r.begin(); freq != r.end(); ++freq) {
          auto Xf = xt::view(mic_signals, freq, all(), all());
          auto Yfs = xt::view(sources_out, freq, 0, newaxis(), all());

          // compute the auxilliary variable (weighted covariance matrix)
          tensor2 V =
              covmat_norm *
              xt::linalg::dot(Xf * inv_act_s, xt::conj(xt::transpose(Xf)));

          // compute the new demixing vector
          auto eigh_ret = xt::linalg::eigh(V);
          auto eig_val = std::get<0>(eigh_ret);
          auto eig_vec = std::get<1>(eigh_ret);

          // The new demixing filter is the eigenvector corresponding to
          // the smallest eigenvalue
          auto norm = 1. / sqrt(eig_val(0));
          auto new_vec = xt::view(eig_vec, newaxis(), all(), 0) * norm;

          // save new demixing vector and demix that frequency
          auto demix_vec = xt::conj(new_vec);
          Yfs = xt::linalg::dot(demix_vec, Xf);

          // Save demixing matrices in last iteration
          // We need to compose with decorrelation matrix
          if (epoch == n_iter - 1)
          {
            auto Wf = xt::view(demixing_matrices, freq, all(), all());
            auto new_mat = xt::conj(xt::transpose(eig_vec));
            Wf = xt::linalg::dot(new_mat, Wf);
            xt::view(Wf, 0, all()) *= norm;
          }
        }
      }
    );

  }  // epochs

  return;
}

}  // namespace impl


// C++ API
template <class T, Distribution D>
void five(xt::xtensor<complex<T>, 3> &mic_signals,
            xt::xtensor<complex<T>, 3> &sources_out,
            xt::xtensor<complex<T>, 3> &demixing_matrices,
            size_t n_iter) {

  impl::five<xt::xtensor<complex<T>, 3>, T, D>(mic_signals, sources_out, demixing_matrices, n_iter);
}


}
#endif  // __FIVE_HPP__
