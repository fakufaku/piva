/*
   Auxiliary function based independent vector analysis for blind
   source separation.
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
#ifndef __AUXIVA_HPP__
#define __AUXIVA_HPP__

#include <array>
#include <cassert>
#include <cmath>
#include <utility>

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
void auxiva(E &mic_signals,
            E &sources_out,
            E &demixing_matrices,
            size_t n_iter) {
  /*
  Implementation of overdetermined IVA algorithm for BSS

  Orthogonal constraints only

  Parameters
  ----------
  mic_signals: xtensor 3D (nframes, nfrequencies, nchannels)
      STFT representation of the signal
  sources_out: xtensor 3D (nframes, nfrequencies, nsrc)
      STFT representation of the separated signals
  demixing_matrices: xtensor 3D (nfrequencies, nchannels, nchannels)
      The demixing matrices
  n_iter: int, optional
      The number of iterations
  */

  using xt::all;
  using xt::newaxis;
  using xt::range;

  // Run checks on the dimensions
  auto shape_in = mic_signals.shape();
  size_t n_freq = shape_in[0];
  size_t n_chan = shape_in[1];
  size_t n_frames = shape_in[2];

  // First demixing
  for (size_t freq = 0; freq < n_freq; freq++) {
    auto Wf = xt::view(demixing_matrices, freq, all(), all());
    auto Xf = xt::view(mic_signals, freq, all(), all());
    xt::view(sources_out, freq, all(), all()) = xt::linalg::dot(Wf, Xf);
  }

  // We need the canonical basis
  xt::xtensor<complex<T>, 2> the_eye = xt::eye(n_chan);

  // Array of inverse activations (real valued)
  xt::xtensor<T, 2> inv_act({n_chan, n_frames});

  // Used for the right hand size when solving for the new filter
  xt::xtensor<complex<T>, 1> rhs = xt::zeros<complex<T>>({n_chan});

  for (size_t epoch = 0; epoch < n_iter; epoch++) {
    compute_activations<E, T, D>(sources_out, inv_act);

    // We use a parallel for loop over the frequency-wise processing
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n_freq),
        [&](const tbb::blocked_range<size_t> &r) {

          auto V = xt::xtensor<complex<T>, 2>::from_shape({n_chan, n_chan});
          auto new_vec = xt::xtensor<complex<T>, 1>::from_shape({n_chan});

          for (size_t src = 0; src < n_chan; src++) {
            auto inv_act_s = xt::view(inv_act, src, newaxis(), all());
            auto rhs = xt::view(the_eye, src, all());

            for (size_t freq = r.begin(); freq != r.end(); ++freq) {
              auto demix_vec = xt::view(demixing_matrices, freq, src, all());
              auto Xf = xt::view(mic_signals, freq, all(), all());
              auto Wf = xt::view(demixing_matrices, freq, all(), all());
              auto Yfs = xt::view(sources_out, freq, src, newaxis(), all());

              // compute the auxilliary variable (weighted covariance matrix)
              V =
                  T(1. / n_frames) *
                  xt::linalg::dot(Xf * inv_act_s, xt::conj(xt::transpose(Xf)));

              // The matrix multiplication W * V
              auto WV = xt::linalg::dot(Wf, V);

              // update the demixing vector
              // Finally solve the equation (W * V) new_w = e_k
              new_vec = xt::linalg::solve(WV, rhs);

              // normalization
              // note: vdot conjugates the first argument for complex vectors
              T denom = xt::real(
                  xt::linalg::vdot(new_vec, xt::linalg::dot(V, new_vec)));
              new_vec /= sqrt(denom);

              // save new demixing vector and demix that frequency
              demix_vec = xt::conj(new_vec);
              Yfs = xt::linalg::dot(xt::view(demix_vec, newaxis(), all()), Xf);
            }
          }
        });
  }

  return;
}

}  // namespace impl

template <class T>
using xtensor3_cpx_t = xt::xtensor<complex<T>, 3>;

// C++ API
template <class T, Distribution D>
void auxiva(xtensor3_cpx_t<T> &mic_signals,
            xtensor3_cpx_t<T> &sources_out,
            xtensor3_cpx_t<T> &demixing_matrices,
            size_t n_iter) {

  impl::auxiva<xt::xtensor<complex<T>, 3>, T, D>(mic_signals,
      sources_out, 
      demixing_matrices, n_iter);
}

template <class T, Distribution D = Distribution::Laplace>
auto auxiva(xtensor3_cpx_t<T> &mic_signals,
            size_t n_iter) {

  // Run checks on the dimensions
  auto shape_in = mic_signals.shape();
  size_t n_freq = shape_in[0];
  size_t n_chan = shape_in[1];
  // size_t n_frames = shape_in[2];

  // Start from identity demixing matrix
  xt::xtensor<complex<T>, 3> demixing_matrices = xt::zeros<complex<T>>({n_freq, n_chan, n_chan});
  for (size_t f = 0 ; f < n_freq ; f++)
    for (size_t c = 0 ; c < n_chan ; c++)
      demixing_matrices(f, c, c) = 1.;

  // Copy the input to output
  xt::xtensor<complex<T>, 3> sources_out = mic_signals;

  impl::auxiva<xt::xtensor<complex<T>, 3>, T, D>(mic_signals, sources_out, demixing_matrices, n_iter);

  return sources_out;
}


} // namespace piva
#endif  // __AUXIVA_HPP__
