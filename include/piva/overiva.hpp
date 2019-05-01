/*
   Overdetermined auxiliary function based independent vector analysis.
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
#ifndef __OVERIVA_HPP__
#define __OVERIVA_HPP__

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

using std::array;
using std::complex;

namespace piva {

namespace impl {

template <class E, class T, Distribution D>
void overiva(E &&mic_signals,
             E &&sources_out,
             E &&demixing_matrices,
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

  using tensor3 = xt::xtensor<complex<T>, 3>;
  using tensor2 = xt::xtensor<complex<T>, 2>;
  using tensor1 = xt::xtensor<complex<T>, 1>;
  using xt::all;
  using xt::newaxis;
  using xt::range;

  // Acquire the dimensions
  auto shape_in = mic_signals.shape();
  auto shape_out = sources_out.shape();
  size_t n_freq = shape_in[0];
  size_t n_chan = shape_in[1];
  size_t n_frames = shape_in[2];
  size_t n_src = shape_out[1];

  // First demixing
  for (size_t freq = 0; freq < n_freq; freq++) {
    auto Wf = xt::view(demixing_matrices, freq, all(), all());
    auto Xf = xt::view(mic_signals, freq, all(), all());
    xt::view(sources_out, freq, all(), all()) = xt::linalg::dot(Wf, Xf);
  }

  // Array of activations (real valued)
  xt::xtensor<T, 2> inv_act({n_src, n_frames});

  // covariance matrix of input signal (n_freq, n_chan, n_chan)
  // Essentially, this should do: X X^H / N, at all frequencies
  tensor3 Cx = xt::zeros<complex<T>>({n_freq, n_chan, n_chan});

  // Fill the noise demixing matrix and the covariance matrix
  if (n_src < n_chan) {
    for (size_t freq = 0; freq < n_freq; freq++) {
      auto Xf = xt::view(mic_signals, freq, all(), all());
      auto Wf = xt::view(demixing_matrices, freq, all(), all());
      auto Cxf = xt::view(Cx, freq, all(), all());

      // Covmat
      Cxf = xt::linalg::dot(Xf, xt::conj(xt::transpose(Xf))) * T(1. / n_frames);
    }
  }

  // Used for the right hand size when solving for the new filter
  xt::xtensor<complex<T>, 2> the_eye = xt::eye(n_chan);

  for (size_t epoch = 0; epoch < n_iter; epoch++) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_frames),
                      [&](const tbb::blocked_range<size_t> &r) {
                        auto Y = xt::view(sources_out, all(), all(),
                                          range(r.begin(), r.end()));
                        auto ia =
                            xt::view(inv_act, all(), range(r.begin(), r.end()));

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

    // We use a parallel for loop over the frequency-wise processing
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n_freq),
        [&](const tbb::blocked_range<size_t> &r) {
          for (size_t freq = r.begin(); freq != r.end(); ++freq) {
            auto Xf = xt::view(mic_signals, freq, all(), all());
            auto Wf = xt::view(demixing_matrices, freq, all(), all());
            auto Cxf = xt::view(Cx, freq, all(), all());

            // Update the demixing filter for each source
            for (size_t src = 0; src < n_src; src++) {
              auto inv_act_s = xt::view(inv_act, src, newaxis(), all());
              auto demix_vec = xt::view(demixing_matrices, freq, src, all());
              auto Yfs = xt::view(sources_out, freq, src, newaxis(), all());

              // compute the auxilliary variable (weighted covariance matrix)
              // This intermediate value is needed again later
              tensor2 V =
                  T(1. / n_frames) *
                  xt::linalg::dot(Xf * inv_act_s, xt::conj(xt::transpose(Xf)));

              // Now we update the noise part of the demixing matrix
              auto WC = xt::linalg::dot(Wf, Cxf);
              auto WC_left = xt::view(WC, all(), range(0, n_src));
              auto WC_right = xt::view(WC, all(), range(n_src, n_chan));
              auto J_H = xt::linalg::solve(WC_left, WC_right);
              auto Jf = xt::conj(xt::transpose(J_H));

              // The matrix multiplication W * V
              auto WV_top = xt::linalg::dot(Wf, V);
              auto WV_bot =
                  (xt::linalg::dot(Jf, xt::view(V, range(0, n_src), all())) -
                   xt::view(V, range(n_src, n_chan), all()));
              auto WV = xt::concatenate(std::make_tuple(WV_top, WV_bot), 0);

              // update the demixing vector
              // Finally solve the equation (W * V) new_w = e_k
              auto rhs = xt::view(the_eye, src, all());
              tensor1 new_vec = xt::linalg::solve(WV, rhs);

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

} // namespace impl

// C++ API
template <class T, Distribution D>
void overiva(xt::xtensor<complex<T>, 3> &mic_signals,
             xt::xtensor<complex<T>, 3> &sources_out,
             xt::xtensor<complex<T>, 3> &demixing_matrices,
             size_t n_iter) {

  impl::overiva<xt::xtensor<complex<T>, 3>, T, D>(mic_signals, sources_out, demixing_matrices, n_iter);
}


} // namespace piva
#endif  // __OVERIVA_HPP__
