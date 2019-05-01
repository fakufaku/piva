/*
   Principal component analysis.
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
#ifndef __WHITENING_H__
#define __WHITENING_H__

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


template <class T>
void pca(xt::pytensor<complex<T>, 3> &mic_signals,
            xt::pytensor<complex<T>, 3> &decorr_matrices) {
  /*
  Whiten the input data by PCA.

  Provides the decorrelation matrices.

  Parameters
  ----------
  mic_signals: tensor3 (nfrequencies, nchannels, nframes)
      STFT representation of the signal
  decorr_matrices: tensor3 (nfrequencies, nchannels, nchannels)
      Tensor to store the decorrlation matrices
  */

  using xt::all;
  using xt::newaxis;
  using xt::range;

  // Run checks on the dimensions
  auto shape_in = mic_signals.shape();
  size_t n_freq = shape_in[0];
  size_t n_frames = shape_in[2];

  // Fill the noise decorr matrix and the covariance matrix
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, n_freq),
      [&](const tbb::blocked_range<size_t> &r) {
        for (size_t freq = r.begin(); freq != r.end(); ++freq) {
          auto Xf = xt::view(mic_signals, freq, all(), all());
          auto Wf = xt::view(decorr_matrices, freq, all(), all());

          // Covmat
          auto Cxf = xt::linalg::dot(Xf, xt::conj(xt::transpose(Xf))) * T(1. / n_frames);

          // Do the PCA
          auto eigh_ret = xt::linalg::eigh(Cxf);
          auto eig_val = std::get<0>(eigh_ret);
          auto eig_vec = std::get<1>(eigh_ret);

          // Whitening matrix
          auto eig_vec_H = xt::conj(xt::transpose(eig_vec, {1, 0}));
          auto eig_val_inv_sqrt = xt::view(1 / xt::sqrt(eig_val), all(), newaxis());
          Wf = eig_val_inv_sqrt * eig_vec_H;

          // Decorrelate the input data
          Xf =  xt::linalg::dot(Wf, Xf);
        }
      });

  return;
}

}  // namespace piva
#endif  // __WHITENING_H__
