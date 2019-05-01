/*
   Some sub-routines for BSS output scaling and to load audio files.
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
#include <iostream>
#include <cmath>
#include <cstdint>
#include <utility>

// xtensor stuff
#include <xtensor/xcomplex.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-io/xaudio.hpp>

// Include TBB later to avoid an error
#include <tbb/tbb.h>

#include "core.hpp"

namespace piva {
namespace impl {

using std::complex;
using xt::all;
using xt::newaxis;
using xt::range;
using xt::placeholders::_;

template <class E1, class E2>
auto projection_back_weights(E1&& signals, E2&& reference) {
  /*
   * Computes the weights that project signal onto reference
   *
   * Parameters
   * ----------
   * signals: complex xtensor (n_frequencies, n_channels, n_frames)
   *   The separated signals
   * reference: complex xtensor (n_frequencies, n_frames)
   *   The reference spectrogram
   *
   * Returns
   * -------
   * complex xtensor (n_frequencies, n_channels)
   */

  // sum over frames
  std::cout << "Computing the num" << std::endl;
  auto num = xt::sum(
      xt::view(std::forward<E2>(reference), all(), newaxis(), all()) * xt::conj(std::forward<E1>(signals)), {2});
  std::cout << "Computing the denom" << std::endl;
  auto denom = xt::norm_sq(std::forward<E1>(signals), {2});

  std::cout << "Computing the division" << std::endl;

  return std::move(num) / xt::maximum(std::move(denom), 1e-15);
}

template <class E1, class E2>
void project_back(E1&& signals, E2&& reference) {
  /*
   * Projects signal onto reference in-place
   *
   * Parameters
   * ----------
   * signals: complex xtensor (n_frequencies, n_channels, n_frames)
   *   The separated signals
   * reference: complex xtensor (n_frequencies, n_frames)
   *   The reference spectrogram
   */
  auto weights = xt::eval(projection_back_weights(std::forward<E1>(signals), std::forward<E2>(reference)));
  signals *= xt::view(std::move(weights), all(), all(), newaxis());
}

}  // namespace impl

// C++ API
template <class T, class E>
auto projection_back_weights(xt::xtensor<complex<T>, 3>& signals,
                             E&& reference) {
  return impl::projection_back_weights(signals, std::forward<E>(reference));
}

template <class T, class E>
void project_back(xt::xtensor<complex<T>, 3>& signals,
                  E&& reference) {
  impl::project_back(signals, std::forward<E>(reference));
}

template<class T>
auto load(const std::string& filename, bool scale = false) {
  /*
   * Load an audio file and does some scaling on it
   */

  auto audio = xt::load_audio(filename);
  auto sampling_freq = std::get<0>(audio);             // sampling frequency
  xt::xtensor<T, 2> audio_arr = xt::cast<T>(std::get<1>(std::move(audio)));  // audio contents

  if (scale)
    audio_arr /= (1 << 15);
 
  return std::make_tuple(sampling_freq, std::move(audio_arr));
}

template<class E>
auto dump(const std::string& filename, E&& array, int samplerate, bool scale = false) {

  if (scale)
    array *= (1 << 15);

  xt::dump_audio(filename, xt::cast<int16_t>(std::forward<E>(array)), samplerate);
}

}  // namespace piva
