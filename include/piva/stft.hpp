/*
   Short-time Fourier transform, forward and backward.
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
#ifndef __STFT_H__
#define __STFT_H__

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <xtensor-fftw/basic.hpp>
#include <xtensor-fftw/helper.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

// TBB needs to be included after the xtensor fail
// lest compilation fails on mac...
#include <tbb/tbb.h>

namespace piva {

namespace impl {

template <class E1, class E2>
auto stft(E1&& td_signals, E2&& window, size_t n_shift, size_t n_zeropad_front,
          size_t n_zeropad_back) {
  /*
   * The FFT length is infered from the window length that is assumed to
   *
   * Parameters
   * ----------
   * td_signals: 2D real xtensor (n_samples, n_channels)
   *   The time-domain input signal
   * window: 1D xtensor (n_frame)
   *   A window function the size of the analysis frame
   *
   * Returns
   * -------
   * 3D complex xtensor (n_frequencies, n_channels, n_frames)
   */

  using xt::all;
  using xt::range;

  // infer the local complex type
  using real_t =
      typename std::remove_pointer<typename std::remove_const<decltype(td_signals.data())>::type>::type;
  using complex_t = std::complex<real_t>;

  // infer the size of each dimensions
  auto shape = td_signals.shape();
  size_t n_samples = shape[0];
  size_t n_chan = shape[1];

  auto shape_win = window.shape();
  size_t n_analysis = shape_win[0];

  size_t n_frames = (n_samples + n_analysis - n_shift) / n_shift;
  if (n_frames * n_shift - n_analysis + n_shift < n_samples) n_frames++;

  size_t n_fft = n_zeropad_front + n_analysis + n_zeropad_back;
  size_t n_freq = n_fft / 2 + 1;

  xt::xtensor<complex_t, 3> stft_signals({n_freq, n_chan, n_frames});

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, n_frames),
      [&](const tbb::blocked_range<size_t>& r) {
        // input buffer for the fft
        xt::xarray<real_t> input = xt::zeros<real_t>({n_fft});

        for (size_t frame = r.begin(); frame != r.end(); ++frame) {
          size_t n = frame * n_shift;    // current offset in the input buffer
          size_t s2 = n + n_shift;       //  end of current analysis window
          ssize_t s1 = s2 - n_analysis;  // beginning of analysis window

          ssize_t i1 = n_zeropad_front;
          ssize_t i2 = i1 + n_analysis;

          // handle the beginning
          if (s1 < 0) {
            i1 -= s1;
            s1 = 0;
          }

          // handle the end of the buffer
          if (s2 > n_samples) {
            // fill with zeros the part of the buffer not used
            size_t new_i2 = i1 + n_samples - s1;
            xt::view(input, range(new_i2, i2)).fill(0.);
            // update the bounds
            i2 = new_i2;
            s2 = n_samples;
          }

          for (size_t chan = 0; chan < n_chan; chan++) {
            // copy from signals to input buffer
            xt::view(input, range(i1, i2)) =
                xt::view(td_signals, range(s1, s2), chan);

            // apply window function
            xt::view(input, range(n_zeropad_front,
                                  n_zeropad_front + n_analysis)) *= window;

            // view to the output buffer
            auto output = xt::view(stft_signals, all(), chan, frame);
            output = xt::fftw::rfft(input);
          }
        }
      });

  return stft_signals;
}

template <class E1, class E2>
auto istft(E1& stft_signals, E2& window, size_t n_shift,
           size_t n_zeropad_front, size_t n_zeropad_back) {
  /*
   * The FFT length is infered from the window length that is assumed to
   *
   * Parameters
   * ----------
   * stft_signals: 3D complex xtensor (n_frequencies, n_channels, n_frames)
   *   The STFT domain signal
   * window: 1D xtensor (n_frame)
   *   A window function the size of the analysis frame
   *
   * Returns
   * -------
   * 2D real xtensor (n_samples, n_channels)
   */

  using xt::all;
  using xt::range;

  // infer the local complex type
  using complex_t =
      typename std::remove_pointer<decltype(stft_signals.data())>::type;
  complex_t tmp{};  // dummy to get the real type associated
  using real_t = decltype(std::real(tmp));

  // infer the size of each dimensions
  auto shape = stft_signals.shape();
  size_t n_freq = shape[0];
  size_t n_chan = shape[1];
  size_t n_frames = shape[2];

  auto shape_win = window.shape();
  size_t n_analysis = shape_win[0];

  size_t n_fft = n_zeropad_front + n_analysis + n_zeropad_back;

  // make sure the parameters are consistent
  assert(n_fft / 2 + 1 == n_freq);

  // create the output array
  size_t n_samples = n_frames * n_shift + (n_shift - n_analysis);

  xt::xtensor<real_t, 2> td_signals = xt::zeros<real_t>({n_samples, n_chan});

  // buffer for current frame
  xt::xarray<complex_t> buffer_fft_in = xt::zeros<complex_t>({n_freq});
  xt::xarray<real_t> buffer_fft_out = xt::zeros<real_t>({n_fft});
  bool odd_last_dim = n_fft % 2 == 1;

  // do the overlap add thing
  // Check how to replace this by a parallel reduce later
  for (size_t frame = 0; frame < n_frames; ++frame) {
    size_t n = frame * n_shift;  // current offset in the output buffer

    size_t s2 = n + n_shift + n_zeropad_back;  //  end of current fft
    ssize_t s1 = s2 - n_fft;                   // beginning of fft

    ssize_t i1 = 0;  // offset in the current frame buffer
    ssize_t i2 = n_fft;

    // handle the beginning
    if (s1 < 0) {
      i1 -= s1;  // in fact adding a positive number here
      s1 = 0;
    }

    // handle the end of the buffer
    if (s2 > n_samples) {
      // update the bounds
      i2 = i1 + n_samples - s1;
      s2 = n_samples;
    }

    for (size_t chan = 0; chan < n_chan; chan++) {
      // inverse fft
      buffer_fft_in = xt::view(stft_signals, all(), chan, frame);
      buffer_fft_out = xt::fftw::irfft(buffer_fft_in, odd_last_dim);

      // apply synthesis window
      xt::view(buffer_fft_out,
               range(n_zeropad_front, n_zeropad_front + n_analysis)) *= window;

      // overlap and add
      xt::view(td_signals, range(s1, s2), chan) +=
          xt::view(buffer_fft_out, range(i1, i2));
    }
  }

  return td_signals;
}

}  // namespace impl

namespace windows {

template <class T>
auto hann(ssize_t length) {
  xt::xtensor<T, 1> window = xt::zeros<T>({length});
  T length_inv = 1. / length;
  for (ssize_t n = 0; n < length; n++)
    window(n) = 0.5 * (1. - cos(2 * M_PI * length_inv * n));
  return window;
}

template <class E>
auto make_dual(E&& awindow, ssize_t n_shift) {
  ssize_t length = awindow.shape()[0];

  using real_t = typename std::remove_pointer<decltype(awindow.data())>::type;

  // buffer to store the synthesis window
  xt::xtensor<real_t, 1> swindow = xt::zeros<real_t>({length});

  real_t norm[length];
  for (ssize_t m = 0; m < length; m++) norm[m] = 0.;

  int n = 0;

  // move the window back as far as possible while still overlapping
  while (n - n_shift > -length) n -= n_shift;

  while (n < length) {
    if (n == 0)
      for (ssize_t m = 0; m < length; m++) norm[m] += awindow(m) * awindow(m);

    else if (n < 0)
      for (ssize_t m = 0; m < n + length; m++)
        norm[m] += awindow(m - n) * awindow(m - n);

    else
      for (int m = n; m < length; m++)
        norm[m] += awindow(m - n) * awindow(m - n);

    n += n_shift;
  }

  for (int m = 0; m < length; m++) swindow(m) = awindow(m) / norm[m];

  return swindow;
}

}  // namespace windows

// C++ API
template <class T>
auto stft(xt::xtensor<T, 2>& td_signals, xt::xtensor<T, 1>& window,
          size_t n_shift, size_t n_zeropad_front, size_t n_zeropad_back) {
  return impl::stft(td_signals, window, n_shift, n_zeropad_front,
                    n_zeropad_back);
}

template <class T>
auto istft(xt::xtensor<std::complex<T>, 3>& stft_signals,
           xt::xtensor<T, 1>& window, size_t n_shift,
           size_t n_zeropad_front, size_t n_zeropad_back) {
  return impl::istft(stft_signals, window, n_shift, n_zeropad_front,
                     n_zeropad_back);
}

}  // namespace piva

#endif  // __STFT_H__
