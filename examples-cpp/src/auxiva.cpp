/*
   Utility that separates a multi-channel audio file
   using AuxIVA-IP.
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
#include <cassert>
#include <iostream>
#include <string>

// xtensor stuff
#include <xtensor-io/xaudio.hpp>
#include <xtensor/xview.hpp>

// piva stuff
#include <piva/auxiva.hpp>
#include <piva/stft.hpp>
#include <piva/utils.hpp>

using std::string;
using xt::all;

int main(int argc, char** argv) {
  // Parse arguments
  assert(argc == 6);
  string input_filename(argv[1]);
  string output_filename(argv[2]);
  size_t n_iter = std::stoi(argv[3]);
  size_t n_analysis = std::stoi(argv[4]);
  size_t n_shift = std::stoi(argv[5]);

  // create the windows for the STFT
  std::cout << "Create the Windows" << std::endl;
  auto analysis_window = piva::windows::hann<double>(n_analysis);
  auto synthesis_window = piva::windows::make_dual(analysis_window, n_shift);

  // open a wav file
  std::cout << "Open the audio file" << std::endl;
  auto audio =
      piva::load<double>(input_filename, true);  // load and scale audio
  auto sampling_freq = std::get<0>(audio);            // sampling frequency
  xt::xtensor<double, 2> audio_arr = std::get<1>(audio);  // audio contents

  // Go to time-frequency domain
  std::cout << "Do the STFT" << std::endl;
  auto stft_data = piva::stft(audio_arr, analysis_window, n_shift, 0, 0);

  // Perform the separation
  std::cout << "Do the separation" << std::endl;
  auto bss_output = piva::auxiva(stft_data, n_iter);

  // projection back (in-place)
  std::cout << "Project back" << std::endl;
  piva::project_back(bss_output, xt::view(stft_data, all(), 0, all()));

  // Go back to time domain
  std::cout << "Do the iSTFT" << std::endl;
  auto y = piva::istft(bss_output, synthesis_window, n_shift, 0, 0);

  // Save output to file
  std::cout << "Save to file" << std::endl;
  piva::dump(output_filename, y, sampling_freq, true);  // scale and dump
}
