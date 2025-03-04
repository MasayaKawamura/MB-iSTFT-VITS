# coding: utf-8

# BSD 3-Clause License
# Copyright (c) 2017, Prem Seetharaman
# Copyright (c) 2024, Musharraf Omer
# All rights reserved.
# * Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from librosa import util as librosa_util
from librosa.util import pad_center, tiny
from scipy.signal import get_window
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

# Calculate as: (seconds * sample_rate) / hop_size
# Common values (22.05KHz): 30 secs=~2600 mels, 60 secs=~5200 mels, 100 secs=~8620 mels
# This determines model size
MAX_FRAMES = 5200

DEFAULT_OPSET = 17
DEFAULT_SEED = 1234
_LOGGER = logging.getLogger("piper_train.export_onnx")


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, max_frames, filter_length, hop_length, win_length, window):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )
        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()
            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window
        window_sum = self.window_sumsquare(
            self.window,
            max_frames,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.filter_length,
            dtype=np.float32,
        )
        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())
        self.register_buffer("window_sum", torch.from_numpy(window_sum))
        self.tiny = tiny(window_sum)

    def forward(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )
        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )
        self.window = None
        if self.window is not None:
            win_dim = inverse_transform.size(-1)
            window_sum = self.window_sum[:win_dim]
            # remove modulation effects
            inverse_transform = inverse_transform.squeeze()
            approx_nonzero_indices = (window_sum > self.tiny).nonzero()
            inverse_transform[approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]
            inverse_transform = inverse_transform.unsqueeze(0).unsqueeze(1)
            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length
        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]
        return inverse_transform

    @staticmethod
    def window_sumsquare(
        window,
        n_frames,
        hop_length=200,
        win_length=800,
        n_fft=800,
        dtype=np.float32,
        norm=None,
    ):
        """
        # from librosa 0.6
        Compute the sum-square envelope of a window function at a given hop length.
        This is used to estimate modulation effects induced by windowing
        observations in short-time fourier transforms.
        Parameters
        ----------
        window : string, tuple, number, callable, or list-like
            Window specification, as in `get_window`
        n_frames : int > 0
            The number of analysis frames
        hop_length : int > 0
            The number of samples to advance between frames
        win_length : [optional]
            The length of the window function.  By default, this matches `n_fft`.
        n_fft : int > 0
            The length of each analysis frame.
        dtype : np.dtype
            The data type of the output
        Returns
        -------
        wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
            The sum-squared envelope of the window function
        """
        if win_length is None:
            win_length = n_fft

        n = n_fft + hop_length * (n_frames - 1)
        x = np.zeros(n, dtype=dtype)

        # Compute the squared window at the desired length
        win_sq = get_window(window, win_length, fftbins=True)
        win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
        win_sq = librosa_util.pad_center(win_sq, size=n_fft)

        # Fill the envelope
        for i in range(n_frames):
            sample = i * hop_length
            x[sample : min(n, sample + n_fft)] += win_sq[
                : max(0, min(n_fft, n - sample))
            ]
        return x


class ExportableISTFTModule(nn.Module):
    def __init__(
        self, max_frames, filter_length, hop_length, win_length, window="hann"
    ):
        super().__init__()
        self.stft = STFT(
            max_frames,
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )

    # def forward(self, mag, x, y):
    def forward(self, mag, phase):
        # phase = torch.atan2(y, x)
        return self.stft(mag, phase)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Export ISTFT to ONNX")

    parser.add_argument("output", type=str, help="Path to output `.onnx` file")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=MAX_FRAMES,
        help="Max number of frames decodable by this model",
    )
    parser.add_argument("--nfft", type=int, default=1024, help="filter-length")
    parser.add_argument("--hop", type=int, default=256, help="hop-length")
    parser.add_argument("--win", type=int, default=1024, help="win-length")
    parser.add_argument("--win-type", type=str, default="hann", help="window type")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version to use"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _LOGGER.info("Exporting ISTFT to ONNX")
    _LOGGER.info(f"Max frames  decodable by exported model: {args.max_frames}")
    _LOGGER.info(
        "ISTFT parameters:\n"
        f"\tfilter-length: {args.nfft}\n"
        f"\thop-length: {args.hop}\n"
        f"\twin-length: {args.win}\n"
        f"\twin-type: `{args.win_type}`"
    )
    istft_model = ExportableISTFTModule(
        args.max_frames,
        filter_length=args.nfft,
        hop_length=args.hop,
        win_length=args.win,
        window=args.win_type,
    )
    dynamic_axes = {
        "audio": {0: "batch", 2: "time"},
        "mag": {0: "Clipmag_dim_0", 1: "Clipmag_dim_1", 2: "Clipmag_dim_2"},
        "x": {0: "Clipmag_dim_0", 1: "Cosx_dim_1", 2: "Clipmag_dim_2"},
        "y": {0: "Clipmag_dim_0", 1: "Cosx_dim_1", 2: "Clipmag_dim_2"},
    }
    n_freqs = int(args.win / 2) + 1
    n_mels = 10
    assert (
        n_mels < MAX_FRAMES
    ), f"MAX_FRAMES `{MAX_FRAMES}` must be greater than `{n_mels}`"
    dummy_input = tuple(
        torch.rand((1, n_freqs, n_mels), dtype=torch.float32) for i in range(3)
    )
    torch.onnx.export(
        istft_model,
        args=dummy_input,
        f=args.output,
        input_names=["mag", "x", "y"],
        output_names=["audio"],
        dynamic_axes=dynamic_axes,
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
    )
    _LOGGER.info(f"Exported ONNX graph to {args.output}")


if __name__ == "__main__":
    main()