# -*- coding: utf-8 -*-
import os
import warnings
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import MaskedConv2d, GDN
from compressai.layers import subpel_conv3x3, ResidualBlockUpsample, ResidualBlockWithStride, ResidualBlock, conv3x3
from compressai.models.utils import conv, deconv, update_registered_buffers
import torch.nn.functional as F
import torch.nn as nn
import math
import torch
from compressai.ops import ste_round

# from modules import GDN  # different from compressai
from compressai.ans import BufferedRansEncoder, RansDecoder

from flexible_modules import ResBottleneckBlock, GainModule, GainModule0

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


# ICLR2018 Variational Image Compression with a Scale Hyperprior
class FactorizedPrior(nn.Module):
    def __init__(self, in_ch=3, N=128, M=128, out_ch=3):
        super().__init__()
        self.N = N
        self.M = M
        self.entropy_bottleneck = EntropyBottleneck(channels=N)

        self.g_a = nn.Sequential(
            conv(in_ch, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )
        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, out_ch),
        )

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        return x_hat, y_likelihoods

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return y_strings, y.size()[-2:]

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return x_hat

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def update(self, force=False):
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


# ICLR2018 Variational Image Compression with a Scale Hyperprior
class ScaleHyperprior(nn.Module):
    def __init__(self, in_ch=3, N=128, M=128, out_ch=3):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(channels=N)

        self.g_a = nn.Sequential(
            conv(in_ch, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, out_ch),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return x_hat, y_likelihoods, z_likelihoods

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return y_strings, z_strings, z.size()[-2:]

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return x_hat


# 2018NeurIPS Joint Autoregressive and Hierarchical Priors for Learned Image Compression
class MeanScaleHyperprior(nn.Module):
    def __init__(self, in_ch=3, N=128, M=128, out_ch=3):
        super().__init__()
        self.N = int(N)
        self.M = int(M)
        self.entropy_bottleneck = EntropyBottleneck(channels=M)
        self.gaussian_conditional = GaussianConditional(None)

        self.g_a = nn.Sequential(
            conv(in_ch, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, out_ch),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)  # [b, 23, w, h] -> [b, 128, w//16, h//16]
        z = self.h_a(y)  # [b, 128, w//16, h//16] -> [b, 128, w//64, h//64]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)  # [b, 128, w//64, h//64]
        gaussian_params = self.h_s(z_hat)  # [b, 128, w//64, h//64] -> [b, 256, w//16, h//16]
        scales_hat, means_hat = gaussian_params.chunk(2, 1)  # [b, 128, w//16, h//16] * 2
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)  # [b, 128, w//16, h//16] * 2
        x_hat = self.g_s(y_hat)  # [b, 128, w//16, h//16] -> [b, 3, w, h]

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


# 2018NeurIPS Joint Autoregressive and Hierarchical Priors for Learned Image Compression
class JointAutoregressivePriors(nn.Module):
    def __init__(self, in_ch=3, N=128, M=128, out_ch=3):
        super().__init__()

        self.N = int(N)
        self.M = int(M)
        self.entropy_bottleneck = EntropyBottleneck(channels=M)
        self.gaussian_conditional = GaussianConditional(None)

        self.g_a = nn.Sequential(
            conv(in_ch, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, out_ch, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(M, 2 * M, kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return x_hat, y_likelihoods, z_likelihoods

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return y_strings, z_strings, z.size()[-2:]

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return x_hat

    def _decompress_ar(self, y_string, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp: hp + 1, wp: wp + 1] = rv

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class Cheng2020Anchor(nn.Module):
    def __init__(self, in_ch=3, N=128, M=128, out_ch=3):
        super().__init__()

        self.N = int(N)
        self.M = int(M)
        self.entropy_bottleneck = EntropyBottleneck(channels=M)
        self.gaussian_conditional = GaussianConditional(None)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(in_ch, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, out_ch, 2),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(M, 2 * M, kernel_size=5, padding=2, stride=1)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(self, y_string, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp: hp + 1, wp: wp + 1] = rv

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class ICIP2020ResB(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 10
        self.max_support_slices = 5

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        self.g_a = nn.Sequential(
            conv(3, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_mean_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.h_scale_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        # self.slice_transform = nn.Sequential(
        #     conv(M, 224),
        #     nn.LeakyReLU(inplace=True),
        #     conv(224, 128),
        #     nn.LeakyReLU(inplace=True),
        #     conv(128, slice_depth, stride=1, kernel_size=3)
        # )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class ICIP2020ResBVB1(nn.Module):
    def __init__(self, N=192, M=320, v0=False, x5=False, psnr=True):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 10
        self.max_support_slices = 5
        if psnr:
            self.lmbda = [0.0808, 0.0483, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]
        else:
            self.lmbda = [187.49, 115.37, 60.50, 31.73, 16.64, 8.73, 4.58, 2.40]
        # self.lmbda = [0.065, 0.0325, 0.01625, 0.008125, 0.004062, 0.00203125]
        self.levels = len(self.lmbda)  # 8

        if not v0:
            self.gain_unit = GainModule(n=self.levels, N=M)
            self.inv_gain_unit = GainModule(n=self.levels, N=M)
            self.hyper_gain_unit = GainModule(n=self.levels, N=N)
            self.hyper_inv_gain_unit = GainModule(n=self.levels, N=N)
        else:
            self.gain_unit = GainModule0(n=self.levels, N=M)
            self.inv_gain_unit = GainModule0(n=self.levels, N=M)
            self.hyper_gain_unit = GainModule0(n=self.levels, N=N)
            self.hyper_inv_gain_unit = GainModule0(n=self.levels, N=N)

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        if not x5:
            self.g_a = nn.Sequential(
                conv(3, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                conv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                conv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                conv(N, M),
            )

            self.g_s = nn.Sequential(
                deconv(M, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                deconv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                deconv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                deconv(N, 3),
            )
        else:
            self.g_a = nn.Sequential(
                conv(3, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                conv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                conv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                conv(N, M),
            )

            self.g_s = nn.Sequential(
                deconv(M, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                deconv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                deconv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                deconv(N, 3),
            )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_mean_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.h_scale_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        # self.slice_transform = nn.Sequential(
        #     conv(M, 224),
        #     nn.LeakyReLU(inplace=True),
        #     conv(224, 128),
        #     nn.LeakyReLU(inplace=True),
        #     conv(128, slice_depth, stride=1, kernel_size=3)
        # )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, n=None, l=1):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        y = self.gain_unit(y, n, l)
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        z = self.hyper_gain_unit(z, n, l)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = self.inv_gain_unit(y_hat, n, l)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, n=None, l=1):
        y = self.g_a(x)
        y = self.gain_unit(y, n, l)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z = self.hyper_gain_unit(z, n, l)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, n=None, l=1):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

if __name__ == "__main__":
    h, w = 256, 256
    model = JointAutoregressivePriors().cuda()
    print(f'Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    x = torch.rand((1, 3, h, w))
    out = model(x.cuda())
    exit()
