import random

import torch.nn as nn
import math
import torch
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from compressai.models import get_scale_table
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ops import ste_round
from matplotlib import pyplot as plt

from torch.autograd import Function

from flexible_modules import Gain_Module,RDVC_ga, RDVC_gs, RDVC_ha, RDVC_hs, RDVC_param, RDVC_fea

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None

class RDVCChannelSplitICIPWithGain(nn.Module):
    def __init__(self, in_ch=3, N=192, out_ch=3, rate_point=5, lower_bound=32, reduce=8, flexible=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.N = int(N)
        self.num_slices = 8
        self.max_support_slices = 4
        self.flexible = flexible

        N_channel_list = list(range(lower_bound, N + 1, reduce))
        print("entropy_channel_num: ", len(N_channel_list), " channel : ", N_channel_list)

        slice_depth = self.N // self.num_slices
        if slice_depth * self.num_slices != self.N:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.N}/{self.num_slices})")

        self.list_len = len(N_channel_list)

        # gain module
        if self.flexible:
            self.gain_unit = Gain_Module(n=rate_point, N=N)
            self.inv_gain_unit = Gain_Module(n=rate_point, N=N)
            self.prior_gain_unit = Gain_Module(n=rate_point, N=N)
            self.prior_inv_gain_unit = Gain_Module(n=rate_point, N=N)

            self.rate_level, self.rate_interpolation_coefficient = None, None
            self.prior_rate_level, self.prior_rate_interpolation_coefficient = None, None
            self.isInterpolation = False

        # mian module
        self.g_a = RDVC_ga(in_ch=in_ch, N=N, out_ch=N)

        self.g_s = RDVC_gs(in_ch=N, N=N,  out_ch=out_ch, N_channel_list=N_channel_list)

        self.h_a = RDVC_ha(in_ch=N, N=N,  out_ch=N, N_channel_list=N_channel_list)

        self.h_s = RDVC_hs(in_ch=N, N=N, out_ch=N, N_channel_list=N_channel_list)

        self.param = RDVC_param(in_ch=N*2, N=N, out_ch=N*3, N_channel_list=N_channel_list)

        self.fea = RDVC_fea(in_ch=64, N=N, out_ch=N, N_channel_list=N_channel_list)


        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i, self.max_support_slices), N, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
                nn.GELU(),
                conv(32, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i, self.max_support_slices), N, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
                nn.GELU(),
                conv(32, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i + 1, self.max_support_slices + 1), N, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N, N // 2, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N // 2, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

        self.cfg_candidates = {
            'g_a': {
                'layer_num': self.g_a.layer_num,
                'c': self.g_a.width_list},
            'g_s': {
                'layer_num': self.g_s.layer_num,
                'c': self.g_s.width_list},
            'h_a': {
                'layer_num': self.h_a.layer_num,
                'c': self.h_a.width_list},
            'h_s': {
                'layer_num': self.h_s.layer_num,
                'c': self.h_s.width_list},
            'mean_scale_factor': {
                'layer_num': self.param.layer_num,
                'c': self.param.width_list},
            'fea': {
                'layer_num': self.fea.layer_num,
                'c': self.fea.width_list},
        }

    def forward(self, x, feature):
        if self.flexible:
            y = self.g_a(x)
            y_gained = self.gain_unit(y, self.rate_level, self.rate_interpolation_coefficient)
            z = self.h_a(y_gained)
            z_gained = self.prior_gain_unit(z, self.rate_level, self.rate_interpolation_coefficient)
            y_shape = y.shape[2:]
            _, z_likelihoods = self.entropy_bottleneck(z_gained)

            # Use rounding (instead of uniform noise) to modify z before passing it
            # to the hyper-synthesis transforms. Note that quantize() overrides the
            # gradient to create a straight-through estimator.
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset
            z_hat_inv_gained = self.prior_inv_gain_unit(z_hat, self.rate_level,
                                                        self.rate_interpolation_coefficient)

            fea = self.fea(feature)
            param = self.h_s(z_hat_inv_gained)
            latent = self.param(torch.concat([param, fea], dim=1))

            q_step, latent_scales, latent_means = latent.chunk(3, 1)
            quant_step = LowerBound.apply(q_step, 0.5)

            y_gained = y_gained / quant_step
            y_slices = y_gained.chunk(self.num_slices, 1)
            y_hat_slices = []
            y_likelihood = []

            for slice_index, y_slice in enumerate(y_slices):
                support_slices = (
                    y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
                mean_support = torch.cat([latent_means] + support_slices, dim=1)
                mu = self.cc_mean_transforms[slice_index](mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                scale_support = torch.cat([latent_scales] + support_slices, dim=1)
                scale = self.cc_scale_transforms[slice_index](scale_support)
                scale = scale[:, :, :y_shape[0], :y_shape[1]]

                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
                y_likelihood.append(y_slice_likelihood)
                y_hat_slice = ste_round(y_slice - mu) + mu

                lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice += lrp
                y_hat_slices.append(y_hat_slice)

            y_hat = torch.cat(y_hat_slices, dim=1)
            y_hat = y_hat * quant_step
            y_likelihoods = torch.cat(y_likelihood, dim=1)
            y_hat_inv_gained = self.inv_gain_unit(y_hat, self.rate_level, self.rate_interpolation_coefficient)
            x_hat = self.g_s(y_hat_inv_gained)
        else:
            y = self.g_a(x)
            z = self.h_a(y)
            y_shape = y.shape[2:]
            _, z_likelihoods = self.entropy_bottleneck(z)

            # Use rounding (instead of uniform noise) to modify z before passing it
            # to the hyper-synthesis transforms. Note that quantize() overrides the
            # gradient to create a straight-through estimator.
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

            fea = self.fea(feature)
            param = self.h_s(z_hat)
            latent = self.param(torch.concat([param, fea], dim=1))

            q_step, latent_scales, latent_means = latent.chunk(3, 1)
            quant_step = LowerBound.apply(q_step, 0.5)

            y = y / quant_step
            y_slices = y.chunk(self.num_slices, 1)
            y_hat_slices = []
            y_likelihood = []

            for slice_index, y_slice in enumerate(y_slices):
                support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
                mean_support = torch.cat([latent_means] + support_slices, dim=1)
                mu = self.cc_mean_transforms[slice_index](mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                scale_support = torch.cat([latent_scales] + support_slices, dim=1)
                scale = self.cc_scale_transforms[slice_index](scale_support)
                scale = scale[:, :, :y_shape[0], :y_shape[1]]

                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
                y_likelihood.append(y_slice_likelihood)
                y_hat_slice = ste_round(y_slice - mu) + mu

                lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice += lrp
                y_hat_slices.append(y_hat_slice)

            y_hat = torch.cat(y_hat_slices, dim=1)
            y_hat = y_hat * quant_step
            y_likelihoods = torch.cat(y_likelihood, dim=1)
            x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, feature):
        if self.flexible:
            y = self.g_a(x)
            y_gained = self.gain_unit(y, self.rate_level, self.rate_interpolation_coefficient, self.isInterpolation)
            z = self.h_a(y_gained)
            z_gained = self.prior_gain_unit(z, self.rate_level, self.rate_interpolation_coefficient,
                                            self.isInterpolation)
            y_shape = y_gained.shape[2:]

            z_strings = self.entropy_bottleneck.compress(z_gained)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

            z_hat_inv_gained = self.prior_inv_gain_unit(z_hat, self.rate_level,
                                                        self.rate_interpolation_coefficient, self.isInterpolation)

            fea = self.fea(feature)
            param = self.h_s(z_hat_inv_gained)
            latent = self.param(torch.concat([param, fea], dim=1))

            q_step, latent_scales, latent_means = latent.chunk(3, 1)
            # # 可视化 q_step
            # if index==1:
            #     plt.figure(figsize=(448/16, 832/16))
            #     for i in range(1):
            #         plt.subplot(1, 1, i + 1)
            #         plt.imshow(q_step[0, i, :, :].detach().cpu().numpy(), cmap='gray')  # 将 q_step 的第一个样本的第 i 个通道可视化
            #         plt.axis('off')
            #         plt.savefig("./largest.png")
            #     plt.show()
            #     exit()


            quant_step = LowerBound.apply(q_step, 0.5)

            y_gained = y_gained / quant_step

            y_slices = y_gained.chunk(self.num_slices, 1)
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

        else:
            y = self.g_a(x)
            z = self.h_a(y)
            y_shape = y.shape[2:]

            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

            fea = self.fea(feature)
            param = self.h_s(z_hat)
            latent = self.param(torch.concat([param, fea], dim=1))

            q_step, latent_scales, latent_means = latent.chunk(3, 1)
            quant_step = LowerBound.apply(q_step, 0.5)

            y = y / quant_step

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

    def decompress(self, strings, shape, feature):
        if self.flexible:
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
            z_hat_inv_gained = self.prior_inv_gain_unit(z_hat, self.rate_level,
                                                        self.rate_interpolation_coefficient, self.isInterpolation)
            fea = self.fea(feature)
            param = self.h_s(z_hat_inv_gained)
            latent = self.param(torch.concat([param, fea], dim=1))

            q_step, latent_scales, latent_means = latent.chunk(3, 1)
            quant_step = torch.clamp_min(q_step, 0.5)

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
            y_hat = y_hat * quant_step
            y_hat_inv_gained = self.inv_gain_unit(y_hat, self.rate_level, self.rate_interpolation_coefficient,
                                                  self.isInterpolation)
            x_hat = self.g_s(y_hat_inv_gained)
        else:
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
            fea = self.fea(feature)
            param = self.h_s(z_hat)
            latent = self.param(torch.concat([param, fea], dim=1))

            q_step, latent_scales, latent_means = latent.chunk(3, 1)
            quant_step = torch.clamp_min(q_step, 0.5)

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
            y_hat = y_hat * quant_step
            x_hat = self.g_s(y_hat)
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

    def load_state_dict(self, state_dict, pretrain=False):
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
        if pretrain:
            super().load_state_dict(state_dict, strict=False)
        else:
            super().load_state_dict(state_dict)

    def set_rate_level(self, index, interpolation_coefficient=1.0, isInterpolation=False):
        self.rate_level = index
        self.prior_rate_level = index
        self.rate_interpolation_coefficient = interpolation_coefficient
        self.prior_rate_interpolation_coefficient = interpolation_coefficient
        self.isInterpolation = isInterpolation

    def set_active_subnet(self, cfg):
        for layer_index, layer in enumerate(self.g_s.g_s):
            layer.active_out_channel = cfg['g_s'][layer_index]
            layer.set_active_channels()

        for layer_index, layer in enumerate(self.h_s.h_s):
            layer.active_out_channel = cfg['h_s'][layer_index]
            layer.set_active_channels()

        for layer_index, layer in enumerate(self.param.param):
            layer.active_out_channel = cfg['mean_scale_factor'][layer_index]
            layer.set_active_channels()

        for layer_index, layer in enumerate(self.fea.fea_enc):
            layer.active_out_channel = cfg['fea'][layer_index]
            layer.set_active_channels()

    def get_active_subnet_settings(self):
        width = {}
        for k in ['g_a', 'h_a', 'g_s', 'h_s', 'mean_scale_factor', 'fea']:
            width[k] = []

        for layer_index, layer in enumerate(self.g_s.g_s):
            width['g_s'].append(layer.active_out_channel)

        for layer_index, layer in enumerate(self.h_s.h_s):
            width['h_s'].append(layer.active_out_channel)

        for layer_index, layer in enumerate(self.param.param):
            width['mean_scale_factor'].append(layer.active_out_channel)

        for layer_index, layer in enumerate(self.fea.fea_enc):
            width['fea'].append(layer.active_out_channel)
        return {"width": width}

    def sample_active_subnet(self, mode='largest', uniform_index=None):
        assert mode in ['largest', 'random', 'smallest', 'uniform']
        if mode == 'random':
            cfg = self._sample_active_subnet(min_net=False, max_net=False)
        elif mode == 'largest':
            cfg = self._sample_active_subnet(max_net=True)
        elif mode == 'smallest':
            cfg = self._sample_active_subnet(min_net=True)
        elif mode == 'uniform':
            cfg = self._sample_active_subnet(uniform=True, uniform_index=uniform_index)
        return cfg

    def sample_active_subnet_within_range(self,image_size, targeted_flops):
        for i in range(self.list_len):
            cfg = self._sample_active_subnet(uniform=True, uniform_index=i)
            cfg['flops'] = self.compute_active_subnet_flops(image_size, g_s=True, h_s=True, bias=True,
                                                            context_model=True) / 1e8
            if cfg['flops'] >= targeted_flops:
                if i == 0:
                    return cfg  # 如果i=0，则返回最佳配置
                else:
                    return self._sample_active_subnet(uniform=True, uniform_index=i - 1)  # 返回i-1的配置
        return self._sample_active_subnet(uniform=True, uniform_index=self.list_len - 1)

    def _sample_active_subnet(self, min_net=False, max_net=False, uniform=False, uniform_index=None):
        sample_cfg = lambda candidates, sample_min, sample_max: \
            min(candidates) if sample_min else (max(candidates) if sample_max else random.choice(candidates))
        cfg = {}
        if not uniform:
            for k in ['g_a', 'h_a', 'g_s', 'h_s', 'mean_scale_factor', 'fea']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):
                    cfg[k].append(sample_cfg(self.cfg_candidates[k]["c"][layer_index], min_net, max_net))
        else:
            for k in ['g_a', 'h_a']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):
                    cfg[k].append(self.cfg_candidates[k]["c"][layer_index][-1])
            selected = uniform_index if uniform_index is not None else -1
            for k in ['g_s', 'h_mean_s', 'h_scale_s']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):
                    if len(self.cfg_candidates[k]["c"][layer_index]) != 1 and len(self.cfg_candidates[k]["c"][layer_index]) > selected:
                        cfg[k].append(self.cfg_candidates[k]["c"][layer_index][selected])
                    else:
                        cfg[k].append(self.cfg_candidates[k]["c"][layer_index][-1])
        self.set_active_subnet(cfg)
        return cfg
