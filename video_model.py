import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
from calflops import calculate_flops
from compressai.layers import MaskedConv2d, conv3x3
from compressai.ops import ste_round
from pytorch_msssim import ms_ssim
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from flexible_modules import RDVCMC, RDVCRefine, \
    Feature_adaptor, RDVCFusion, RDVCRecNet, RDVCFeaExt


from modules import RDVCChannelSplitICIPWithGain

from optical_flow import backwarp, SpyNet, torch_warp, ME_Spynet
from compressai.entropy_models import EntropyBottleneck
import torch.nn as nn
import math
import torch
import numpy as np


class RDVC(nn.Module):
    def __init__(self, flexible=False):
        super().__init__()
        self.opticFlow = ME_Spynet()
        self.mv_codec = RDVCChannelSplitICIPWithGain(8, 64, 2, lower_bound=32, reduce=8, flexible=flexible)
        self.res_codec = RDVCChannelSplitICIPWithGain(64 + 6, 96, 64, lower_bound=48, reduce=8, flexible=flexible)
        self.MC = RDVCMC(hidden=64, lower_bound=32, reduce=8)

        self.RefineMvNet = RDVCRefine(5, 64, 2, lower_bound=32, reduce=8)
        self.RefineResiNet = RDVCRefine(64 + 3, 64, 64, lower_bound=32, reduce=8)

        self.FeatureExtractor = RDVCFeaExt(3, 64, 64, lower_bound=32, reduce=8)
        self.Fea_adapter = Feature_adaptor(64)
        self.FeatureFusion = RDVCFusion(128, 64, 64, lower_bound=32, reduce=8)
        self.enhance = RDVCRecNet(128, 64, 3, lower_bound=32, reduce=8)

    def forward(self, ref_frame, curr_frame, feature=None):
        pixels = np.prod(curr_frame.size()) // curr_frame.size()[1]

        feature = self.Fea_adapter(ref_frame, feature)
        # motion estimation
        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_enc_out = self.mv_codec(torch.cat([curr_frame, estimated_mv, ref_frame], 1), feature)
        recon_mv1 = mv_enc_out['x_hat']
        recon_mv = self.RefineMvNet(recon_mv1, ref_frame)

        # motion compensation
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_loss = torch.mean((warped_frame - curr_frame).pow(2))
        bpp_mv = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in mv_enc_out["likelihoods"].values()
        )
        # MC_input = torch.cat([ref_frame, warped_frame], dim=1)
        warp_fea, predict_frame = self.MC(warped_frame, recon_mv, feature)
        warp_fea = self.FeatureFusion(warp_fea, feature)
        mc_loss = torch.mean((predict_frame - curr_frame).pow(2))

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        curr_frame_fea = self.FeatureExtractor(curr_frame)

        res = curr_frame_fea - predict_frame_fea
        res_enc_out = self.res_codec(torch.cat([ref_frame, res, predict_frame], 1), warp_fea)
        recon_res1 = res_enc_out['x_hat']
        bpp_res = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in res_enc_out["likelihoods"].values()
        )
        recon_res = self.RefineResiNet(recon_res1, ref_frame)
        recon_image_fea = predict_frame_fea + recon_res

        feature, recon_image = self.enhance(recon_image_fea, warp_fea)
        mse_loss = torch.mean((recon_image - curr_frame).pow(2))
        bpp = bpp_mv + bpp_res

        return recon_image, feature, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp

    def forward_msssim(self, ref_frame, curr_frame, feature=None):
        pixels = np.prod(curr_frame.size()) // curr_frame.size()[1]

        feature = self.Fea_adapter(ref_frame, feature)
        # motion estimation
        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_enc_out = self.mv_codec(torch.cat([curr_frame, estimated_mv, ref_frame], 1), feature)
        recon_mv1 = mv_enc_out['x_hat']
        recon_mv = self.RefineMvNet(recon_mv1, ref_frame)

        # motion compensation
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_msssim = ms_ssim(warped_frame, curr_frame, 1.0)
        bpp_mv = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in mv_enc_out["likelihoods"].values()
        )
        # MC_input = torch.cat([ref_frame, warped_frame], dim=1)
        warp_fea, predict_frame = self.MC(warped_frame, recon_mv, feature)
        warp_fea = self.FeatureFusion(warp_fea, feature)
        mc_msssim = ms_ssim(predict_frame, curr_frame, 1.0)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        curr_frame_fea = self.FeatureExtractor(curr_frame)

        res = curr_frame_fea - predict_frame_fea
        res_enc_out = self.res_codec(torch.cat([ref_frame, res, predict_frame], 1), warp_fea)
        recon_res1 = res_enc_out['x_hat']
        bpp_res = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in res_enc_out["likelihoods"].values()
        )
        recon_res = self.RefineResiNet(recon_res1, ref_frame)
        recon_image_fea = predict_frame_fea + recon_res

        feature, recon_image = self.enhance(recon_image_fea, warp_fea)
        msssim = ms_ssim(recon_image, curr_frame, 1.0)
        bpp = bpp_mv + bpp_res

        return recon_image, feature, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp

    def compress(self, ref_frame, curr_frame,  feature=None):
        feature = self.Fea_adapter(ref_frame, feature)
        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_out_enc = self.mv_codec.compress(torch.cat([curr_frame, estimated_mv, ref_frame], 1), feature)
        recon_mv = self.mv_codec.decompress(mv_out_enc["strings"], mv_out_enc["shape"], feature)['x_hat']
        recon_mv = self.RefineMvNet(recon_mv, ref_frame)
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_fea, predict_frame = self.MC(warped_frame, recon_mv, feature)
        warp_fea = self.FeatureFusion(warp_fea, feature)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        curr_frame_fea = self.FeatureExtractor(curr_frame)
        res = curr_frame_fea - predict_frame_fea

        res_out_enc = self.res_codec.compress(torch.cat([ref_frame, res, predict_frame], 1), warp_fea)
        return mv_out_enc, res_out_enc

    def decompress(self, ref_frame, mv_out_enc, res_out_enc, feature=None):
        feature = self.Fea_adapter(ref_frame, feature)
        recon_mv = self.mv_codec.decompress(mv_out_enc["strings"], mv_out_enc["shape"], feature)['x_hat']
        recon_mv = self.RefineMvNet(recon_mv, ref_frame)
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_fea, predict_frame = self.MC(warped_frame, recon_mv, feature)
        warp_fea = self.FeatureFusion(warp_fea, feature)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        recon_res = self.res_codec.decompress(res_out_enc["strings"], res_out_enc["shape"], warp_fea)['x_hat']
        recon_res = self.RefineResiNet(recon_res, ref_frame)

        recon_image_fea = predict_frame_fea + recon_res
        feature, recon_image = self.enhance(recon_image_fea, warp_fea)

        return feature, recon_image.clamp(0., 1.), warped_frame.clamp(0., 1.), predict_frame.clamp(0., 1.)

    def set_rate_level(self, index, interpolation_coefficient=1.0, isInterpolation=False):
        self.mv_codec.set_rate_level(index=index, interpolation_coefficient=interpolation_coefficient,
                                     isInterpolation=isInterpolation)
        self.res_codec.set_rate_level(index=index, interpolation_coefficient=interpolation_coefficient,
                                     isInterpolation=isInterpolation)

    def get_active_subnet_settings(self):
        mv_width = self.mv_codec.get_active_subnet_settings()
        res_width = self.res_codec.get_active_subnet_settings()
        mc_width = self.MC.get_active_subnet_settings()

        refinemv_width = self.RefineMvNet.get_active_subnet_settings()
        refineres_width = self.RefineResiNet.get_active_subnet_settings()

        feaExt_width = self.FeatureExtractor.get_active_subnet_settings()
        fusionfea_width = self.FeatureFusion.get_active_subnet_settings()
        enhance_width = self.enhance.get_active_subnet_settings()

        return {"mv_decoder_settings": mv_width,
                "res_decoder_settings": res_width,
                "mc_width": mc_width,
                "refinemv_width": refinemv_width,
                "refineres_width": refineres_width,
                "feaExt_width": feaExt_width,
                "fusionfea_width": fusionfea_width,
                "enhance_width": enhance_width}

    def sample_active_subnet(self, mv_mode='largest', res_mode='largest', mc_mode='largest', refinemv_mode='largest',refineres_mode='largest', FeaExt_mode='largest', fusionfea_mode='largest',
                              enhance_mode='largest'):
        cfg_mv = self.mv_codec.sample_active_subnet(mv_mode)
        cfg_res = self.res_codec.sample_active_subnet(res_mode)
        cfg_mc = self.MC.sample_active_subnet(mc_mode)

        cfg_refinemv = self.RefineMvNet.sample_active_subnet(refinemv_mode)
        cfg_refineres = self.RefineResiNet.sample_active_subnet(refineres_mode)
        cfg_FeaExt = self.FeatureExtractor.sample_active_subnet(FeaExt_mode)
        cfg_fusionfea = self.FeatureFusion.sample_active_subnet(fusionfea_mode)
        cfg_enhance = self.enhance.sample_active_subnet(enhance_mode)

        return {"cfg_mv": cfg_mv,
                "cfg_res": cfg_res,
                "cfg_mc": cfg_mc,
                "cfg_refinemv": cfg_refinemv,
                "cfg_refineres": cfg_refineres,
                "cfg_FeaExt": cfg_FeaExt,
                "cfg_fusionfea": cfg_fusionfea,
                "cfg_enhance": cfg_enhance}

    def set_active_subnet(self, all_cfg):
        cfg_mv = all_cfg['cfg_mv']
        cfg_res = all_cfg['cfg_res']
        cfg_mc = all_cfg['cfg_mc']
        cfg_refinemv = all_cfg['cfg_refinemv']
        cfg_refineres = all_cfg['cfg_refineres']
        cfg_FeaExt = all_cfg['cfg_FeaExt']
        cfg_fusionfea = all_cfg['cfg_fusionfea']
        cfg_enhance = all_cfg['cfg_enhance']

        self.mv_codec.set_active_subnet(cfg_mv)
        self.res_codec.set_active_subnet(cfg_res)
        self.MC.set_active_subnet(cfg_mc)
        self.RefineMvNet.set_active_subnet(cfg_refinemv)
        self.RefineResiNet.set_active_subnet(cfg_refineres)
        self.FeatureExtractor.set_active_subnet(cfg_FeaExt)
        self.FeatureFusion.set_active_subnet(cfg_fusionfea)
        self.enhance.set_active_subnet(cfg_enhance)

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def mv_aux_loss(self):
        return sum(m.loss() for m in self.mv_codec.modules() if isinstance(m, EntropyBottleneck))

    def res_aux_loss(self):
        return sum(m.loss() for m in self.res_codec.modules() if isinstance(m, EntropyBottleneck))

    def update(self, force=False):
        updated = self.mv_codec.update(force=force)
        updated |= self.res_codec.update(force=force)
        return updated

    def load_state_dict(self, state_dict):
        mv_codec_dict = {k[len('mv_codec.'):]: v for k, v in state_dict.items() if 'mv_codec' in k}
        res_codec_dict = {k[len('res_codec.'):]: v for k, v in state_dict.items() if 'res_codec' in k}

        self.mv_codec.load_state_dict(mv_codec_dict)
        self.res_codec.load_state_dict(res_codec_dict)

        super().load_state_dict(state_dict)

    def mv_aux_params(self):
        params = []
        names = []
        for k, v in self.mv_codec.named_parameters():
            if 'quantiles' not in k:
                continue
            params.append(v)
            names.append(k)
        return names, params

    def res_aux_params(self):
        params = []
        names = []
        for k, v in self.res_codec.named_parameters():
            if 'quantiles' not in k:
                continue
            params.append(v)
            names.append(k)
        return names, params

    def aux_params(self):
        params = []
        names = []
        for k, v in self.res_codec.named_parameters():
            if 'quantiles' not in k:
                continue
            params.append(v)
            names.append(k)
        for k, v in self.mv_codec.named_parameters():
            if 'quantiles' not in k:
                continue
            params.append(v)
            names.append(k)
        return names, params



if __name__ == "__main__":

    model = RDVC(True)
    model.set_rate_level(1)
    print(f'[*] Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')














