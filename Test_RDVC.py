# -*- coding: utf-8 -*-
import inspect
import os
import torchvision

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import glob
from image_model import ICIP2020ResBVB1, ICIP2020ResB
from utils import AverageMeter, load_pretrained
import torch
import time
import numpy as np
from pytorch_msssim import ms_ssim
from video_model import RDVC
from utils import read_image, cal_psnr
import json


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)

#Test datasets

TEST_DATA_bpg = {
    'HEVC_B': {
        'path': '',
        'frames': 100,
        'gop': 10,
        'org_resolution': '1920x1080',
        'x16_resolution': '1920x1072',
        'x64_resolution': '1920x1024',
        'sequences': [
            'BasketballDrive_1920x1080_50',
            'BQTerrace_1920x1080_60',
            'Cactus_1920x1080_50',
            'Kimono1_1920x1080_24',
            'ParkScene_1920x1080_24',
        ],
    },

    'HEVC_C': {
        'path': '',
        'frames': 100,
        'gop': 10,
        'org_resolution': '832x480',
        'x16_resolution': '832x480',
        'x64_resolution': '832x448',
        'sequences': [
            'BasketballDrill_832x480_50',
            'BQMall_832x480_60',
            'PartyScene_832x480_50',
            'RaceHorses_832x480_30',
        ],
    },

    'HEVC_D': {
        'path': '',
        'frames': 100,
        'gop': 10,
        'org_resolution': '416x240',
        'x16_resolution': '416x240',
        'x64_resolution': '384x192',
        'sequences': [
            'BasketballPass_416x240_50',
            'BlowingBubbles_416x240_50',
            'BQSquare_416x240_60',
            'RaceHorses_416x240_30',
        ],
    },

    'HEVC_E': {
        'path': '',
        'frames': 100,
        'gop': 10,
        'org_resolution': '1280x720',
        'x16_resolution': '1280x720',
        'x64_resolution': '1280x704',
        'sequences': [
            'FourPeople_1280x720_60',
            'Johnny_1280x720_60',
            'KristenAndSara_1280x720_60',
        ],
    },

    'UVG': {
        'path': '',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x16_resolution': '1920x1072',
        'x64_resolution': '1920x1024',
        'sequences': [
            'Beauty_1920x1080_120fps_420_8bit_YUV',
            'Bosphorus_1920x1080_120fps_420_8bit_YUV',
            'HoneyBee_1920x1080_120fps_420_8bit_YUV',
            'Jockey_1920x1080_120fps_420_8bit_YUV',
            'ReadySteadyGo_1920x1080_120fps_420_8bit_YUV',
            'ShakeNDry_1920x1080_120fps_420_8bit_YUV',
            'YachtRide_1920x1080_120fps_420_8bit_YUV',
        ],
    },
}

TEST_DATA = {
    'HEVC_B': {
        'path': '',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',
        'sequences': {
            'BasketballDrive_1920x1080_50',
            'BQTerrace_1920x1080_60',
            'Cactus_1920x1080_50',
            'Kimono1_1920x1080_24',
            'ParkScene_1920x1080_24',
        },
    },

    'HEVC_C': {
        'path': '',
        'frames': 96,
        'gop': 12,
        'org_resolution': '832x480',
        'x64_resolution': '832x448',
        'sequences': [
            'BasketballDrill_832x480_50',
            'BQMall_832x480_60',
            'PartyScene_832x480_50',
            'RaceHorses_832x480_30',
        ],
    },

    'HEVC_D': {
        'path': '/tdx/databak/EXT4/LHB/data/TestSets/ClassD/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '416x240',
        'x64_resolution': '384x192',
        'sequences': [
            'BasketballPass_416x240_50',
            'BlowingBubbles_416x240_50',
            'BQSquare_416x240_60',
            'RaceHorses_416x240_30',
        ],
    },

    'HEVC_E': {
        'path': '',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1280x720',
        'x64_resolution': '1280x704',
        'sequences': [
            'FourPeople_1280x720_60',
            'Johnny_1280x720_60',
            'KristenAndSara_1280x720_60',
        ],
    },

    'UVG': {
        'path': '',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',
        'sequences': [
            'Beauty_1920x1080_120fps_420_8bit_YUV',
            'Bosphorus_1920x1080_120fps_420_8bit_YUV',
            'HoneyBee_1920x1080_120fps_420_8bit_YUV',
            'Jockey_1920x1080_120fps_420_8bit_YUV',
            'ReadySteadyGo_1920x1080_120fps_420_8bit_YUV',
            'ShakeNDry_1920x1080_120fps_420_8bit_YUV',
            'YachtRide_1920x1080_120fps_420_8bit_YUV',
        ],
    },

    "MCL-JCV": {
        "path": '',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',  # 18,20,24,25
        "sequences": [
            "videoSRC01_1920x1080_30",
            "videoSRC02_1920x1080_30",
            "videoSRC03_1920x1080_30",
            "videoSRC04_1920x1080_30",
            "videoSRC05_1920x1080_25",
            "videoSRC06_1920x1080_25",
            "videoSRC07_1920x1080_25",
            "videoSRC08_1920x1080_25",
            "videoSRC09_1920x1080_25",
            "videoSRC10_1920x1080_30",
            "videoSRC11_1920x1080_30",
            "videoSRC12_1920x1080_30",
            "videoSRC13_1920x1080_30",
            "videoSRC14_1920x1080_30",
            "videoSRC15_1920x1080_30",
            "videoSRC16_1920x1080_30",
            "videoSRC17_1920x1080_24",
            "videoSRC18_1920x1080_25",
            "videoSRC19_1920x1080_30",
            "videoSRC20_1920x1080_25",
            "videoSRC21_1920x1080_24",
            "videoSRC22_1920x1080_24",
            "videoSRC23_1920x1080_24",
            "videoSRC24_1920x1080_24",
            "videoSRC25_1920x1080_24",
            "videoSRC26_1920x1080_30",
            "videoSRC27_1920x1080_30",
            "videoSRC28_1920x1080_30",
            "videoSRC29_1920x1080_24",
            "videoSRC30_1920x1080_30",
        ]
    },

    'VTL': {
        'path': '',
        'frames': 96,
        'gop': 12,
        'org_resolution': '352x288',
        'x64_resolution': '320x256',
        'sequences': [
            'akiyo_cif',
            'BigBuckBunny_CIF_24fps',
            'bridge-close_cif',
            'bridge-far_cif',
            'bus_cif',
            'coastguard_cif',
            'container_cif',
            'ElephantsDream_CIF_24fps',
            'flower_cif',
            'foreman_cif',
            'hall_cif',
            'highway_cif',
            'mobile_cif',
            'mother-daughter_cif',
            'news_cif',
            'paris_cif',
            'silent_cif',
            'stefan_cif',
            'tempete_cif',
            'waterfall_cif',
        ],
    },

}

def get_argument_count(func):
    args = inspect.getfullargspec(func).args
    return len(args)

def get_quality(l_PSNR=0, l_MSSSIM=0):
    QP, I_level = 0, 0
    if l_PSNR == 256:
        QP = 37
    elif l_PSNR == 512:
        QP = 32
    elif l_PSNR == 1024:
        QP = 27
    elif l_PSNR == 2048:
        QP = 22
    elif l_PSNR == 4096:
        QP = 17

    if l_MSSSIM == 8:
        I_level = 2
    elif l_MSSSIM == 16:
        I_level = 3
    elif l_MSSSIM == 32:
        I_level = 5
    elif l_MSSSIM == 64:
        I_level = 7
    return QP, I_level

def get_results_single_point(model_mode="largest", mode="PSNR", test_tgt='HEVC_D'):
    test_info = TEST_DATA[test_tgt]
    resolution_tgt = 'x64_resolution'

    GOP = int(test_info['gop'])
    total_frame_num = int(test_info['frames'])
    resolution = test_info['x64_resolution']
    W, H = int(resolution.split('x')[0]), int(resolution.split('x')[1])
    print(f'Test {test_tgt}, GOP={GOP}, H={H}, W={W}')

    porposed_psnr, porposed_bpp, porposed_msssim, porposed_enct, porposed_dect = [], [], [], [], []
    for l in [180, 420, 920, 1840]:
        model = RDVC().cuda()
        model_mode = model_mode
        if mode == "PSNR":
            if l ==180:
                restore_path = './checkpoints/p_pretrained/Single/PSNR/180/checkpoint.pth'
                lamda = 0.013
            elif l==420:
                restore_path = './checkpoints/p_pretrained/Single/PSNR/420/checkpoint.pth'
                lamda = 0.025
            elif l==920:
                restore_path = './checkpoints/p_pretrained/Single/PSNR/920/checkpoint.pth'
                lamda = 0.0483
            elif l==1840:
                restore_path = './checkpoints/p_pretrained/Single/PSNR/1840/checkpoint.pth'
                lamda = 0.0932
        elif mode == "MS_SSIM":
            if l ==180:
                restore_path = './checkpoints/p_pretrained/Single/MS-SSIM/180_50/checkpoint.pth'
                lamda = 16.64
            elif l==420:
                restore_path = './checkpoints/p_pretrained/Single/MS-SSIM/420_50/checkpoint.pth'
                lamda = 31.73
            elif l==920:
                restore_path = './checkpoints/p_pretrained/Single/MS-SSIM/920_50/checkpoint.pth'
                lamda = 60.5
            elif l==1840:
                restore_path = './checkpoints/p_pretrained/Single/MS-SSIM/1840_50/checkpoint.pth'
                lamda = 115.37

        print(f"INFO Try Load Pretrained Model From {restore_path}...")
        checkpoint = torch.load(restore_path, map_location='cuda:0')
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.update(force=True)

        log_path = 'Single_point'
        result_save_path = f'./logs/{log_path}/{mode}/{model_mode}/{test_tgt}'
        os.makedirs(result_save_path, exist_ok=True)

        # test model
        QP, I_level = get_quality(l_PSNR=l)

        PSNR, MSSSIM, Bits = AverageMeter(), AverageMeter(), AverageMeter()
        PSNR1, MSSSIM1, Bits1 = AverageMeter(), AverageMeter(), AverageMeter()
        encT, decT = AverageMeter(), AverageMeter()
        for seq_info in test_info['sequences']:

            print(f'INFO Process {seq_info}')
            video_frame_path = os.path.join(test_info['path'], 'PNG_Frames',
                                            seq_info.replace(test_info['org_resolution'],
                                                             test_info[resolution_tgt]))
            # exit()
            _PSNR, _MS_SSIM, _Bits = AverageMeter(), AverageMeter(), AverageMeter()
            _encT, _decT = AverageMeter(), AverageMeter()
            with torch.no_grad():
                for gop_index in range(int(np.ceil(total_frame_num / GOP))):
                    f = gop_index * GOP
                    print(f"\rINFO Seq={seq_info[0]}, QP= {QP}, Frame={f}", end='', flush=True)
                    if mode == "PSNR":
                        org_frame_path = os.path.join(video_frame_path, "f" + str(f + 1).zfill(3) + ".png")
                        curr_frame = read_image(org_frame_path).unsqueeze(0).cuda()
                        icodec = ICIP2020ResB().cuda()
                        ckpt = f'./checkpoints/i_pretrained/Single/PSNR/lambda_{lamda}.pth'
                        state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
                        state_dict = load_pretrained(state_dict)
                        icodec.load_state_dict(state_dict)

                        icodec.eval()
                        icodec.update(force=True)

                        i_out_enc = icodec.compress(curr_frame)
                        i_out_dec = icodec.decompress(i_out_enc["strings"], i_out_enc['shape'])
                        I_frame_bpp = sum(len(s[0]) for s in i_out_enc["strings"]) * 8.0
                        I_frame_psnr = cal_psnr(curr_frame, i_out_dec["x_hat"])
                        I_frame_msssim = ms_ssim(curr_frame, i_out_dec["x_hat"], data_range=1.0).item()
                        _PSNR.update(I_frame_psnr)
                        PSNR.update(I_frame_psnr)

                        _MS_SSIM.update(I_frame_msssim)
                        MSSSIM.update(I_frame_msssim)

                        _Bits.update(I_frame_bpp / H / W)
                        Bits.update(I_frame_bpp / H / W)

                        ref_frame = i_out_dec["x_hat"]

                    elif mode == "MS_SSIM":
                        org_frame_path = os.path.join(video_frame_path, "f" + str(f + 1).zfill(3) + ".png")
                        curr_frame = read_image(org_frame_path).unsqueeze(0).cuda()

                        icodec = ICIP2020ResB().cuda()
                        ckpt = f'./checkpoints/i_pretrained/Single/MS-SSIM/lambda_{lamda}.pth'
                        state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
                        state_dict = load_pretrained(state_dict)
                        icodec.load_state_dict(state_dict)

                        icodec.eval()
                        icodec.update(force=True)

                        i_out_enc = icodec.compress(curr_frame)
                        i_out_dec = icodec.decompress(i_out_enc["strings"], i_out_enc['shape'])
                        I_frame_bpp = sum(len(s[0]) for s in i_out_enc["strings"]) * 8.0
                        I_frame_psnr = cal_psnr(curr_frame, i_out_dec["x_hat"])
                        I_frame_msssim = ms_ssim(curr_frame, i_out_dec["x_hat"], data_range=1.0).item()
                        _PSNR.update(I_frame_psnr)
                        PSNR.update(I_frame_psnr)

                        _MS_SSIM.update(I_frame_msssim)
                        MSSSIM.update(I_frame_msssim)

                        _Bits.update(I_frame_bpp / H / W)
                        Bits.update(I_frame_bpp / H / W)

                        ref_frame = i_out_dec["x_hat"]

                    p_frame_num = np.min([GOP - 1, total_frame_num - f - 1])
                    feature = None

                    for p_frame in range(p_frame_num):
                        model.sample_active_subnet(model_mode, model_mode, model_mode, model_mode, model_mode,
                                                   model_mode, model_mode, model_mode)
                        f = f + 1
                        curr_frame_path = os.path.join(video_frame_path, "f" + str(f + 1).zfill(3) + ".png")
                        curr_frame = read_image(curr_frame_path).unsqueeze(0).cuda()
                        torch.cuda.synchronize()

                        torch.cuda.synchronize()
                        start = time.time()
                        mv_out_enc, res_out_enc, = model.compress(ref_frame, curr_frame, feature)
                        torch.cuda.synchronize()
                        enc_time = time.time() - start
                        encT.update(enc_time)
                        _encT.update(enc_time)

                        torch.cuda.synchronize()
                        start = time.time()
                        feature,  ref_frame, _, _ = model.decompress(ref_frame, mv_out_enc, res_out_enc, feature)
                        torch.cuda.synchronize()
                        dec_time = time.time() - start
                        decT.update(dec_time)
                        _decT.update(dec_time)


                        mse = torch.mean((curr_frame - ref_frame).pow(2)).item()
                        psnr = 10 * np.log10(1.0 / mse)
                        msssim = ms_ssim(curr_frame, ref_frame, data_range=1.0).item()
                        res_bpp = sum(len(s[0]) for s in res_out_enc["strings"]) * 8.0
                        mv_bpp = sum(len(s[0]) for s in mv_out_enc["strings"]) * 8.0
                        bits = mv_bpp + res_bpp

                        recon_image, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = model(ref_frame, curr_frame, feature)
                        PSNR1.update(10 * np.log10(1.0 / mse_loss.item()))
                        Bits1.update(bpp_res.item() + bpp_mv.item())

                        _PSNR.update(psnr)
                        PSNR.update(psnr)

                        _MS_SSIM.update(msssim)
                        MSSSIM.update(msssim)

                        _Bits.update(bits / H / W)
                        Bits.update(bits / H / W)

                print(f'\nQP {QP} | Seq={test_tgt}| Average BPP {_Bits.avg:.4f} | PSRN {_PSNR.avg:.4f} | '
                      f'MSSSIM {_MS_SSIM.avg:.4f} | Encode Time {_encT.avg:.4f} | Decode Time {_decT.avg:.4f}')

        print(f'Seq={test_tgt}| Average BPP {Bits.avg:.4f} | PSRN {PSNR.avg:.4f} | '
              f'MSSSIM {MSSSIM.avg:.4f} | Encode Time {encT.avg:.4f} | Decode Time {decT.avg:.4f}')

        porposed_psnr.append(PSNR.avg)
        porposed_msssim.append(MSSSIM.avg)
        porposed_bpp.append(Bits.avg)
        porposed_enct.append(encT.avg)
        porposed_dect.append(decT.avg)

    print('bpp', porposed_bpp)
    print('psnr', porposed_psnr)
    print('mssim', porposed_msssim)

    results = {"psnr": porposed_psnr, "ms-ssim": porposed_msssim, "bpp": porposed_bpp,
               "encoding_time": porposed_enct, "decoding_time": porposed_dect}
    output = {
        "name": 'RDVC',
        "description": "Inference (ans)",
        "results": results,
    }
    with open(os.path.join(f'{result_save_path}', f'{test_tgt}.json'), 'w',
              encoding='utf-8') as json_file:
        json.dump(output, json_file, indent=2)
    return None

#VB and VC
def get_results_flexible(indicator, test_tgt):
    test_info = TEST_DATA[test_tgt]
    resolution_tgt = 'x64_resolution'
    GOP = test_info['gop']
    total_frame_num = test_info['frames']
    resolution = test_info[resolution_tgt]
    W, H = int(resolution.split('x')[0]), int(resolution.split('x')[1])
    print(f'Test {test_tgt}, GOP={GOP}, H={H}, W={W}')
    i_l_offset = 3

    porposed_psnr, porposed_bpp, porposed_msssim, porposed_bpp2l = [], [], [], []
    porposed_ipsnr, porposed_ibpp, porposed_imsssim = [], [], []
    porposed_ppsnr, porposed_pbpp, porposed_pmsssim = [], [], []
    porposed_mcpsnr, porposed_warppsnr, porposed_mvbpp, porposed_resbpp = [], [], [], []
    porposed_mcmsssim, porposed_warmsssim = [], []
    porposed_ienc, porposed_idec, porposed_pent, porposed_pdec = [], [], [], []
    porposed_ent, porposed_dec = [], []
    with torch.no_grad():
        if indicator == 'PSNR':
            restore_path = f'./checkpoints/p_pretrained/Flexible/PSNR/checkpoint.pth'
        else:
            restore_path = f'./checkpoints/p_pretrained/Flexible/MS-SSIM/checkpoint.pth'
        print(restore_path)
        pcheckpoint = torch.load(restore_path, map_location='cpu')
        print(f"INFO Load Pretrained P-Model From Epoch {pcheckpoint['epoch']}...")
        p_model = RDVC(flexible=True).cuda()
        p_model.load_state_dict(pcheckpoint["state_dict"])
        p_model.eval()
        p_model.update(force=True)
        # mode = 'smallest'
        mode = 'largest'
        p_model.sample_active_subnet(mode, mode, mode, mode, mode, mode, mode, mode)

        for l in [1,2,3,4]:
            L1 = [1., 0.8, 0.6, 0.4, 0.2]
            intervals = L1 if l != 4 else [1]
            # intervals = [1]
            p_inerflag = True if l != 4 else False

            for interval in intervals:
                log_path = 'Flexible'
                result_save_path = f'./logs/{log_path}/{indicator}/{mode}/{test_tgt}'
                os.makedirs(result_save_path, exist_ok=True)

                file_path = f'{result_save_path}/detail/log_{test_tgt}_{l}_{interval:.2f}.txt'
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                log_txt = open(file_path, 'w')

                if indicator == 'MS_SSIM':
                    i_model = ICIP2020ResBVB1(v0=True, psnr=False).cuda()
                    i_restore_path = f'./checkpoints/i_pretrained/Flexible/MS-SSIM/ICIP2020ResBVB1_mssim.pth'
                elif indicator == 'PSNR':
                    i_model = ICIP2020ResBVB1(v0=True).cuda()
                    i_restore_path = f'./checkpoints/i_pretrained/Flexible/PSNR/ICIP2020ResBVB1_psnr.pth'
                print(i_restore_path)
                icheckpoint = torch.load(i_restore_path, map_location='cpu')
                print(f"INFO Load Pretrained I-Model From Epoch {icheckpoint['epoch']}...")
                state_dict = load_pretrained(icheckpoint["state_dict"])
                i_model.load_state_dict(state_dict)
                i_model.update(force=True)
                i_model.eval()

                PSNR, MSSSIM, Bits, Bits2l = [], [], [], []
                iPSNR, iMSSSIM, iBits = [], [], []
                pPSNR, pMSSSIM, pBits = [], [], []
                mcPSNR, warpPSNR, mvBits, resBits = [], [], [], []
                mcMSSSIM, warpMSSSIM = [], []
                iEnc, iDec, pEnc, pDec, Enc, Dec = [], [], [], [], [], []
                for ii, seq_info in enumerate(test_info['sequences']):
                    _PSNR, _MSSSIM, _Bits, _Bits2l = [], [], [], []
                    _iPSNR, _iMSSSIM, _iBits = [], [], []
                    _pPSNR, _pMSSSIM, _pBits = [], [], []
                    _mcPSNR, _warpPSNR, _mvBits, _resBits = [], [], [], []
                    _mcMSSSIM, _warpMSSSIM = [], []
                    _iEnc, _iDec, _pEnc, _pDec, _Enc, _Dec = [], [], [], [], [], []

                    video_frame_path = os.path.join(test_info['path'], 'PNG_Frames',
                                                    seq_info.replace(test_info['org_resolution'],
                                                                     test_info[resolution_tgt]))
                    images = sorted(glob.glob(os.path.join(video_frame_path, '*.png')))
                    print(f'INFO Process {seq_info}, Find {len(images)} images, Default test frames {total_frame_num}')
                    image = read_image(images[0]).unsqueeze(0)
                    num_pixels = image.size(0) * image.size(2) * image.size(3)
                    for i, im in enumerate(images):

                        if i >= total_frame_num:
                            break
                        curr_frame = read_image(im).unsqueeze(0).cuda()

                        if i % GOP == 0:
                            torch.cuda.synchronize()
                            start_time = time.perf_counter()
                            i_out_enc = i_model.compress(curr_frame,
                                                         [i_model.levels - i_l_offset - l], 1.0 - interval)
                            torch.cuda.synchronize()
                            elapsed_enc = time.perf_counter() - start_time
                            torch.cuda.synchronize()
                            start_time = time.perf_counter()
                            i_out_dec = i_model.decompress(i_out_enc["strings"], i_out_enc["shape"],
                                                           [i_model.levels - i_l_offset - l], 1.0 - interval)
                            torch.cuda.synchronize()
                            elapsed_dec = time.perf_counter() - start_time

                            i_bpp = sum(len(s[0]) for s in i_out_enc["strings"]) * 8.0 / num_pixels
                            i_psnr = cal_psnr(curr_frame, i_out_dec["x_hat"])
                            i_ms_ssim = ms_ssim(curr_frame, i_out_dec["x_hat"], data_range=1.0).item()

                            _iPSNR.append(i_psnr)
                            _iMSSSIM.append(i_ms_ssim)
                            _iBits.append(i_bpp)
                            _Bits2l.append(i_bpp)
                            _PSNR.append(i_psnr)
                            _MSSSIM.append(i_ms_ssim)
                            _Bits.append(i_bpp)
                            _iEnc.append(elapsed_enc)
                            _iDec.append(elapsed_dec)
                            _Enc.append(elapsed_enc)
                            _Dec.append(elapsed_dec)
                            print(
                                f"i={i}, {seq_info} I-Frame | bpp {i_bpp:.3f} | PSNR {i_psnr:.3f} | MS-SSIM {i_ms_ssim:.3f} | Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s ")
                            log_txt.write(
                                f"i={i} {seq_info} I-Frame | bpp {i_bpp:.3f} | PSNR {i_psnr:.3f} | MS-SSIM {i_ms_ssim:.3f} | Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s\n")
                            log_txt.flush()

                            ref_frame = i_out_dec["x_hat"]

                            feature = None
                        else:
                            # #此处可进行网络复杂度的详细设置
                            # a = 0  #a:0~4
                            # b = 0  #b:0~6
                            # c = 0  #c:0~4
                            # d = 0  #c:0~4
                            # e = 0  #c:0~4
                            # f = 0  #c:0~4
                            # cfg = {'cfg_mv': {'g_a': [64, 64, 64, 64], 'h_a': [64, 64, 64, 64],
                            #                   'g_s': [32 + (a * 8), 32 + (a * 8), 32 + (a * 8), 2],
                            #                   'h_s': [32 + (a * 8), 32 + (a * 8), 32 + (a * 8), 64],
                            #                   'mean_scale_factor': [96 + (a * 24), 96 + (a * 24), 96 + (a * 24), 192],
                            #                   'fea': [32 + (a * 8), 32 + (a * 8), 32 + (a * 8), 64]},
                            #        'cfg_res': {'g_a': [96, 96, 96, 96], 'h_a': [96, 96, 96, 96],
                            #                    'g_s': [48 + (b * 8), 48 + (b * 8), 48 + (b * 8), 64],
                            #                    'h_s': [48 + (b * 8), 48 + (b * 8), 48 + (b * 8), 96],
                            #                    'mean_scale_factor': [144 + (b * 24), 144 + (b * 24), 144 + (b * 24),
                            #                                          288],
                            #                    'fea': [48 + (b * 8), 48 + (b * 8), 48 + (b * 8), 96]},
                            #        'cfg_mc': {'MC_net': [32 + (c * 8), 32 + (c * 8), 3]},
                            #        'cfg_refinemv': {'Refine': [32 + (d * 8)]},
                            #        'cfg_refineres': {'Refine': [32 + (d * 8)]},
                            #        'cfg_FeaExt': {'FeaExt': [32 + (d * 8), 32 + (d * 8), 64]},
                            #        'cfg_fusionfea': {'Fusion': [32 + (d * 8)]},
                            #        'cfg_enhance': {'Rec_net': [64, 32 + (e * 8), 32 + (f * 8), 3, 3, 32 + (f * 8)]}}
                            # p_model.set_active_subnet(cfg)

                            p_model.set_rate_level(index=l, interpolation_coefficient=interval, isInterpolation=p_inerflag)

                            torch.cuda.synchronize()
                            start = time.time()
                            mv_out_enc, res_out_enc = p_model.compress(ref_frame, curr_frame, feature)

                            torch.cuda.synchronize()
                            elapsed_enc = time.time() - start

                            torch.cuda.synchronize()
                            start = time.time()
                            feature, dec_p_frame, _, _ = p_model.decompress(ref_frame, mv_out_enc, res_out_enc, feature)
                            torch.cuda.synchronize()
                            elapsed_dec = time.time() - start

                            mse = torch.mean((curr_frame - dec_p_frame).pow(2)).item()
                            p_psnr = 10 * np.log10(1.0 / mse).item()
                            p_ms_ssim = ms_ssim(curr_frame, dec_p_frame, data_range=1.0).item()
                            res_bpp = sum(len(s[0]) for s in res_out_enc["strings"]) * 8.0 / num_pixels
                            mv_bpp = sum(len(s[0]) for s in mv_out_enc["strings"]) * 8.0 / num_pixels
                            p_bpp = mv_bpp + res_bpp

                            ref_frame = dec_p_frame.detach()
                            _PSNR.append(p_psnr)
                            _MSSSIM.append(p_ms_ssim)
                            _Bits.append(p_bpp)
                            _pPSNR.append(p_psnr)
                            _pMSSSIM.append(p_ms_ssim)
                            _pBits.append(p_bpp)

                            _mvBits.append(mv_bpp)
                            _resBits.append(res_bpp)
                            _Bits2l.append(mv_bpp)

                            _pEnc.append(elapsed_enc)
                            _pDec.append(elapsed_dec)
                            _Enc.append(elapsed_enc)
                            _Dec.append(elapsed_dec)

                            reconstructed_image_path = os.path.join(result_save_path,
                                                                    f'reconstructed_p_frame_{i}.png')
                            # torchvision.utils.save_image(dec_p_frame, reconstructed_image_path)
                            log_txt.write(
                                f"{l, interval}, i={i}, {seq_info} P-Frame | bpp {p_bpp:.3f} | PSNR {p_psnr:.3f} | Saved to {reconstructed_image_path}\n")

                            print(
                                f"{l, interval}, i={i}, {seq_info} P-Frame | bpp [{mv_bpp:.3f}, {res_bpp:.4f}, {p_bpp:.3f}] "
                                f"| PSNR [{p_psnr:.3f}] "
                                f"| |{p_ms_ssim:.3f}]"
                                f"| Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s ")
                            log_txt.write(
                                f"{l, interval}, i={i}, {seq_info} P-Frame | bpp [{mv_bpp:.3f}, {res_bpp:.4f}, {p_bpp:.3f}] "
                                f"| PSNR |{p_psnr:.3f}] "
                                f"| MS-SSIM [{p_ms_ssim:.3f}] "
                                f"| Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s\n")
                            log_txt.flush()
                    print(f'{l, interval}, I-Frame | Average BPP {np.average(_iBits):.4f} | PSRN {np.average(_iPSNR):.4f}')
                    print(f'{l, interval}, P-Frame | Average BPP {np.average(_pBits):.4f} | PSRN {np.average(_pPSNR):.4f}')
                    print(f'{l, interval}, Frame | Average BPP {np.average(_Bits):.4f} | PSRN {np.average(_PSNR):.4f}')

                    log_txt.write(f'{l, interval}, I-Frame | Average BPP {np.average(_iBits):.4f} | PSRN {np.average(_iPSNR):.4f}\n')
                    log_txt.write(f'{l, interval}, P-Frame | Average BPP {np.average(_pBits):.4f} | PSRN {np.average(_pPSNR):.4f}\n')
                    log_txt.write(f'{l, interval}, Frame | Average BPP {np.average(_Bits):.4f} | PSRN {np.average(_PSNR):.4f}\n')

                    PSNR.append(np.average(_PSNR))
                    MSSSIM.append(np.average(_MSSSIM))
                    Bits.append(np.average(_Bits))
                    Bits2l.append(np.average(_Bits2l))
                    iPSNR.append(np.average(_iPSNR))
                    iMSSSIM.append(np.average(_iMSSSIM))
                    iBits.append(np.average(_iBits))
                    pPSNR.append(np.average(_pPSNR))
                    pMSSSIM.append(np.average(_pMSSSIM))
                    pBits.append(np.average(_pBits))
                    mcPSNR.append(np.average(_mcPSNR))
                    warpPSNR.append(np.average(_warpPSNR))
                    mvBits.append(np.average(_mvBits))
                    resBits.append(np.average(_resBits))
                    mcMSSSIM.append(np.average(_mcMSSSIM))
                    warpMSSSIM.append(np.average(_warpMSSSIM))
                    iEnc.append(np.average(_iEnc))
                    iDec.append(np.average(_iDec))
                    pEnc.append(np.average(_pEnc))
                    pDec.append(np.average(_pDec))
                    Enc.append(np.average(_Enc))
                    Dec.append(np.average(_Dec))

                porposed_psnr.append(np.average(PSNR))
                porposed_bpp.append(np.average(Bits))
                porposed_bpp2l.append(np.average(Bits2l))
                porposed_msssim.append(np.average(MSSSIM))
                porposed_ipsnr.append(np.average(iPSNR))
                porposed_ibpp.append(np.average(iBits))
                porposed_imsssim.append(np.average(iMSSSIM))
                porposed_ppsnr.append(np.average(pPSNR))
                porposed_pbpp.append(np.average(pBits))
                porposed_pmsssim.append(np.average(pMSSSIM))

                porposed_mcpsnr.append(np.average(mcPSNR))
                porposed_warppsnr.append(np.average(warpPSNR))
                porposed_mvbpp.append(np.average(mvBits))
                porposed_resbpp.append(np.average(resBits))
                porposed_mcmsssim.append(np.average(mcMSSSIM))
                porposed_warmsssim.append(np.average(warpMSSSIM))
                porposed_ienc.append(np.average(iEnc))
                porposed_idec.append(np.average(iDec))
                porposed_pent.append(np.average(pEnc))
                porposed_pdec.append(np.average(pDec))
                porposed_ent.append(np.average(Enc))
                porposed_dec.append(np.average(Dec))

            log_txt.close()
        print(porposed_bpp)
        print(porposed_psnr)
        print(porposed_msssim)
        results = {
            "psnr": porposed_psnr, "bpp": porposed_bpp, "msssim": porposed_msssim, "bpp2l": porposed_bpp2l,
            "ipsnr": porposed_ipsnr, "ibpp": porposed_ibpp, "imsssim": porposed_imsssim,
            "ppsnr": porposed_ppsnr, "pbpp": porposed_pbpp, "pmsssim": porposed_pmsssim,
            "mcpsnr": porposed_mcpsnr, "warppsnr": porposed_warppsnr, "mvbpp": porposed_mvbpp,
            "resbpp": porposed_resbpp, "mcmsssim": porposed_mcmsssim, "warmsssim": porposed_warmsssim,
            "ienc": porposed_ienc, "idec": porposed_idec, "pent": porposed_pent,
            "pdec": porposed_pdec, "ent": porposed_ent, "dec": porposed_dec,
        }
        output = {
            "name": f'{test_tgt}',
            "description": "Inference (ans)",
            "results": results,
        }
        with open(os.path.join(result_save_path, f'{test_tgt}.json'), 'w',
                  encoding='utf-8') as json_file:
            json.dump(output, json_file, indent=2)

    return None


if __name__ == '__main__':
    pass

