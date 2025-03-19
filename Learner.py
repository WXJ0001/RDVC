import json
import os
import random
import shutil
from glob import glob
import torch
from tensorboardX import SummaryWriter
import logging
import utils
from image_model import ICIP2020ResB
from utils import AverageMeter, load_pretrained
from dataset import get_dataset11
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim
import numpy as np
import datetime
from tqdm import tqdm

from video_model import RDVC


def random_index(rate):
    start = 0
    index = 0
    rand = random.randint(1, sum(rate))
    for index, scope in enumerate(rate):
        start += scope
        if rand <= start:
            break
    return index

#RDVC
class Trainer_RDVC(object):
    def __init__(self, args):
        # args
        args.cuda = torch.cuda.is_available()
        self.args = args
        utils.fix_random_seed(args.seed)

        self.stage1_step = 3e5  # 2frames
        self.stage2_step = self.stage1_step + 1e5  # 2frames
        self.stage3_step = self.stage2_step + 2e5  # 5frames
        self.lambda_list = [60, 180, 420, 920, 1840]

        self.mode_list = ['largest', 'random', 'largest']
        self.grad_clip = 1.0

        # logs
        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(args.log_root, f"RDVC_flexible_{date}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        dirs_to_make = next(os.walk('./'))[1]
        not_dirs = ['.data', '.checkpoint', 'logs', '.gitignore', '.venv', '__pycache__']
        os.makedirs(os.path.join(self.log_dir, 'codes'), exist_ok=True)
        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            os.makedirs(os.path.join(self.log_dir, 'codes', to_make))

        pyfiles = glob("./*.py")
        for py in pyfiles:
            shutil.copyfile(py, os.path.join(self.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                shutil.copyfile(py, os.path.join(self.log_dir, 'codes', py[2:]))

        # logger
        utils.setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')

        # data
        self.frames = args.frames
        self.batch_size = args.batch_size
        self.Height, self.Width, self.Channel = self.args.image_size
        self.l_PSNR = args.l_PSNR
        self.l_MSSSIM = args.l_MSSSIM

        training_set, valid_set = get_dataset11(args, mf=7, crop=True)
        # training_set, valid_set = get_dataset11(args, mf=7, crop=True, worgi=True)
        self.training_set_loader = DataLoader(training_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              )

        self.valid_set_loader = DataLoader(valid_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           )
        self.logger.info(f'[*] Train File Account For {len(training_set)}, val {len(valid_set)}')

        # epoch
        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = 0
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        #image_model
        self.key_frame_models = {}
        self.logger.info(f"[*] Try Load Pretrained Image Codec Model...")

        for i, I_lambda in enumerate([0.0067, 0.013, 0.025, 0.0483, 0.0932]):
            codec = ICIP2020ResB()
            ckpt = f'/tdx/databak/wxj/IModel/deepi/ICIP2020ResB/mse/lambda_{I_lambda}.pth'
            state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
            state_dict = load_pretrained(state_dict)
            codec.load_state_dict(state_dict)
            self.key_frame_models[i] = codec

        for q in self.key_frame_models.keys():
            for param in self.key_frame_models[q].parameters():
                param.requires_grad = False
            self.key_frame_models[q] = self.key_frame_models[q].eval().cuda()

        # video_model
        self.mode_type = args.mode_type
        self.graph = RDVC(flexible=True).cuda()

        # self.logger.info(f"[*] Try Load Pretrained Video Codec Model...")
        # ckpt = ''
        # tgt_model_dict = self.graph.state_dict()
        # src_pretrained_dict = torch.load(ckpt)['state_dict']
        # _pretrained_dict = {k: v for k, v in src_pretrained_dict.items() if k in tgt_model_dict}
        # for k, v in src_pretrained_dict.items():
        #     if k not in tgt_model_dict:
        #         print(k)
        # tgt_model_dict.update(_pretrained_dict)
        # self.graph.load_state_dict(tgt_model_dict)

        # device
        self.cuda = args.use_gpu
        self.device = next(self.graph.parameters()).device
        self.logger.info(
            f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')

        self.configure_optimizers(args)

        self.lowest_val_loss = float("inf")

        if args.load_pretrained:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        sample_list_num = len(self.mode_list)
        state_list = ["train_bpp", 'train_loss', 'train_warp_psnr', 'train_mc_psnr', 'train_res_bpp', 'train_mv_bpp', 'train_psnr', 'train_msssim', 'train_aux',
                      "train_res_aux", "train_mv_aux"]

        for epoch in range(self.start_epoch, self.num_epochs):
            self.global_epoch = epoch
            log_state = {}
            for index in range(len(self.lambda_list)):
                train_bpp, train_loss = [AverageMeter() for i in range(0, sample_list_num)], \
                                        [AverageMeter() for i in range(0, sample_list_num)]

                train_warp_psnr, train_mc_psnr = [AverageMeter() for i in range(0, sample_list_num)], \
                                                 [AverageMeter() for i in range(0, sample_list_num)]

                train_res_bpp, train_mv_bpp = [AverageMeter() for i in range(0, sample_list_num)], \
                                                 [AverageMeter() for i in range(0, sample_list_num)]

                train_psnr, train_msssim = [AverageMeter() for i in range(0, sample_list_num)], \
                                           [AverageMeter() for i in range(0, sample_list_num)]

                train_aux, train_res_aux, train_mv_aux = [AverageMeter() for i in range(0, sample_list_num)], \
                                                         [AverageMeter() for i in range(0, sample_list_num)], [AverageMeter() for i in range(0, sample_list_num)]


                train_aux = [AverageMeter() for i in range(0, sample_list_num)]
                log_state[str(index)] = {}
                for name in state_list:
                    log_state[str(index)][name] = eval(name)

            # adjust learning_rate
            self.adjust_lr()
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'[*] lr = {lr}')
            self.graph.train()
            train_bar = tqdm(self.training_set_loader)
            for kk, batch in enumerate(train_bar):
                frames = [frame.to(self.device) for frame in batch]
                f = self.get_f()
                if self.global_step > self.stage1_step + self.stage2_step // 2:
                    rate_index = random_index([19, 16, 15, 16, 34])
                    sample_idx = random_index([35, 30, 35])
                    # sample_idx = 0
                    #rate_index = 4
                else:
                    rate_index = random.randint(0, len(self.lambda_list) - 1)
                    sample_idx = random.randint(0, len(self.mode_list) - 1)

                with torch.no_grad():
                    ref_frame = self.key_frame_models[rate_index](frames[0])['x_hat']
                feature = None
                model_mode = self.mode_list[sample_idx]
                self.graph.set_rate_level(rate_index, 1.0, isInterpolation=False)
                self.graph.sample_active_subnet(mv_mode=model_mode, res_mode=model_mode, mc_mode=model_mode,
                                                refinemv_mode=model_mode, refineres_mode=model_mode, FeaExt_mode=model_mode,
                                                fusionfea_mode=model_mode,
                                                enhance_mode=model_mode)
                if 0 <= self.global_step < self.stage3_step:
                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, feature)
                        ref_frame = decoded_frame.detach().clone()
                        feature = feature1.detach().clone()

                        if self.global_epoch < self.stage1_step // 2:
                            distortion = mse_loss + 0.1 * warp_loss + 0.2 * mc_loss
                        elif self.stage1_step // 2 <= self.global_epoch < self.stage1_step:
                            distortion = mse_loss + 0.2 * mc_loss
                        else:
                            distortion = mse_loss
                        loss = distortion * self.lambda_list[rate_index] + bpp
                        self.optimizer.zero_grad()
                        self.aux_optimizer.zero_grad()
                        loss.backward()

                        self.clip_gradient(self.optimizer, self.grad_clip)

                        self.optimizer.step()
                        aux_loss = self.graph.aux_loss()
                        aux_loss.backward()

                        # self.clip_gradient(self.aux_optimizer, self.grad_clip)
                        self.aux_optimizer.step()

                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        psnr = 10 * np.log10(1.0 / mse_loss.detach().cpu())
                        mc_psnr = 10 * np.log10(1.0 / mc_loss.detach().cpu())
                        warp_psnr = 10 * np.log10(1.0 / warp_loss.detach().cpu())

                        log_state[str(rate_index)]['train_mc_psnr'][sample_idx].update(mc_psnr.mean().detach().item(),
                                                                                    self.batch_size)
                        log_state[str(rate_index)]['train_warp_psnr'][sample_idx].update(warp_psnr.mean().detach().item(),
                                                                                    self.batch_size)
                        log_state[str(rate_index)]['train_psnr'][sample_idx].update(psnr.mean().detach().item(),
                                                                                    self.batch_size)
                        log_state[str(rate_index)]['train_mv_bpp'][sample_idx].update(bpp_mv.mean().detach().item(),
                                                                                   self.batch_size)
                        log_state[str(rate_index)]['train_res_bpp'][sample_idx].update(bpp_res.mean().detach().item(),
                                                                                   self.batch_size)
                        log_state[str(rate_index)]['train_bpp'][sample_idx].update(bpp.mean().detach().item(),
                                                                                   self.batch_size)
                        log_state[str(rate_index)]['train_loss'][sample_idx].update(loss.mean().detach().item(),
                                                                                    self.batch_size)
                        log_state[str(rate_index)]['train_res_aux'][sample_idx].update(res_aux_loss.mean().detach().item(),
                                                                                   self.batch_size)
                        log_state[str(rate_index)]['train_mv_aux'][sample_idx].update(mv_aux_loss.mean().detach().item(),
                                                                                   self.batch_size)
                        log_state[str(rate_index)]['train_aux'][sample_idx].update(aux_loss.mean().detach().item(),
                                                                                   self.batch_size)
                        if self.global_step % 300 == 0:
                            self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_mv_aux', mv_aux_loss.detach().item(),
                                                   self.global_step)
                            self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_res_aux', res_aux_loss.detach().item(),
                                                   self.global_step)
                            self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_aux', aux_loss.detach().item(),
                                                   self.global_step)
                            self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_psnr', psnr, self.global_step)
                            self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_mc_psnr', mc_psnr, self.global_step)
                            self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_warp_psnr', warp_psnr, self.global_step)
                            self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_mv_bpp', bpp_mv.mean().detach().item(),
                                                   self.global_step)
                            self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_res_bpp', bpp_res.mean().detach().item(),
                                                   self.global_step)
                            self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_bpp', bpp.mean().detach().item(),
                                                   self.global_step)
                            self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_loss', loss.mean().detach().item(),
                                                   self.global_step)
                        train_bar.desc = "T-ALL lanmda:{} mode:{}_f:{} [{}|{}] [{}|{}] LOSS[{:.4f}], PSNR[{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.3f}]". \
                            format(self.lambda_list[rate_index],
                                   self.mode_list[sample_idx],
                                   f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   self.lambda_list[rate_index],
                                   loss.mean().detach().item(),
                                   warp_psnr,
                                   mc_psnr,
                                   psnr,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )
                        self.global_step += 1

                else:
                    _mse, _bpp, _aux_loss = torch.zeros([]).cuda(), torch.zeros([]).cuda(), torch.zeros([]).cuda()
                    for index in range(1, f):
                        curr_frame = frames[index]
                        ref_frame, feature, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, feature)
                        _mse += mse_loss * index
                        _bpp += bpp * index
                        _aux_loss += self.graph.aux_loss() * index

                        aux_loss = self.graph.aux_loss()
                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        psnr = 10 * np.log10(1.0 / torch.mean(mse_loss).detach().cpu())
                        mc_psnr = 10 * np.log10(1.0 / torch.mean(mc_loss).detach().cpu())
                        warp_psnr = 10 * np.log10(1.0 / torch.mean(warp_loss).detach().cpu())

                        _loss = torch.mean(mse_loss) + bpp

                        log_state[str(rate_index)]['train_mc_psnr'][sample_idx].update(mc_psnr.mean().detach().item(),
                                                                                    self.batch_size)
                        log_state[str(rate_index)]['train_warp_psnr'][sample_idx].update(warp_psnr.mean().detach().item(),
                                                                                    self.batch_size)
                        log_state[str(rate_index)]['train_psnr'][sample_idx].update(psnr.mean().detach().item(),
                                                                                    self.batch_size)
                        log_state[str(rate_index)]['train_mv_bpp'][sample_idx].update(bpp_mv.mean().detach().item(),
                                                                                   self.batch_size)
                        log_state[str(rate_index)]['train_res_bpp'][sample_idx].update(bpp_res.mean().detach().item(),
                                                                                   self.batch_size)
                        log_state[str(rate_index)]['train_bpp'][sample_idx].update(bpp.mean().detach().item(),
                                                                                   self.batch_size)
                        log_state[str(rate_index)]['train_loss'][sample_idx].update(_loss.mean().detach().item(),
                                                                                    self.batch_size)
                        log_state[str(rate_index)]['train_res_aux'][sample_idx].update(res_aux_loss.mean().detach().item(),
                                                                                   self.batch_size)
                        log_state[str(rate_index)]['train_mv_aux'][sample_idx].update(mv_aux_loss.mean().detach().item(),
                                                                                   self.batch_size)
                        log_state[str(rate_index)]['train_aux'][sample_idx].update(aux_loss.mean().detach().item(),
                                                                                   self.batch_size)
                        self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_mv_aux', mv_aux_loss.detach().item(),
                                               self.global_step)
                        self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_res_aux', res_aux_loss.detach().item(),
                                               self.global_step)
                        self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_aux', aux_loss.detach().item(),
                                               self.global_step)
                        self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_mc_psnr', mc_psnr, self.global_step)
                        self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_warp_psnr', warp_psnr, self.global_step)
                        self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_mv_bpp', bpp_mv.mean().detach().item(),
                                               self.global_step)
                        self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_res_bpp', bpp_res.mean().detach().item(),
                                               self.global_step)
                        self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_bpp', bpp.mean().detach().item(),
                                               self.global_step)
                        self.writer.add_scalar(f'{self.lambda_list[rate_index]}_{sample_idx}_train_loss', _loss.mean().detach().item(),
                                               self.global_step)

                        train_bar.desc = "Final lanmda:{} mode:{} f:{} [{}|{}|{}] [{}|{}] LOSS[{:.1f}], PSNR[{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(self.lambda_list[rate_index],
                                   sample_idx,
                                    f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   rate_index,
                                   self.lambda_list[rate_index],
                                   _loss.mean().detach().item(),
                                   warp_psnr,
                                   mc_psnr,
                                   psnr,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )

                    distortion = _mse * self.lambda_list[rate_index]
                    # print(distortion)
                    num = f * (f + 1) // 2
                    loss = distortion.div(num) + _bpp.div(num)
                    # print('===loss', loss)

                    self.optimizer.zero_grad()
                    self.aux_optimizer.zero_grad()
                    loss.backward()
                    self.clip_gradient(self.optimizer, self.grad_clip)
                    self.optimizer.step()
                    aux_loss = _aux_loss.div(num)
                    aux_loss.backward()
                    self.aux_optimizer.step()
                    self.global_step += 1

            train_loss_avg = 0
            for rate_index in range(len(self.lambda_list)):
                for sample_idx, model_mode in enumerate(self.mode_list):
                    self.logger.info("T-ALL-{}-{} [{}|{}] LOSS[{:.4f}], PSNR[{:.3f}|{:.3f}|{:.3f}], " \
                                     "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.3f}|{:.3f}|{:.3f}]". \
                                     format(self.lambda_list[rate_index],
                                            model_mode,
                                            epoch + 1,
                                            self.num_epochs,
                                            log_state[str(rate_index)]['train_loss'][sample_idx].avg,
                                            log_state[str(rate_index)]['train_warp_psnr'][sample_idx].avg,
                                            log_state[str(rate_index)]['train_mc_psnr'][sample_idx].avg,
                                            log_state[str(rate_index)]['train_psnr'][sample_idx].avg,
                                            log_state[str(rate_index)]['train_mv_bpp'][sample_idx].avg,
                                            log_state[str(rate_index)]['train_res_bpp'][sample_idx].avg,
                                            log_state[str(rate_index)]['train_bpp'][sample_idx].avg,
                                            log_state[str(rate_index)]['train_mv_aux'][sample_idx].avg,
                                            log_state[str(rate_index)]['train_res_aux'][sample_idx].avg,
                                            log_state[str(rate_index)]['train_aux'][sample_idx].avg
                                            ))
                    train_loss_avg += log_state[str(rate_index)]['train_loss'][sample_idx].avg / len(self.mode_list)


            # Needs to be called once after training
            self.graph.update()
            self.save_checkpoint(train_loss_avg, f"checkpoint_{epoch}.pth", is_best=False)
            if epoch % self.args.val_freq == 0:
                self.validate()
                # pass
        # Needs to be called once after training
        self.graph.update()

    def validate(self):
        sample_list = ['largest']
        state_list = ["val_bpp", 'val_loss', 'val_warp_psnr', 'val_mc_psnr', 'val_res_bpp', 'val_mv_bpp', 'val_psnr', 'val_msssim', 'val_aux',
                      "val_res_aux", "val_mv_aux"]
        self.graph.eval()
        log_state = {}
        for index in range(len(self.lambda_list)):
            val_bpp, val_loss = [AverageMeter() for i in range(0, len(sample_list))], \
                [AverageMeter() for i in range(0, len(sample_list))]

            val_warp_psnr, val_mc_psnr = [AverageMeter() for i in range(0, len(sample_list))], \
                [AverageMeter() for i in range(0, len(sample_list))]

            val_res_bpp, val_mv_bpp = [AverageMeter() for i in range(0, len(sample_list))], \
                [AverageMeter() for i in range(0, len(sample_list))]

            val_psnr, val_msssim = [AverageMeter() for i in range(0, len(sample_list))], \
                [AverageMeter() for i in range(0, len(sample_list))]

            val_aux, val_res_aux, val_mv_aux = [AverageMeter() for i in range(0, len(sample_list))], \
                [AverageMeter() for i in range(0, len(sample_list))], [AverageMeter() for i in range(0, len(sample_list))]

            log_state[str(index)] = {}
            for name in state_list:
                log_state[str(index)][name] = eval(name)

        with torch.no_grad():
            valid_bar = tqdm(self.valid_set_loader)
            for k, batch in enumerate(valid_bar):
                frames = [frame.to(self.device) for frame in batch]
                f = self.get_f()
                for rate_index in range(0, len(self.lambda_list)):
                    for sample_idx, model_mode in enumerate(sample_list):
                        ref_frame = self.key_frame_models[rate_index](frames[0])['x_hat']
                        self.graph.set_rate_level(rate_index)
                        self.graph.sample_active_subnet(mv_mode=model_mode, res_mode=model_mode, mc_mode=model_mode,
                                                        refinemv_mode=model_mode, refineres_mode=model_mode,
                                                        FeaExt_mode=model_mode, fusionfea_mode=model_mode, enhance_mode=model_mode)
                        feature = None
                        for frame_index in range(1, f):
                            curr_frame = frames[frame_index]
                            decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                                self.graph(ref_frame, curr_frame, feature)
                            ref_frame = decoded_frame.detach().clone()
                            feature = feature1.detach().clone()
                            distortion = mse_loss * self.lambda_list[rate_index]
                            loss = distortion + bpp
                            self.optimizer.zero_grad()

                            msssim = ms_ssim(curr_frame.detach(), decoded_frame.detach(), data_range=1.0)
                            psnr = 10 * np.log10(1.0 / mse_loss.detach().cpu())
                            mc_psnr = 10 * np.log10(1.0 / mc_loss.detach().cpu())
                            warp_psnr = 10 * np.log10(1.0 / warp_loss.detach().cpu())

                            mv_aux = self.graph.mv_aux_loss()
                            res_aux = self.graph.res_aux_loss()
                            aux = self.graph.aux_loss()

                            log_state[str(rate_index)]['val_mc_psnr'][sample_idx].update(mc_psnr.mean().detach().item(),
                                                                                        self.args.val_freq)
                            log_state[str(rate_index)]['val_warp_psnr'][sample_idx].update(warp_psnr.mean().detach().item(),
                                                                                        self.args.val_freq)
                            log_state[str(rate_index)]['val_mv_bpp'][sample_idx].update(bpp_mv.mean().detach().item(),
                                                                                       self.args.val_freq)
                            log_state[str(rate_index)]['val_res_bpp'][sample_idx].update(bpp_res.mean().detach().item(),
                                                                                       self.args.val_freq)
                            log_state[str(rate_index)]['val_loss'][sample_idx].update(loss.mean().detach().item(),
                                                                                        self.args.val_freq)
                            log_state[str(rate_index)]['val_res_aux'][sample_idx].update(res_aux.mean().detach().item(),
                                                                                       self.args.val_freq)
                            log_state[str(rate_index)]['val_mv_aux'][sample_idx].update(mv_aux.mean().detach().item(),
                                                                                       self.args.val_freq)
                            log_state[str(rate_index)]['val_aux'][sample_idx].update(aux.mean().detach().item(),
                                                                                       self.args.val_freq)
                            log_state[str(rate_index)]['val_psnr'][sample_idx].update(psnr.mean(), self.args.val_freq)
                            log_state[str(rate_index)]['val_bpp'][sample_idx].update(bpp.mean().detach().item(),
                                                                                     self.args.val_freq)
                            log_state[str(rate_index)]['val_msssim'][sample_idx].update(msssim.mean().detach().item(),
                                                                                     self.args.val_freq)
                            valid_bar.desc = "V{}-{} {:4d} [{}|{}] LOSS[{:.1f}], PSNR[{:.3f}|{:.3f}|{:.3f}], BPP[{:.3f}|{:.3f}|{:.3f}] " \
                                             "AUX[{:.1f}|{:.1f}|{:1f}]".format(
                                f,
                                model_mode,
                                self.lambda_list[rate_index],
                                self.global_epoch + 1,
                                self.num_epochs,
                                loss.mean().detach().item(),
                                warp_psnr.mean(),
                                mc_psnr.mean(),
                                psnr.mean(),
                                bpp_mv.mean().detach().item(),
                                bpp_res.mean().detach().item(),
                                bpp.mean().detach().item(),
                                mv_aux.detach().item(),
                                res_aux.detach().item(),
                                aux.detach().item(),
                            )
                self.global_eval_step += 1

                if k > 100:
                    break

        self.logger.info(f"VALID [{self.global_epoch + 1}|{self.num_epochs}]")
        val_loss_avg = 0
        for rate_index in range(len(self.lambda_list)):
            for sample_idx, model_mode in enumerate(sample_list):
                self.logger.info("VALID-{}-{} [{}|{}] LOSS[{:.4f}], PSNR[{:.3f}|{:.3f}|{:.3f}], " \
                                 "BPP[{:.3f}|{:.3f}|{:.3f}]". \
                                 format(self.lambda_list[rate_index],
                                        model_mode,
                                        self.global_epoch + 1,
                                        self.num_epochs,
                                        log_state[str(rate_index)]['val_loss'][sample_idx].avg,
                                        log_state[str(rate_index)]['val_warp_psnr'][sample_idx].avg,
                                        log_state[str(rate_index)]['val_mc_psnr'][sample_idx].avg,
                                        log_state[str(rate_index)]['val_psnr'][sample_idx].avg,
                                        log_state[str(rate_index)]['val_mv_bpp'][sample_idx].avg,
                                        log_state[str(rate_index)]['val_res_bpp'][sample_idx].avg,
                                        log_state[str(rate_index)]['val_bpp'][sample_idx].avg,
                                        ))
                val_loss_avg += log_state[str(rate_index)]['val_loss'][sample_idx].avg / len(sample_list)

        is_best = bool(val_loss_avg < self.lowest_val_loss)
        self.lowest_val_loss = min(self.lowest_val_loss, val_loss_avg)
        self.save_checkpoint(val_loss_avg, "checkpoint.pth", is_best)
        self.graph.train()

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.model_restore_path}...")
        checkpoint = torch.load(self.args.model_restore_path, map_location=self.device)
        last_epoch = checkpoint["epoch"] + 1
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")

        self.graph.load_state_dict(checkpoint["state_dict"])
        self.aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = last_epoch
        self.global_step = checkpoint["global_step"] + 1
        del checkpoint

    def adjust_lr(self):

        if self.global_step < self.stage3_step:
            pass
        elif self.stage2_step < self.global_step <= self.stage3_step:
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        else:
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 10.0

    def get_f(self):
        if self.global_step < self.stage2_step:
            f = 2
        elif self.stage2_step < self.global_step < self.stage3_step:
            f = 5
        else:
            f = 5
        return f

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "global_step": self.global_step,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "aux_optimizer": self.aux_optimizer.state_dict(),
        }
        torch.save(state, os.path.join(self.checkpoints_dir, name))
        if is_best:
            torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))

    def configure_optimizers(self, args):
        bp_parameters = list(p for n, p in self.graph.named_parameters() if not n.endswith(".quantiles"))
        aux_parameters = list(p for n, p in self.graph.named_parameters() if n.endswith(".quantiles"))
        self.optimizer = torch.optim.Adam(bp_parameters, lr=args.lr)
        self.aux_optimizer = torch.optim.Adam(aux_parameters, lr=args.aux_lr)
        return None

    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)










