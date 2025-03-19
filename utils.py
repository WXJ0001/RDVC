import os
import numpy as np
import random
import logging
import datetime
import argparse
import torch
from torchvision import transforms
import math
import torch.nn.functional as F
from PIL import Image


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_net_device(net):
    return net.parameters().__next__().device


def file_path_init(args):
    from glob import glob
    from shutil import copyfile
    date = str(datetime.datetime.now())
    date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
    makedirs('./logs')
    args.log_dir = os.path.join(args.log_root, f"{args.CompressorName}_PSNR{args.l_PSNR}_{date}")
    makedirs(args.log_dir)
    dirs_to_make = next(os.walk('./'))[1]
    not_dirs = ['.data', '.chekpoint', 'logs', 'results', '.gitignore', '.nsmlignore', 'resrc']
    makedirs(os.path.join(args.log_dir, 'codes'))
    for to_make in dirs_to_make:
        if to_make in not_dirs:
            continue
        makedirs(os.path.join(args.log_dir, 'codes', to_make))

    # if not args.load_pretrained:
    pyfiles = glob("./*.py")
    for py in pyfiles:
        copyfile(py, os.path.join(args.log_dir, 'codes') + "/" + py)

    for to_make in dirs_to_make:
        if to_make in not_dirs:
            continue
        tmp_files = glob(os.path.join('./', to_make, "*.py"))
        for py in tmp_files:
            copyfile(py, os.path.join(args.log_dir, 'codes', py[2:]))

def get_args():
    parser = argparse.ArgumentParser(description='Deep Video Coding')
    parser.add_argument("--CompressorName", type=str, default="RDVC")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--state", type=str, default="train", choices=["train", "test"])
    # model path
    parser.add_argument("--model_restore_path", type=str, default="")
    parser.add_argument("--load_pretrained", type=bool, default=False)
    parser.add_argument("--train_gain", type=bool, default=False)
    parser.add_argument("--train_gate", type=bool, default=False)
    parser.add_argument("--log_root", type=str, default="./logs/train")

    parser.add_argument("--mode_type", type=str, default='PSNR')  # PSNR  MSSSIM  I_level
    parser.add_argument("--l_PSNR", type=int, default=1840, choices= [256, 512, 1024, 2048]) #[256, 512, 1024, 2048, 4096, 40960]

    parser.add_argument("--l_MSSSIM", type=int, default=32, choices=[8, 16, 32, 64])
    parser.add_argument("--batch_size", type=int, default=2)  # 8
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--image_size", type=list, default=[256, 256, 3])

    # Dataset preprocess parameters
    parser.add_argument("--dataset_root", type=str, default='')
    parser.add_argument("--frames", type=int, default=5)

    # Optimizer parameters
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--aux_lr', type=float, default=1e-3)
    parser.add_argument('--warmup_iter', type=int, default=-1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.9999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--regular_weight", type=float, default=1e-5)
    parser.add_argument("--clip_max_norm", default=0.5, type=float, help="gradient clipping max norm ")

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4, help='# num_workers')

    return parser.parse_args()


def read_image(filepath):
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def cal_psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def get_timestamp():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)



def fix_random_seed(seed_value=2021):
    os.environ['PYTHONPATHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    # torch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    return 0


def save_weights(name, model, optim, scheduler, root, iteration):
    path = os.path.join(root, "{}_weights.pth".format(name))
    state = dict()
    state["name"] = name
    state["iteration"] = iteration
    state["modelname"] = model.__class__.__name__
    state["model"] = model.state_dict()
    state["optim"] = optim.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    else:
        state["scheduler"] = None
    torch.save(state, path)


def save_model(root, name, model):
    path = os.path.join(root, "{}_all.pth".format(name))
    torch.save(model, path)


def load_state(path, cuda):
    if cuda:
        print("INFO: [*] Load Mode To GPU")
        state = torch.load(path)
    else:
        print("INFO: [*] Load To CPU")
        state = torch.load(path, map_location=lambda storage, loc: storage)
    return state

def rename_key(key):
    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    if ".downsample." in key:
        return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key

def load_pretrained(state_dict):
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    return state_dict


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_codec_settings(model):
    codec_settings = model.get_active_subnet_settings()
    print('mv_codec:',codec_settings["mv_decoder_settings"])
    print('res_codec:',codec_settings["res_decoder_settings"])
    print('mc:',codec_settings["mc_width"])


def get_channel(modules):
    channel_list = []
    for module in modules.modules():
        if isinstance(module, DSConv2d):
            channel_list.append(module.active_out_channel1)
        elif isinstance(module, SubpelDSConv2d):
            channel_list.append(module.sub_activate_out_channel)

    return channel_list
