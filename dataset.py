import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

#BPG
class VimeoDataset(Dataset):
    def __init__(self, root, model_type='PSNR', transform=None, split="train", QP=None, Level=None):
        assert split == 'train' or 'test'
        assert model_type == 'PSNR' or 'MSSSIM'
        if transform is None:
            raise Exception("Transform must be applied")
        if (model_type == 'PSNR' and QP is None) or (model_type == 'MSSSIM' and Level is None):
            raise Exception("QP or Level must be specified")

        self.max_frames = 5  # for Vimeo DataSet
        self.QP = QP
        self.Level = Level
        self.transform = transform
        self.model_type = model_type

        self.file_name_list = os.path.join(root, f'sep_{split}list.txt')
        self.frames_dir = [os.path.join(root, 'sequences', x.strip())
                           for x in open(self.file_name_list, "r").readlines()]
        self.ref_frame_dir = [os.path.join('/media/user/f126fd00-4370-4fdf-9a0a-fc93e66106eb1/FWZ/data/train/img',
                                           'sequences', x.strip())
                              for x in open(self.file_name_list, "r").readlines()]

    def __getitem__(self, index):
        sample_folder = self.frames_dir[index]
        ref_folder = self.ref_frame_dir[index]
        frame_paths = []
        for i in range(self.max_frames):
            if i == 0:
                if self.model_type == "PSNR":
                    # frame_paths.append(os.path.join(ref_folder, f'im1_CA_6.png'))
                    frame_paths.append(os.path.join(sample_folder.replace('sequences', 'bpg'), f'im1_bpg444_QP{self.QP}.png'))
                elif self.model_type == "MSSSIM":
                    # frame_paths.append(os.path.join(ref_folder, f'im1_CA_6.png'))
                    frame_paths.append(os.path.join(sample_folder, 'CA_Model', f'im1_level{self.Level}_ssim.png'))
            else:
                frame_paths.append(os.path.join(sample_folder, f'im{i + 1}.png'))

        frames = np.concatenate(
            [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1
        )
        frames = self.transform(frames)
        frames = torch.chunk(frames, chunks=self.max_frames, dim=0)
        return frames

    def __len__(self):
        return len(self.frames_dir)

def get_dataset(args, part='all'):
    QP, I_level = 0, 0
    if args.l_PSNR == 256:
        QP = 37
    elif args.l_PSNR == 512:
        QP = 32
    elif args.l_PSNR == 1024:
        QP = 27
    elif args.l_PSNR >= 2048:
        QP = 22

    if args.l_MSSSIM == 8:
        I_level = 2
    elif args.l_MSSSIM == 16:
        I_level = 3
    elif args.l_MSSSIM == 32:
        I_level = 5
    elif args.l_MSSSIM == 64:
        I_level = 7

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(args.image_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(args.image_size[0]),
        ]
    )

    training_set = VimeoDataset(root=args.dataset_root,
                                model_type=args.mode_type,
                                transform=train_transforms,
                                split="train",
                                QP=QP,
                                Level=I_level,
                                )
    valid_set = VimeoDataset(root=args.dataset_root,
                             model_type=args.mode_type,
                             transform=test_transforms,
                             split="test",
                             QP=QP,
                             Level=I_level,
                             )
    if part == 'all':
        return training_set, valid_set
    elif part == 'train':
        return training_set
    elif part == 'valid':
        return valid_set

#ORG
class VimeoDataset11(Dataset):
    def __init__(self, root, model_type='PSNR', transform=None, split="train", QP=None,
                 Level=None, mf=5, return_orgi=False, worg=False):
        assert split == 'train' or 'test'
        assert model_type == 'PSNR' or 'MSSSIM'
        if transform is None:
            raise Exception("Transform must be applied")
        if (model_type == 'PSNR' and QP is None) or (model_type == 'MSSSIM' and Level is None):
            raise Exception("QP or Level must be specified")

        self.max_frames = mf  # for Vimeo DataSet
        self.return_orgi = return_orgi
        self.QP = QP
        self.Level = Level
        self.transform = transform
        self.model_type = model_type
        self.worg = worg

        self.file_name_list = os.path.join(root, f'sep_{split}list.txt')
        self.frames_dir = [os.path.join(root, 'sequences', x.strip())
                           for x in open(self.file_name_list, "r").readlines()]

    def __getitem__(self, index):
        sample_folder = self.frames_dir[index]
        frame_paths = []
        for i in range(1, self.max_frames + 1):
            frame_paths.append(os.path.join(sample_folder, f'im{i}.png'))
        frames = np.concatenate(
            [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1
        )
        # print(frame_paths)
        frames = self.transform(frames)

        nn = self.max_frames if not self.worg else self.max_frames + 1
        frames = torch.chunk(frames, chunks=nn,  dim=0)

        return frames

    def __len__(self):
        return len(self.frames_dir)

def get_dataset11(args, part='all', mf=5, return_orgi=False, crop=True, worgi=False):
    QP, I_level = 0, 0
    if args.l_PSNR == 256:
        QP = 37
    elif args.l_PSNR == 512:
        QP = 32
    elif args.l_PSNR == 1024:
        QP = 27
    elif args.l_PSNR == 2048:
        QP = 22

    if args.l_MSSSIM == 8:
        I_level = 2
    elif args.l_MSSSIM == 16:
        I_level = 3
    elif args.l_MSSSIM == 32:
        I_level = 5
    elif args.l_MSSSIM == 64:
        I_level = 7
    if crop:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(args.image_size[0]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(args.image_size[0]),
            ]
        )
    else:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    training_set = VimeoDataset11(root=args.dataset_root,
                                model_type=args.mode_type,
                                transform=train_transforms,
                                split="train",
                                QP=QP,
                                Level=I_level,
                                mf=mf,
                                return_orgi=return_orgi,
                                worg=worgi,
                                )
    valid_set = VimeoDataset11(root=args.dataset_root,
                             model_type=args.mode_type,
                             transform=test_transforms,
                             split="test",
                             QP=QP,
                             Level=I_level,
                             mf=mf,
                             return_orgi=return_orgi,
                             worg=worgi,
                             )
    if part == 'all':
        return training_set, valid_set
    elif part == 'train':
        return training_set
    elif part == 'valid':
        return valid_set

if __name__ == "__main__":
    pass
