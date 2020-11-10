import glob
import torch

from imageio import imread
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, img_size=32, mode='train', vin=False):
        self.img_size = img_size
        if not vin:
            self.inp_files = sorted(glob.glob('%s/*_img.png' % root), key=lambda fname: fname[:-8])
            # self.inp_files = sorted(glob.glob('%s/*[0-9].png' % root), key=lambda fname: fname[:-4])
            self.out_files = sorted(glob.glob('%s/*_log.png' % root), key=lambda fname: fname[:-8])

        # print(len(self.inp_files), len(self.out_files))
        # print(self.inp_files[:10])
        # print(self.out_files[:10])
        # print('\n')
        # print(self.inp_files[-10:])
        # print(self.out_files[-10:])
        if mode == 'train':
            if not vin:
                train_split = int(len(self.inp_files) * 0.7)

                self.inp_files = self.inp_files[:train_split] if mode == 'train' else self.inp_files[train_split:]
                self.out_files = self.out_files[:train_split] if mode == 'train' else self.out_files[train_split:]
            else:
                self.inp_files = glob.glob('%s/trainingset_' % root + str(img_size // 2) + '*[0-9].png')
                self.out_files = glob.glob('%s/trainingset_' % root + str(img_size // 2) + '*[0-9]_log.png')
        elif mode != 'eval':
            if not vin:
                self.inp_files = self.inp_files[-5000:]
                self.out_files = self.out_files[-5000:]
            else:
                self.inp_files = glob.glob('%s/validationset_' % root + str(img_size // 2) + '*[0-9].png')
                self.out_files = glob.glob('%s/validationset_' % root + str(img_size // 2) + '*[0-9]_log.png')
        else:
            if not vin:
                self.inp_files = self.inp_files[-5000:]
                self.out_files = self.out_files[-5000:]
            else:
                self.inp_files = glob.glob('%s/evaluationset_' % root + str(img_size // 2) + '*[0-9].png')
                self.out_files = glob.glob('%s/evaluationset_' % root + str(img_size // 2) + '*[0-9]_log.png')

        # print(len(self.inp_files), len(self.out_files))
        # print(self.inp_files[:5])
        # print(self.out_files[:5])
        # print('\n')
        # print(self.inp_files[-5:])
        # print(self.out_files[-5:])

    def __getitem__(self, index):

        inp_img = imread(self.inp_files[index % len(self.inp_files)])
        out_img = imread(self.out_files[index % len(self.out_files)])

        inp_img = (torch.from_numpy(inp_img).type(torch.FloatTensor))
        out_img = (torch.from_numpy(out_img).type(torch.FloatTensor))

        mask = torch.where(inp_img == 0, torch.ones_like(inp_img), torch.zeros_like(inp_img))

        return inp_img.view(1, self.img_size, self.img_size), out_img.view(1, self.img_size, self.img_size), mask.view(1, self.img_size, self.img_size)

    def __len__(self):
        return len(self.inp_files)
