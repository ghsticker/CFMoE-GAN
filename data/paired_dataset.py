import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class PairedDataset(BaseDataset):
    """A dataset class for paired image dataset stored in two folders.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.

    Pairing rule: images are paired by filename, e.g., 'trainA/0001.png' <-> 'trainB/0001.png'.

    During test time, it expects '/path/to/data/testA' and '/path/to/data/testB'.
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        self.B_paths_by_name = {os.path.basename(p): p for p in self.B_paths}

        self.pairs = []
        for a_path in self.A_paths:
            name = os.path.basename(a_path)
            b_path = self.B_paths_by_name.get(name)
            if b_path is None:
                raise FileNotFoundError(
                    f"PairedDataset: cannot find corresponding image in '{self.dir_B}' for A image '{a_path}'. "
                    f"Expected: '{os.path.join(self.dir_B, name)}'"
                )
            self.pairs.append((a_path, b_path))

        assert self.opt.load_size >= self.opt.crop_size
        btoA = self.opt.direction == "BtoA"
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __getitem__(self, index):
        A_path, B_path = self.pairs[index]
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")

        transform_params = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A_img)
        B = B_transform(B_img)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        return len(self.pairs)
