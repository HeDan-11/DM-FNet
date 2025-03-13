import torchvision.transforms
import glob

from PIL.Image import Image
from torch.utils.data.dataset import Dataset
from data.util import *
from torchvision.transforms import functional as F
import os
import cv2


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames

def prepare_data_path_hb(dataset_path, dataset2_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))

    data.extend(glob.glob(os.path.join(dataset2_path, "*.tif")))
    data.extend(glob.glob((os.path.join(dataset2_path, "*.jpg"))))
    data.extend(glob.glob((os.path.join(dataset2_path, "*.png"))))
    data.extend(glob.glob((os.path.join(dataset2_path, "*.png"))))

    data.sort()
    filenames.sort()
    return data, filenames

def prepare_data_path_hb2(dataset_path, dataset2_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))

    data.extend(glob.glob(os.path.join(dataset2_path, "*.tif")))
    data.extend(glob.glob((os.path.join(dataset2_path, "*.jpg"))))
    data.extend(glob.glob((os.path.join(dataset2_path, "*.png"))))
    data.extend(glob.glob((os.path.join(dataset2_path, "*.png"))))

    data.sort()
    filenames.sort()
    return data, filenames


class FusionDataset(Dataset):
    def __init__(self,
                 split,
                 crop_size=128,  # resolution in training
                 min_max=(-1, 1),
                 ir_path='/media/sata1/hedan/MSRS-main/test\ir',
                 vi_path='/media/sata1/hedan/MSRS-main/test/vi',
                 is_crop=True):
        super(FusionDataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.is_crop = is_crop
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor=1)
        self.hr_transform = train_hr_transform(crop_size)
        self.vis_ir_transform = train_vis_ir_transform()  # transform from rgb to grayscale
        self.min_max = min_max
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.1)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.1)

        self.new_size_test = (480, 480)

        if split == 'train':
            data_dir_vis = vi_path
            data_dir_ir = ir_path

            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'val':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            prepare_data_path_hb2 (data_dir_vis, data_dir_ir)
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split == 'train':
            visible_image = Image.open(self.filepath_vis[index]).convert ('L')
            infrared_image = Image.open(self.filepath_ir[index]).convert ('L')

            rgb_img = cv2.imread(visible_image)
            rgb_to_yuv = ToTensor()(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb))

            if self.is_crop:
                crop_size = self.hr_transform(visible_image)
                visible_image, infrared_image = F.crop(visible_image, crop_size[0], crop_size[1], crop_size[2],
                                                   crop_size[3]), \
                                                F.crop(infrared_image, crop_size[0], crop_size[1], crop_size[2], crop_size[3])
            # Random horizontal flipping
            if random.random() > 0.5:
                visible_image = self.hflip(visible_image)
                infrared_image = self.hflip(infrared_image)
            # Random vertical flipping
            if random.random() > 0.5:
                visible_image = self.vflip(visible_image)
                infrared_image = self.vflip(infrared_image)

            visible_image = ToTensor()(visible_image)*(self.min_max[1] - self.min_max[0]) + self.min_max[0]
            infrared_image = ToTensor()(infrared_image)*(self.min_max[1] - self.min_max[0]) + self.min_max[0]

            cat_img = torch.cat([visible_image, infrared_image[0:1, :, :]], axis=0)

            return {'img': cat_img, 'vis': visible_image, 'ir': infrared_image[0:1, :, :], 'yuv': rgb_to_yuv},  self.filenames_vis[index]

        elif self.split == 'val':
            visible_image = Image.open(self.filepath_vis[index]).convert ('L')
            # visible_image = cv2.imread (self.filepath_vis[index])
            # visible_image = cv2.cvtColor (visible_image, cv2.COLOR_BGR2RGB)
            infrared_image = Image.open(self.filepath_ir[index]).convert ('L')
            #
            visible_image = ToTensor()(visible_image)
            visible_image = visible_image*(self.min_max[1] - self.min_max[0]) + self.min_max[0]
            infrared_image = ToTensor()(infrared_image)
            infrared_image = infrared_image * (self.min_max[1] - self.min_max[0]) + self.min_max[0]

            cat_img = torch.cat([visible_image[:, :, :], infrared_image[0:1, :, :]], axis=0)

            return {'img': cat_img, 'vis': visible_image, 'ir': infrared_image[0:1, :, :]}, self.filenames_vis[index]

    def __len__(self):
        return self.length
