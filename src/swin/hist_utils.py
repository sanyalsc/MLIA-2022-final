from skimage.exposure import match_histograms
import torch
from einops.layers.torch import Rearrange
import os
from PIL import Image, ImageOps
import random


def preprocess(input_batch, reference):
    """Assumes input is (b, c, h, w)"""
    print('running histogram match')
    matched_imgs = []
    device = input_batch.device
    for img in input_batch.cpu().numpy():
        match_img = match_histograms(img, reference, channel_axis=0)
        matched_imgs.append(torch.from_numpy(match_img))
        # TODO: Make sure output of match_histograms is c, h, w !!
    
    result = torch.stack(matched_imgs)
    return result.to(device=device,dtype=torch.float32)


def augment_data(input_data_dir, input_mask_dir, output_data_dir, output_mask_dir, multiplier,
                 rotate_range=(0, 90), translate_range=(-20, 20)):
    """Augment training set of data images and label masks
        :param input_data_dir: original directory of data images
        :param input_mask_dir: original directory of label mask images
        :param output_data_dir: directory of augmented data images
        :param output_mask_dir: directory of augmented label mask images
        :param multiplier: number of augmentations for each original image/mask
        :param rotate_range: tuple of min/max rotation angle (degrees)
        :param translate_range: tuple of min/max translation in x/y direction
    """
    data_files = [f for f in os.listdir(input_data_dir) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(input_mask_dir) if f.endswith('.png')]
    assert len(data_files) == len(mask_files), 'number of images does not match number of masks'

    if not os.path.exists(output_data_dir):
        os.mkdir(output_data_dir)
    if not os.path.exists(output_mask_dir):
        os.mkdir(output_mask_dir)

    aug_no = 1 + len(data_files)
    for filename in data_files:
        src_no = int(filename.replace('im', '').replace('.png', ''))
        data_path = os.path.join(input_data_dir, filename)
        mask_path = os.path.join(input_mask_dir, f'mask{src_no}.png')
        assert os.path.exists(mask_path), f'{mask_path} does not exist'

        data_img = Image.open(data_path)
        mask_img = Image.open(mask_path)

        for i in range(multiplier):
            # random horizontal flip
            if random.randint(0, 1) == 1:
                data_img = ImageOps.mirror(data_img)
                mask_img = ImageOps.mirror(mask_img)

            # random rotate
            angle = random.randint(*rotate_range)
            aug_data = data_img.rotate(angle)
            aug_mask = mask_img.rotate(angle)

            # (x, y) -> (ax + by + c, dx + ey + f)
            a = 1
            b = 0
            c = random.randint(*translate_range)  # left/right
            d = 0
            e = 1
            f = random.randint(*translate_range)  # up/down
            transform = (a, b, c, d, e, f)
            # random translate
            aug_data = aug_data.transform(data_img.size, Image.AFFINE, transform, Image.BILINEAR)
            aug_mask = aug_mask.transform(mask_img.size, Image.AFFINE, transform, Image.BILINEAR)

            aug_data.save(os.path.join(output_data_dir, f'img{aug_no}.png'))
            aug_mask.save(os.path.join(output_mask_dir, f'mask{aug_no}.png'))
            aug_no += 1
