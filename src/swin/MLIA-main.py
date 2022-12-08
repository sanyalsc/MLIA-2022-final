import argparse
import os
import sys

import numpy as np
import json
from PIL import Image

from swin.hist_utils import augment_data
from swin.mlia_swin_transformer import SwinUNETR

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',action='store_true',help='Run training on the network')
    parser.add_argument('--inference',action='store_true',help='Run inference on the network')
    parser.add_argument('--net-cfg',required=True,help='Network configuration')
    parser.add_argument('--input',required=True,help='Input directory of images')
    return parser.parse_args()


def load_model_config(config_file:str):
    with open(config_file,'r') as cfg:
        return json.load(cfg)



def augment(training_dir, multiplier):
    src_img_dir = os.path.join(training_dir, 'train_imageData')
    src_mask_dir = os.path.join(training_dir, 'train_myocardium_segmentations')

    augment_img_dir = src_img_dir + '_AUGMENTED'
    augment_mask_dir = src_mask_dir + '_AUGMENTED'
    augment_data(src_img_dir, src_mask_dir, augment_img_dir, augment_mask_dir, multiplier)

    return (augment_img_dir, augment_mask_dir)


def dataloader(data_dir, mask_dir, batch_size=1):
    """
    Loads images in directory and formats them into a list of (b, c, h, w)
    Output arrays are guaranteed to be 4D.

    :param data_dir: input data directory
    :param mask_dir: input label mask directory
    :param batch_size: number of images per batch
    :returns: list of dictionaries (k: string "image"/"label", v: 4D numpy array)
    """
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    label_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    assert len(image_files) == len(label_files), 'number of images does not match number of labels'

    num_imgs = len(image_files)
    assert num_imgs % batch_size == 0, f'number of images ({num_imgs}) is not divisible by batch size ({batch_size})'

    data = np.asarray(Image.open(os.path.join(data_dir, image_files[0])))
    h, w = data.shape

    data_list = []
    image_arr = np.empty([batch_size, 1, h, w])
    label_arr = np.empty([batch_size, 1, h, w])
    batch_idx = 0
    for f_idx, filename in enumerate(image_files):
        src_no = int(filename.replace('im', '').replace('.png', ''))
        image_path = os.path.join(data_dir, filename)
        label_path = os.path.join(mask_dir, f'mask{src_no}.png')
        assert os.path.exists(label_path), f'{label_path} does not exist'

        # new batch
        if f_idx % batch_size == 0 and f_idx > 0:
            data_list.append({"image": image_arr, "label": label_arr})
            image_arr = np.empty([batch_size, 1, h, w])
            label_arr = np.empty([batch_size, 1, h, w])
            batch_idx = 0

        image_arr[batch_idx, 0] = np.asarray(Image.open(image_path))
        label_arr[batch_idx, 0] = np.asarray(Image.open(label_path))
        batch_idx += 1

    return data_list


def zero_pad_image(data):
    """Pads the width of the images out to 256x256
    
    :input data - list of (b, c, h, w )
    :output - list of (b, c, h, w)"""
    lastPix = data[-1]
    prevH = lastPix[2]
    prevW = lastPix[3]
    diff = (256 - prevW)/2
    for i in prevH:
        for j in range(diff):
            data = [0,0,0,0] + data + [0,0,0,0]
    lastPix_new = data[-1]
    currentH = lastPix_new[2]
    currentW = lastPix_new[3]
    print("the images have been padded out to", currentH, currentW)
    raise NotImplementedError


def train_network(config,input_dir):
    model = SwinUNETR(**config)
    #TODO: 1) implement training
    #TODO: 2) save trained weights and 


def main(config_filepath,train,inference,input_dir):
    config = load_model_config(config_filepath)
    if args.train:
        training_dir = os.path.join(os.getcwd(), '..', '..', 'data', 'Training')
        image_dir, label_dir = augment(training_dir, multiplier=6)

        #TODO: zero-pad augmented data

        data_dict = dataloader(image_dir, training_dir, batch_size=1)

        train_network(config,input_dir)
    elif args.inference:
        raise NotImplementedError
        #TODO: implement this for testing...
        #run_inference(config,input)


if __name__ =='__main__':
    args = load_args()
    main(args.net_cfg,args.train,args.inference,args.input)
