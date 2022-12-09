import argparse
import os
import sys

import numpy as np
import json
from PIL import Image
import torch

from swin.hist_utils import augment_data
from swin.mlia_swin_transformer import SwinUNETR
from swin.train import train, validation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',action='store_true',help='Run training on the network')
    parser.add_argument('--inference',action='store_true',help='Run inference on the network')
    parser.add_argument('--net-cfg',required=True,help='Network configuration')
    parser.add_argument('--input',required=True,help='Input directory of images')
    parser.add_argument('--output',default='/scratch/ejg8qa/network_out',help='output directory for results and weights')
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


def dataloader(x_dir,y_dir,batch_size=1):
    """
    Loads images in directory and formats them into a list of (b, c, h, w)
    Output arrays are guaranteed to be 4D.

    :param directory: input data directory
    :param batch_size: number of images per batch
    :returns: list of 4D numpy array
    """
    x_files = [f for f in os.listdir(x_dir) if f.endswith('.png')]
    if y_dir:
        y_files = [f for f in os.listdir(y_dir) if f.endswith('.png')]
    num_imgs = len(x_files)
    #assert num_imgs % batch_size == 0, f'number of images ({num_imgs}) is not divisible by batch size ({batch_size})'
    leftover_batch_size = num_imgs % batch_size
    full_batches = num_imgs // batch_size

    full_data = []
    for batch in range(full_batches - 1):
        b_start = batch*batch_size
        b_end = (batch+1)*batch_size
        files_x = x_files[b_start:b_end]
        cur_x = []
        x_names = []
        cur_y = []
        y_names = []
        for img in files_x:
            fp = os.path.join(x_dir,img)
            cur_x.append(np.asarray(Image.open(fp).resize((256,256))))
            x_names.append(img)
            if y_dir:
                idx = img[2:]
                y_file = f'mask{idx}'
                fpy = os.path.join(y_dir,y_file)
                cur_y.append(np.asarray(Image.open(fpy).resize((256,256))))
                y_names.append(y_file)
        batch_x = np.expand_dims(np.stack(cur_x),axis=1)
        batch_y = np.expand_dims(np.stack(cur_y),axis=1)

        full_data.append({"image":np.stack(batch_x),"label":np.stack(batch_y),"xnames":x_names,"ynames":y_names})

    if leftover_batch_size:
        files_x = x_files[full_batches*20:]
        cur_x = []
        cur_y = []
        x_names = []
        y_names = []
        for img in files_x:
            fp = os.path.join(x_dir,img)
            cur_x.append(np.asarray(Image.open(fp).resize((256,256))))
            x_names.append(img)
            if y_dir:
                idx = img[2:]
                y_file = f'mask{idx}'
                fpy = os.path.join(y_dir,y_file)
                cur_y.append(np.asarray(Image.open(fpy).resize((256,256))))
                y_names.append(y_file)

        batch_x = np.expand_dims(np.stack(cur_x),axis=1)
        batch_y = np.expand_dims(np.stack(cur_y),axis=1)
        full_data.append({"image":np.stack(batch_x),"label":np.stack(batch_y),"xnames":x_names,"ynames":y_names})

    return full_data


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


def train_network(config,input_dir,output_dir):
    model = SwinUNETR(**config['swin']).to(DEVICE)

    hyp = config['hyperparams']
    global_step = 0
    dice_val_best = 0
    train_X_location = os.path.join(input_dir,hyp['X_data_folder'])
    train_Y_location = os.path.join(input_dir,hyp['Y_data_folder'])
    data = dataloader(train_X_location,train_Y_location,batch=hyp['batch_size'])
    train_data = data[:-2]
    val_data = data[-2:]
    while global_step < hyp['max_iterations']:
        global_step, dice_val_best, global_step_best = \
            train(global_step, train_data, val_data, model, dice_val_best=0, global_step_best=0,
            device=DEVICE, output_dir='/scratch/ejg8qa/RESULTS'
        )
    
    result_log = os.path.join(output_dir,'final_result.txt')
    with open(result_log,'w') as f:
        f.write(f'Final results:\nBest Dice: {dice_val_best}\nGlobal step:{global_step_best}')


def main(config_filepath,train,inference,input_dir):
    config = load_model_config(config_filepath)
    if args.train:
        training_dir = os.path.join(os.getcwd(), '..', '..', 'data', 'Training')
        augment(training_dir, multiplier=6)

        train_network(config,input_dir)
    elif args.inference:
        raise NotImplementedError
        #TODO: implement this for testing...
        #run_inference(config,input)


if __name__ =='__main__':
    args = load_args()
    main(args.net_cfg,args.train,args.inference,args.input,args.output)
