import argparse
import os
import sys
from time import time

import numpy as np
import json
from PIL import Image
import torch
from torchvision.transforms import transforms
import pdb
from sklearn.model_selection import train_test_split

from swin.hist_utils import augment_data
from swin.mlia_swin_transformer import SwinUNETR
from swin.train import train, validation


from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
)

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


def dataloader(x_dir,y_dir,batch_size=1,ref=None):
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
                y = np.asarray(Image.open(fpy).resize((256,256)))
                cur_y.append(np.where(y > 127,1,0))
                y_names.append(y_file)
        batch_x = np.expand_dims(np.array(cur_x),axis=1)
        batch_y = np.expand_dims(np.array(cur_y),axis=1)
        full_data.append({"image":batch_x,"label":batch_y,"xnames":x_names,"ynames":y_names})

    if leftover_batch_size:
        files_x = x_files[full_batches*batch_size:]
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
                y = np.asarray(Image.open(fpy).resize((256,256)))
                cur_y.append(np.where(y > 127,1,0))
                y_names.append(y_file)

        batch_x = np.expand_dims(np.array(cur_x),axis=1)
        batch_y = np.expand_dims(np.array(cur_y),axis=1)
        full_data.append({"image":batch_x,"label":batch_y,"xnames":x_names,"ynames":y_names})

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


def train_network(config,input_dir,output_dir):
    model = SwinUNETR(**config['swin']).to(DEVICE)

    hyp = config['hyperparams']
    if hyp['weights']:
        weights = torch.load(hyp['weights'])
        model.load_state_dict(weights)
    global_step = 0
    dice_val_best = 0
    ref = Image.open(config['swin']['histogram_matching_reference'])
    ref = np.expand_dims(np.array(ref),axis=0)
    sd = time()
    train_X_location = os.path.join(input_dir,hyp['X_data_folder'])
    train_Y_location = os.path.join(input_dir,hyp['Y_data_folder'])
    data = dataloader(train_X_location,train_Y_location,batch_size=hyp['batch_size'],ref=ref)
    #train_data = data[:-2]
    #val_data = data[-2:]
    train_data, val_data = train_test_split(data,test_size=0.2)
    ed = time()

    print(f'Loaded images in {ed-sd} sec.')
    print(f"Beginning training with {len(train_data)} epochs and {len(val_data)} validation epochs")
    while global_step < hyp['max_iterations']:
        step_start = time()
        global_step, dice_val_best, global_step_best = \
            train(global_step, train_data, val_data, model, dice_val_best=0, global_step_best=0,
            device=DEVICE, output_dir=os.path.join(output_dir,'RESULTS')
        )
        print(f'Step {global_step} finished in {time() - step_start}')
    
    result_log = os.path.join(output_dir,'final_result.txt')
    with open(result_log,'w') as f:
        f.write(f'Final results:\nBest Dice: {dice_val_best}\nGlobal step:{global_step_best}')


def run_inference(config,input_dir, output_dir):

    model = SwinUNETR(**config['swin']).to(DEVICE)

    hyp = config['hyperparams']
    if hyp['weights']:
        weights = torch.load(hyp['weights'],map_location=torch.device('cpu'))
        model.load_state_dict(weights)
    ref = Image.open(config['swin']['histogram_matching_reference'])
    ref = np.expand_dims(np.array(ref),axis=0)
    sd = time()
    X_location = os.path.join(input_dir,hyp['X_data_folder'])
    Y_location = os.path.join(input_dir,hyp['Y_data_folder'])
    data = dataloader(X_location,Y_location,batch_size=1,ref=ref)


    post_label = AsDiscrete(to_onehot=2)
    post_pred = AsDiscrete(argmax=True,to_onehot=2)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    for i, img in enumerate(data):
        name = img['ynames'][0]
        result = model(torch.from_numpy(img['image']))
        bg = result[0,0].detach().numpy()
        detection = result[0,1].detach().numpy()
        im = np.where(bg < detection,1,0)
        seg_image = Image.fromarray(255*im.astype(np.uint8),'L')
        outpath = os.path.join(output_dir,name)
        seg_image.save(outpath)
        
        
def visualize_results(data_dir, mask_dir, vis_dir, color=(245, 84, 66)):
    """
    Creates segmentation result visualization by overlaying label masks on input images

    :param data_dir: input data directory
    :param mask_dir: label mask directory
    :param vis_dir: output directory
    :param color: highlight color for segmentation mask
    """
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    label_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    assert len(image_files) == len(label_files), 'number of images does not match number of labels'

    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    for filename in image_files:
        src_no = int(filename.replace('im', '').replace('.png', ''))
        image_path = os.path.join(data_dir, filename)
        label_path = os.path.join(mask_dir, f'mask{src_no}.png')
        assert os.path.exists(label_path), f'{label_path} does not exist'

        image_arr = np.asarray(Image.open(image_path))
        label_arr = np.asarray(Image.open(label_path))

        vis_arr = np.stack((image_arr,) * 3, axis=2)
        vis_arr[np.where(label_arr > 0)] = color

        vis_img = Image.fromarray(vis_arr, 'RGB')
        vis_img.save(os.path.join(vis_dir, f'vis{src_no}.png'))


def main(config_filepath,train,inference,input_dir, output_dir):
    config = load_model_config(config_filepath)
    if train:
        #training_dir = os.path.join(os.getcwd(), '..', '..', 'data', 'Training')
        augment(input_dir, multiplier=6)

        train_network(config,input_dir,output_dir)
    elif inference:
        run_inference(config,input_dir,output_dir)


if __name__ =='__main__':
    args = load_args()
    main(args.net_cfg,args.train,args.inference,args.input,args.output)
