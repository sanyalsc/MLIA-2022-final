import argparse
import sys

import json

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


def dataloader(directory,batch_size=1):
    """
    Loads images in directory and formats them into a list of (b, c, h, w)
    Output arrays are guaranteed to be 4D.
    
    :output numpy array?"""
    #TODO: implement.
    raise NotImplementedError


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
        train_network(config,input_dir)
    elif args.inference:
        raise NotImplementedError
        #TODO: implement this for testing...
        #run_inference(config,input)


if __name__ =='__main__':
    args = load_args()
    main(args.net_cfg,args.train,args.inference,args.input)
