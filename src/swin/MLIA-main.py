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


def train_network(config,input_dir):
    model = SwinUNETR(**config)


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
