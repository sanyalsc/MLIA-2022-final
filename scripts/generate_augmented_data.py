import argparse

from swin.hist_utils import augment_data


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-mask',required=True,help='Network configuration')
    parser.add_argument('--input-data',required=True,help='Input directory of images')
    parser.add_argument('--output-mask',required=True,help='Network configuration')
    parser.add_argument('--output-data',required=True,help='Input directory of images')
    parser.add_argument('--multiplier',default=500)
    return parser.parse_args()

if __name__ == '__main__':
    args = load_args()
    augment_data(args.input_data, args.input_mask, args.output_data, args.output_mask, int(args.multiplier))