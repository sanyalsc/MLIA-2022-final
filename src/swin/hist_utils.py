from skimage.exposure import match_histograms
import torch
from einops.layers.torch import Rearrange


def preprocess(input_batch, reference):
    """Assumes input is (b, c, h, w)"""
    print('running histogram match')
    matched_imgs = []
    for img in input_batch:
        matched_imgs.append(match_histograms(img,reference,channel_axis=0))
        #TODO: Make sure output of match_histograms is c, h, w !!
    return torch.stack(matched_imgs)