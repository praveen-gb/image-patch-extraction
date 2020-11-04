# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 16:37:31 2020

@author: n54961pp
"""
import os
import argparse
from skimage import io
from skimage.color import rgb2gray
from patch_extraction_stitching import patch_extraction_stitching

config = dict()
config["frame_x"] = int(os.getenv('FRAME_X', 256))  # patch extaction size
config["frame_y"] = int(os.getenv('FRAME_Y', 256))  # patch extaction size

parser = argparse.ArgumentParser(description=' end to end system testing')

for key, value in config.items():
    t = type(value)
    if t is list or t is tuple:
        parser.add_argument('--' + key, nargs='+',
                            default=value, type=type(value[0]),
                            help="path to testing images")
    else:
        parser.add_argument('--' + key,
                            default=value, type=t,
                            help="path to testing images")

options = parser.parse_args()


def main():
    output_dir = os.path.join('output/')
    os.makedirs(output_dir, exist_ok=True)
    config['output_dir'] = os.getenv('OUTPUT_DIR', output_dir)

    input_image = io.imread('input_image.png').astype('float')  # reading input image from the file
    grayscale_image = rgb2gray(input_image)

    patch_extraction_stitching(grayscale_image, config, options)


if __name__ == '__main__':
    main()
