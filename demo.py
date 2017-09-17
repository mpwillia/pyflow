# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import os
import pyflow

parser = argparse.ArgumentParser(
    description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
    '-viz', dest='viz', action='store_true',
    help='Visualize (i.e. save) output of flow.')
args = parser.parse_args()

examples_dir = "./examples"
im1 = np.array(Image.open(os.path.join(examples_dir, 'img1.jpg')))
im2 = np.array(Image.open(os.path.join(examples_dir, 'img2.jpg')))
im1 = im1.astype(float) / 255.
im2 = im2.astype(float) / 255.

# Flow Options:
alpha = 0.012 # default 0.012
ratio = 0.75 # default 0.75
minWidth = 20 # default 20
nOuterFPIterations = 7 # default 7
nInnerFPIterations = 1 # default 1
nSORIterations = 30 # default 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
threshold = 0.000005

s = time.time()
u, v, im2W = pyflow.coarse2fine_flow(
    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    nSORIterations, colType, verbose = True, threshold = threshold)
e = time.time()
print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
    e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
flow = np.concatenate((u[..., None], v[..., None]), axis=2)
np.save(os.path.join(examples_dir, 'outFlow.npy'), flow)

always_viz = True
if args.viz or always_viz:
    import cv2
    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(examples_dir, 'outFlow_new.png'), rgb)
    cv2.imwrite(os.path.join(examples_dir, 'img2Warped_new.jpg'), im2W[:, :, ::-1] * 255)
