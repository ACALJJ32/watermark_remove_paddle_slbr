import argparse
import glob
import time
import os.path
import os
import paddle
import paddle.nn as nn
import cv2
import shutil

import sys
from slbr_naf import SLBRNAF, SLBR
from utils import load_pretrained_model, chop_forward


def inference(model, img):
    height, width, _ = img.shape
    print(img.shape)

    model.eval()
    img = paddle.to_tensor(img)
    img /= 255.0
    img = paddle.transpose(img, [2, 0, 1])
    img = img.unsqueeze(0)
    
    # img_out, mask, _ = model(img)
    
    #if height * width < 800 * 800:
    #    img_out, _, _ = model(img)
    #else:
    
    img_out = chop_forward(model, img)
    
    img_out = img_out.squeeze(0)
    img_out = img_out * 255.0
    img_out = paddle.clip(img_out, 0, 255)
    img_out = paddle.transpose(img_out, [1, 2, 0])
    img_out = img_out.numpy()
    return img_out


def main(src_image_dir, save_dir):
    # load model
    generator = SLBR()
    load_pretrained_model(generator, "./model.pdparams")
    
    generator.eval()
    im_files = glob.glob(os.path.join(src_image_dir, "*.jpg"))

    for idx, im in enumerate(im_files):
        print("{} | {}".format(idx+1, len(im_files)))
        img = cv2.imread(im)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_out = inference(generator, img)

        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, im.split('/')[-1]), img_out)



if __name__ == '__main__':
    assert len(sys.argv) == 3
    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    main(src_image_dir, save_dir)
   
