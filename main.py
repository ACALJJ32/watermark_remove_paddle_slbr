import argparse
import os.path
import random
import time
import datetime
import sys
import glob
import cv2
import os

import numpy as np
import paddle.nn as nn
import paddle
import paddle.distributed as dist
import matplotlib.pylab as plt
from losses import LossNetwork

from transforms import RandomHorizontalFlip, Normalize
from dataset import TrainDataset
from slbr_naf import SLBRNAF, SLBR
from utils import load_pretrained_model
from losses import compute_l1_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    parser.add_argument('--dataset_root',dest='dataset_root',help='The path of dataset root',type=str,\
         default='/disk3/baidu_water_mark/train_datasets')
    parser.add_argument('--batch_size',dest='batch_size',help='batch_size',type=int,default=4)
    parser.add_argument('--max_epochs',dest='max_epochs',help='max_epochs',type=int,default=10)
    parser.add_argument('--log_iters',dest='log_iters',help='log_iters',type=int,default=100)
    parser.add_argument('--save_interval',dest='save_interval',help='save_interval',type=int,default=1)
    parser.add_argument('--sample_interval',dest='sample_interval',help='sample_interval',type=int,default=100)
    parser.add_argument('--with_eval',dest='with_eval',help='with eval',type=bool,default=True)
    parser.add_argument('--seed',dest='seed',help='random seed',type=int,default=24)
    return parser.parse_args()



def main(args):
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dist.init_parallel_env()

    transforms = [RandomHorizontalFlip(), Normalize()]

    dataset = TrainDataset(dataset_root=args.dataset_root, transforms=transforms)
    dataloader = paddle.io.DataLoader(dataset, batch_size=args.batch_size,
                                      num_workers=10, shuffle=True, drop_last=True,
                                      return_list=True)


    # Define model
    generator = SLBR()
    load_pretrained_model(generator, "./model.pdparams")

    generator = paddle.DataParallel(generator)
    generator.train()

    optimizer = paddle.optimizer.Adam(parameters=generator.parameters(), learning_rate=1e-4,
                                      beta1=0.5, beta2=0.999)

    max_psnr = 0.0
    psnr_list = []
    max_psnr_index = 1
    
    iteration = 1

    prev_time = time.time()
    for epoch in range(1, args.max_epochs + 1):
        for i, data_batch in enumerate(dataloader):
            lq, gt, mask = data_batch[0], data_batch[1], data_batch[2] 
            gt_generated, mask_generated, _, refine = generator(lq)
            
            # refine = refine[0]
            #refine = refine * 255.0
            #img_out = paddle.clip(refine, 0, 255)
            #img_out = paddle.transpose(img_out, [1, 2, 0])
            #img_out = img_out.numpy()
            #cv2.imwrite("ttest.png", img_out)
            
            loss = compute_l1_loss(refine, gt)  # refine   gt_generated
            loss.backward()

            optimizer.step()
            generator.clear_gradients()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = args.max_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            iteration += 1
            
            if iteration % 5000:
                paddle.save(generator.state_dict(), './model.pdparams')
            
            if i % args.log_iters == 0:
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %s" %
                                 (epoch, args.max_epochs, i, len(dataloader), loss.numpy()[0], time_left))
            


        if epoch % args.save_interval == 0:
            current_save_dir = os.path.join("train_result", "model", f'epoch_{epoch}')
            if not os.path.exists(current_save_dir):
                try:
                    os.mkdir(current_save_dir)
                except:
                    print("Path already exists.")

            paddle.save(generator.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(current_save_dir, 'model.pdopt'))
        


if __name__ == '__main__':
    args = parse_args()
    main(args)
