import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import cv2
import time

from models import *
from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./dataset")
parser.add_argument("--output_path", type=str, default="./results")
parser.add_argument("--img_height", type=int, default=384, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
opt = parser.parse_args()

generator = UnetGenerator(3, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)
discriminator = Discriminator()
generator.load(f"{opt.output_path}/single_gpu/saved_models/generator_180.pkl")
discriminator.load(f"{opt.output_path}/single_gpu/saved_models/discriminator_180.pkl")

transforms = [
    transform.Resize(size=(opt.img_height, opt.img_width)),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

test_dataloader = ImageDataset(opt.data_path, mode="val",transforms=transforms).set_attrs(
    batch_size=10,
    shuffle=False,
    num_workers=1,
)


# @jt.single_process_scope()
def eval():
    os.makedirs(f"{opt.output_path}/images/test_imgs", exist_ok=True)
    for i, (_, real_A, photo_id) in enumerate(test_dataloader):
        fake_B = generator(real_A)

        fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
        for idx in range(fake_B.shape[0]):
            cv2.imwrite(f"{opt.output_path}/images/test_imgs/{photo_id[idx]}.jpg",
                        fake_B[idx].transpose(1, 2, 0)[:, :, ::-1])
if __name__=='__main__':
    eval()
