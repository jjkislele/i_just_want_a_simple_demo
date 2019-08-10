#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import argparse, os, sys, subprocess
import setproctitle, colorama
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *

import models, losses, datasets
from utils import flow_utils, tools
import utils.frame_utils as frame_utils
from flowlib import flow_to_image
from scipy.misc import imsave

# fp32 copy of parameters for update
global param_copy

from scipy.misc import imresize

# Reusable function for inference
def inference(args, model):

    model.eval()
    
    if args.save_flow or args.render_validation:
        flow_folder = "{}".format(args.save)
        if not os.path.exists(flow_folder):
            os.makedirs(flow_folder)
    

    input_image_list = glob(args.input_dir+'*.jpg')
    input_image_list.sort()
    print(args.input_dir, "len: ", len(input_image_list))

    for i in range(0, len(input_image_list)-1, 2):
        print("img1: ", input_image_list[i])
        print("img2: ", input_image_list[i+1])
        img1 = frame_utils.read_gen(input_image_list[i])
        img2 = frame_utils.read_gen(input_image_list[i+1])

        # resize to 512
        img1_in = imresize(img1,(512,512))
        img2_in = imresize(img2,(512,512))

        images = [img1_in, img2_in]
        images = np.array(images).transpose(3,0,1,2)
        images = torch.from_numpy(images.astype(np.float32))
        images = torch.unsqueeze(images, 0)
        images = [images]

        # when ground-truth flows are not available for inference_dataset, 
        # the targets are set to all zeros. thus, losses are actually L1 or L2 norms of compute optical flows, 
        # depending on the type of loss norm passed in
        flow = np.zeros([512,512,2], dtype = np.float32)
        flow = flow.transpose(2,0,1)
        flow = torch.from_numpy(flow.astype(np.float32))
        flow = torch.unsqueeze(flow, 0)
        flow = [flow]

        if args.cuda:
            data, target = [d.cuda() for d in images], [t.cuda() for t in flow]
        data, target = [Variable(d) for d in data], [Variable(t) for t in target]

        with torch.no_grad():
            losses, output = model(data[0], target[0], inference=True)

        _pflow = output[0].data.cpu().numpy().transpose(1, 2, 0)
        frame_name = input_image_list[i].split('/')[-1]
        flow_path = join(flow_folder, '{}.flo'.format(frame_name))
        print(flow_path)
        flow_utils.writeFlow(flow_path,  _pflow)

        # and saved as image
        flow = flow_utils.readFlow(flow_path)
        if not os.path.exists(flow_folder+'_img'):
            os.makedirs(flow_folder+'_img')
        img_path = join(flow_folder+'_img', frame_name)
        print('img saved as: ', img_path)
        img = flow_to_image(flow)
        img = imresize(img, (img1.shape[0],img1.shape[1]))
        imsave(img_path, img)

    return

def demo(parser):
    args = parser.parse_args()
        if args.number_gpus < 0 : args.number_gpus = torch.cuda.device_count()
    model = tools.module_to_dict(models)[args.model]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print('Initializing CUDA')
    model = model.cuda()

    print("Loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model_and_loss.module.model.load_state_dict(checkpoint['state_dict'])
    print("Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))

    print("Initializing save directory: {}".format(args.save))
    if not os.path.exists(args.save):
        os.makedirs(args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--crop_size', type=int, nargs='+', default = [512, 512], help="Spatial dimension to crop training samples for training")
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--input_dir', type=str)

    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)

    # main
    stats = inference(args=args, model=model_and_loss)
    print("Demo test done")