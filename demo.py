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
    
    if args.save_flow:
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
        # inputs of the net are 256/512/1024...
        img1_in = imresize(img1,(512,512))
        img2_in = imresize(img2,(512,512))

        images = [img1_in, img2_in]
        images = np.array(images).transpose(3,0,1,2)
        images = torch.from_numpy(images.astype(np.float32))
        images = torch.unsqueeze(images, 0)
        images = [images]

        if args.cuda:
            data = [d.cuda() for d in images]
        data = [Variable(d) for d in data]

        with torch.no_grad():
            output = model(data[0])

        if args.save_flow:
            _pflow = output[0].data.cpu().numpy().transpose(1, 2, 0)
            frame_name = input_image_list[i].split('/')[-1]
            flow_path = join(flow_folder, '{}.flo'.format(frame_name))
            print("flow saved as: ", flow_path)
            flow_utils.writeFlow(flow_path,  _pflow)

            if args.save_img:
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
    
    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')
    args = parser.parse_args()
    if args.number_gpus < 0 : args.number_gpus = torch.cuda.device_count()
    
    kwargs = tools.kwargs_from_args(args, 'model')
    model = tools.module_to_dict(models)[args.model]
    model = model(args, **kwargs)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print('Initializing CUDA')
    model = model.cuda()

    print("Loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))

    print("Initializing save directory: {}".format(args.save))
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    stats = inference(args=args, model=model)
    print("Demo test done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument("--rgb_max", type=float, default = 255.)
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')
    parser.add_argument('--save_img', action='store_true', help='illustrate the predicted flows')
    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--input_dir', type=str)

    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)

    demo(parser)