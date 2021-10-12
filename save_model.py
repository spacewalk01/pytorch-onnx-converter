from __future__ import print_function
import argparse
import torch
from model import DLN
import torchvision.transforms as transforms
import numpy as np
from os.path import join
import time
import math
from lib.dataset import is_image_file
from PIL import Image
from os import listdir
import io
import os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=32, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=False)
parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256, help='0 to use original frame size')
parser.add_argument('--stride', type=int, default=16, help='0 to use original patch size')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--image_dataset', type=str, default='test_img')
parser.add_argument('--model_type', type=str, default='DLN')
parser.add_argument('--output', default='./output/', help='Location to save checkpoint models')
parser.add_argument('--modelfile', default='models/DLN_pretrained.pth', help='sr pretrained base model')
parser.add_argument('--image_based', type=bool, default=True, help='use image or video based ULN')
parser.add_argument('--chop', type=bool, default=False)

opt = parser.parse_args()

gpus_list = range(opt.gpus)
print(opt)

cuda = False
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Building model ', opt.model_type)

model = DLN()

# Serializing DLN Module for C++
model = torch.nn.DataParallel(model, device_ids=gpus_list)
model.load_state_dict(torch.load(
    opt.modelfile,
    map_location=lambda storage, loc: storage))

model.eval()

trans = transforms.ToTensor()

with torch.no_grad():
    LL_in = Image.open("test_img/Input.png").convert('RGB')

    LL = trans(LL_in)
    print(LL.shape)
    LL = LL.unsqueeze(0)
    LL = LL.cuda()

    torch_out = model(LL)

    f = io.BytesIO()

    torch.onnx._export(model.module, (LL), "model.onnx", f, verbose=False, export_type=torch.onnx.ExportTypes.PROTOBUF_FILE)

print("model saved.")