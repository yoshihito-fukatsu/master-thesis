'''
generate images by trained generator
'''

import argparse
import os
import random
import math
import time
import datetime
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchinfo
import numpy as np
import matplotlib.pyplot as plt


### initial setting ###
generate_img_num = 50


num_epochs = 1000
save_epoch = 50 # save interval of training models
batch_size = 64
image_size = 256

nc = 1 # number of channels in the training images
nz = 100 # size of Z latent vector (1 dimensional)
ngf = 64 # size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
dropout = 0.5 # dropout rate in discriminator
momentum = 0.01 # parameter for batchnormalzation "momentum(pytorch)= 1 - momentum(tensorflow)"
eps = 0.001 # parameter for batchnormalzation
lr = 1e-6 # learning rate for Adam optimizers
beta1 = 0.9 # beta1 hyperparameter for Adam optimizers


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            ## input is Z, going into a transposed convolution
            nn.Linear(nz,ngf*8*16*16, bias=False),
            nn.BatchNorm1d(ngf*8*16*16, momentum=momentum, eps=eps),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Unflatten(1, (ngf*8,16,16)),
            ## state size. '(ngf*8) x 16 x 16'
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4, momentum=momentum, eps=eps),
            nn.LeakyReLU(0.3, inplace=True),
            ## state size. '(ngf*4) x 32 x 32'
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2, momentum=momentum, eps=eps),
            nn.LeakyReLU(0.3, inplace=True),
            ## state size. '(ngf*2) x 64 x 64'
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf, momentum=momentum, eps=eps),
            nn.LeakyReLU(0.3, inplace=True),
            ## state size. '(ngf) x128 x 128'
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            ## state size. '(nc) x 256 x 256'
        )

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


checkpoint_netG = './output/'+save_dirname+'/model/netG/netG_final.pth'

os.makedirs(os.path.dirname(checkpoint_netG).replace('/model/netG','')+'/generated_image', exist_ok=True)

with open(os.path.dirname(checkpoint_netG).replace('/model/netG','')+'/generated_image/used_checkpoint_netG.txt', 'w') as f:
    f.write(checkpoint_netG)

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

netG = Generator().to(device)
netG.apply(weights_init)
netG.load_state_dict(torch.load(checkpoint_netG))

noise_for_generate = torch.randn(generate_img_num, nz, device=device) #generate random noise from normal distribution(mean:0, var:1) for visualization of the progression of the generator

noisedata_for_generate = noise_for_generate.cpu().numpy().copy()
np.savetxt(os.path.dirname(checkpoint_netG).replace('/model/netG','')+'/generated_image/noise_for_generate_BS{}_nz{}.csv'.format(generate_img_num,nz), noisedata_for_generate, fmt = "%.18f", delimiter = ",", comments = "")

with torch.no_grad():
    generate_img = netG(noise_for_generate).detach().squeeze().cpu().numpy()

def save_to_grayscale_img(x, savepath):
    x = (x + 1) / 2 # [-1,1] -> [0,1]
    x = 255*x # [0,1] -> [0,255]
    x = x.astype(np.uint8)
    x = Image.fromarray(x)
    return x.save(savepath)

for i in range(generate_img_num):
    savepath = os.path.dirname(checkpoint_netG).replace('/model/netG','')+'/generated_image/generate_image_{0:04}.png'.format(i+1)
    save_to_grayscale_img(generate_img[i], savepath)