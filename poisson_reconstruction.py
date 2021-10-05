import os.path
import cv2
import logging

import numpy as np
from collections import OrderedDict

import torch
from torch import optim

from utils import utils_model
from utils import utils_mosaic
from utils import utils_logger
from utils import utils_pnp as pnp
from utils import utils_sisr as sr
from utils import utils_image as util

from scipy import sparse
from scipy.sparse.linalg import lsqr
from models.network_unet import UNetRes as net
from scipy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt

noise_level_img = 0.3/255.0         # default: 0, noise level for LR image
noise_level_model = noise_level_img  # noise level of model, default 0
model_name = 'drunet_gray'
x8 = True
sf = 1
n_channels = 1
iter_num = 8                         # number of iterations
modelSigma1 = 49
modelSigma2 = max(0.6, noise_level_model*255.)
model_path = os.path.join('model_zoo', model_name+'.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

mosaic_gradx = cv2.imread('mosaic_gradx.exr', cv2.IMREAD_UNCHANGED)
mosaic_grady = cv2.imread('mosaic_grady.exr', cv2.IMREAD_UNCHANGED)

mosaic_gradx2 = np.roll(mosaic_gradx, -1, axis=1) - mosaic_gradx
mosaic_grady2 = np.roll(mosaic_grady, -1, axis=0) - mosaic_grady
mosaic_lap = (mosaic_gradx2 + mosaic_grady2) / 30.0

laplacian = np.array([[0, 1, 0],
                      [1,-4, 1],
                      [0, 1, 0]])
PSF = np.zeros_like(mosaic_gradx)
PSF[:3,:3] = laplacian
PSF = np.roll(PSF, shift=(-1, -1), axis=(0, 1))
OTF = fft2(PSF)
lap_F = fft2(mosaic_lap)
factor_div = OTF
factor_div = np.where(factor_div == 0.0, 1.0, factor_div)
lap_F2 = lap_F / factor_div
lap_F2[0, 0] = (1024.0 * 2048.0) / 2.0
mosaic_rec = np.real(ifft2(lap_F2))
x = torch.from_numpy(np.ascontiguousarray(mosaic_rec)).float().unsqueeze(0).unsqueeze(0).to(device)
plt.imshow(mosaic_rec, cmap='gray')
plt.colorbar()
plt.show()

model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for _, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255/255., noise_level_model), iter_num=iter_num, modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

img_L_tensor = torch.from_numpy(np.ascontiguousarray(mosaic_lap)).float().unsqueeze(0).unsqueeze(0)
k_tensor = torch.from_numpy(np.ascontiguousarray(laplacian)).float().unsqueeze(0).unsqueeze(0)
[k_tensor, img_L_tensor] = util.todevice([k_tensor, img_L_tensor], device)
FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_tensor, k_tensor, sf)

for i in range(iter_num):

    # --------------------------------
    # step 1, FFT
    # --------------------------------

    tau = rhos[i].float().repeat(1, 1, 1, 1)
    x = sr.data_solution(x, FB, FBC, F2B, FBFy, tau, sf)
    x_k = x.data.squeeze().squeeze().float().cpu().numpy()
    plt.imshow(x_k, cmap='gray')
    plt.colorbar()
    plt.show()

    # --------------------------------
    # step 2, denoiser
    # --------------------------------

    if x8:
        x = util.augment_img_tensor4(x, i % 8)

    x = torch.cat((x, sigmas[i].float().repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
    x = utils_model.test_mode(model, x, mode=2, refield=32, min_size=256, modulo=16)

    if x8:
        if i % 8 == 3 or i % 8 == 5:
            x = util.augment_img_tensor4(x, 8 - i % 8)
        else:
            x = util.augment_img_tensor4(x, i % 8)
    
    x_k = x.data.squeeze().squeeze().float().cpu().numpy()
    plt.imshow(x_k, cmap='gray')
    plt.colorbar()
    plt.show()

img_E = x.data.squeeze().squeeze().float().clamp_(0, 1).cpu().numpy()
plt.imshow(img_E, cmap='gray')
plt.show()