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
from utils import utils_image as util
from utils import utils_synthetic

from scipy import sparse
from scipy.sparse.linalg import lsqr

"""
Spyder (Python 3.7)
PyTorch 1.6.0
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/DPIR
        https://github.com/cszn/IRCNN
        https://github.com/cszn/KAIR
@article{zhang2020plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint},
  year={2020}
}
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; homepage: https://cszn.github.io/)
by Kai Zhang (01/August/2020)

# --------------------------------------------
|--model_zoo               # model_zoo
   |--drunet_gray          # model_name, for color images
   |--drunet_color
|--testset                 # testsets
|--results                 # results
# --------------------------------------------

How to run:
step 1: download [drunet_color.pth, ircnn_color.pth] from https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D
step 2: set your own testset 'testset_name' and parameter setting such as 'noise_level_model', 'iter_num'. 
step 3: 'python main_dpir_demosaick.py'

"""

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 0/255.0            # set AWGN noise level for LR image, default: 0
    noise_level_model = noise_level_img  # set noise level of model, default: 0
    model_name = 'drunet_gray'           # set denoiser, 'drunet_color' | 'ircnn_color'
    testset_name = 'set3c'               # set testing set,  'set18' | 'set24'
    x8 = True                            # set PGSE to boost performance, default: True
    iter_num = 60                        # set number of iterations, default: 40 for demosaicing
    modelSigma1 = 49                     # set sigma_1, default: 49
    modelSigma2 = max(0.6, noise_level_model*255.) # set sigma_2, default
    matlab_init = True

    show_img = False                     # default: False
    save_L = True                        # save LR image
    save_E = True                        # save estimated image
    save_LEH = True                     # save zoomed LR, E and H images
    border = 10                          # default 10 for demosaicing

    task_current = 'syn'                  # 'dm' for demosaicing
    n_channels = 1                       # fixed
    model_zoo = 'model_zoo'              # fixed
    testsets = 'testsets'                # fixed
    results = 'results'                  # fixed
    result_name = testset_name + '_' + task_current + '_' + model_name
    model_path = os.path.join(model_zoo, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    if 'drunet' in model_name:
        from models.network_unet import UNetRes as net
        model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
    elif 'ircnn' in model_name:
        from models.network_dncnn import IRCNN as net
        model = net(in_nc=n_channels, out_nc=n_channels, nc=64)
        model25 = torch.load(model_path)
        former_idx = 0

    logger.info('model_name:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, noise_level_img, noise_level_model))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    test_results = OrderedDict()
    test_results['psnr'] = []

    for idx, img in enumerate(L_paths):

        # --------------------------------
        # (1) get img_H and img_L
        # --------------------------------

        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=n_channels)
        util.imshow(img_H)
        height_, width_, channel_ = img_H.shape
        img_iwe = np.load('IWE2.npy')
        util.imshow(img_iwe.reshape(1, height_, width_), cbar=True)
        map_y, map_x = np.mgrid[0:height_, 0:width_]
        map_y = map_y.astype(np.float32)
        map_x = map_x.astype(np.float32)
        fx, fy, px, py = 199.092366542, 198.82882047, 132.192071378, 110.712660011
        dist_co = (-0.368436311798, 0.150947243557, -0.000296130534385, -0.000759431726241, 0.0)
        angular_velocity = (1.9373462, 0.00596814, -0.2673945)
        y_cord, x_cord, flow_mag, flow_rgb = utils_synthetic.calc_dense_flow_rot(height_, width_, fx, fy, px, py, angular_velocity)
        flow_y = y_cord - map_y
        flow_x = x_cord - map_x
        center_mask = np.zeros((height_, width_), dtype=np.float32)
        center_mask[1:-1, 1:-1] = 1.0
        center_mask = center_mask.astype(np.float32)
        flow_y = flow_y * center_mask
        flow_x = flow_x * center_mask
        H_mat, H_mat_T = utils_synthetic.calc_A_T(flow_y, flow_x, smooth=False)
        b = H_mat @ img_H.ravel()
        # b_tmp = b / 255.0
        # x_init = np.ones(height_ * width_)
        # x_best, istop, itn, r1norm = lsqr(H_mat, b_tmp, x0=x_init, atol=1e-6, btol=1e-6, iter_lim=400, show=True)[:4]
        # util.imshow(x_best.reshape(1, height_, width_),cbar=True)
        # b = cv2.normalize(b, None, 0.0, 255.0, cv2.NORM_MINMAX)
        img_L = b.reshape(height_, width_, 1)
        util.imshow(utils_synthetic.padding(b.reshape(height_, width_), 20), cbar=True)
        #mask = np.random.rand(height_, width_, channel_) >= 0.8
        #mask = np.repeat(mask, n_channels, axis=2)
        #img_L = img_H * mask
        #util.imshow(img_L) if show_img else None
        #img_L, mask, mask_float = util.drop_and_noise(img_H, 255 * .01, 0.8)
        #H_mat = sparse.diags((mask_float).ravel())
        # mask = util.imread_uint(M_paths[idx-1], n_channels=n_channels)
        # img_L = img_H * mask

        # --------------------------------
        # (2) initialize x
        # --------------------------------
        #x = util.median_inpainting(img_L, mask)
        # x = util.uint2tensor4(x).to(device)
        # z = x.clone()
        #x = util.uint2tensor4(img_L).to(device)
        #img_L = img_iwe * center_mask# / flow_mag
        img_init = np.zeros_like(img_L)
        x = util.uint2tensor4(img_init).to(device)
        # x = torch.from_numpy(np.ascontiguousarray(img_init)).permute(2, 0, 1).float().unsqueeze(0).to(device)
        z = x.clone()
        y = util.uint2tensor4(img_L).to(device)
        # y = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().unsqueeze(0).to(device)
        #mask = util.single2tensor4(mask.astype(np.float32)).to(device)

        # --------------------------------
        # (3) get rhos and sigmas
        # --------------------------------

        rhos_np, sigmas_np = pnp.get_rho_sigma(sigma=max(0.255/255., noise_level_img), iter_num=iter_num, modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
        rhos, sigmas = torch.tensor(rhos_np).to(device), torch.tensor(sigmas_np).to(device)

        # --------------------------------
        # (4) main iterations
        # --------------------------------
        lr = 0.05
        lr_step = 200
        grad_des_iters = 100
        lr_decay = 0.1
        loss = torch.nn.MSELoss()
    
        for i in range(iter_num):

            # --------------------------------
            # step 1, closed-form solution
            # --------------------------------
            z_k = z.data.squeeze().float().cpu().numpy().ravel()
            #z_k = cv2.normalize(z_k, None, 0.0, 1.0, cv2.NORM_MINMAX).ravel()
            util.imshow(z_k.reshape(1, height_, width_), cbar=True)
            y_k = y.data.squeeze().float().cpu().numpy().ravel()
            y_k = img_L.ravel() / 255.0
            # y_k = img_L.ravel()
            A_mat = H_mat.T @ H_mat + sparse.eye(height_ * width_) * rhos_np[i]
            b_vec = (H_mat.T @ y_k + rhos_np[i] * z_k)
            x_best, istop, itn, r1norm = lsqr(A_mat, b_vec, x0=z_k, atol=1e-6, btol=1e-6, iter_lim=400)[:4]
            #x_best = cv2.normalize(x_best, None, 0.0, 1.0, cv2.NORM_MINMAX).ravel()
            #x_best = x_best - np.amin(x_best)
            util.imshow(x_best.reshape(1, height_, width_),cbar=True)
            x = torch.from_numpy(np.ascontiguousarray(x_best.reshape(height_, width_, 1))).permute(2, 0, 1).float().unsqueeze(0).to(device)

            #x = (y+rhos[i].float()*z).div(mask+rhos[i])
            # first_term = y - x * mask
            # second_term = x - z_hat
            #x_k = x.clone().detach()
            # x.requires_grad = True
            # optimizer = optim.Adam([x], lr=0.03)
            # #scheduler = optim.lr_scheduler.StepLR(optimizer, lr_step, lr_decay)
            # for it in range(grad_des_iters):
            #     optimizer.zero_grad()
            #     output = loss(y, x * mask) + rhos[i].float() * loss(x, z)
            #     try:
            #         output.backward()
            #     except Exception as e:
            #         print(e)
            #     optimizer.step()
            #     #scheduler.step()
            # x = x.detach()
            # --------------------------------
            # step 2, denoiser
            # --------------------------------

            if 'ircnn' in model_name:
                current_idx = np.int(np.ceil(sigmas[i].cpu().numpy()*255./2.)-1)
                if current_idx != former_idx:
                    model.load_state_dict(model25[str(current_idx)], strict=True)
                    model.eval()
                    for _, v in model.named_parameters():
                        v.requires_grad = False
                    model = model.to(device)
                former_idx = current_idx

            #z = torch.clamp(x, 0, 1)
            z = x
            if x8:
                z = util.augment_img_tensor4(z, i % 8)

            if 'drunet' in model_name:
                z = torch.cat((z, sigmas[i].float().repeat(1, 1, z.shape[2], z.shape[3])), dim=1)
                z = utils_model.test_mode(model, z, mode=2, refield=32, min_size=256, modulo=16)
                # z = model(z)
            elif 'ircnn' in model_name:
                z = model(z)

            if x8:
                if i % 8 == 3 or i % 8 == 5:
                    z = util.augment_img_tensor4(z, 8 - i % 8)
                else:
                    z = util.augment_img_tensor4(z, i % 8)
        x = z
        #x[mask.to(torch.bool)] = y[mask.to(torch.bool)]

        # --------------------------------
        # (4) img_E
        # --------------------------------

        img_E = util.tensor2uint(x)
        if n_channels == 1:
            img_L = img_L.squeeze()
            img_H = img_H.squeeze()
        psnr = util.calculate_psnr(img_E, img_H, border=border)
        test_results['psnr'].append(psnr)
        logger.info('{:->4d}--> {:>10s} -- PSNR: {:.2f}dB'.format(idx, img_name+ext, psnr))

        if save_E:
            util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+'.png'))

        if save_L:
            util.imsave(img_L, os.path.join(E_path, img_name+'_L.png'))

        if save_LEH:
            util.imsave(np.concatenate([img_L, img_E, img_H], axis=1), os.path.join(E_path, img_name+model_name+'_LEH.png'))

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    logger.info('------> Average PSNR(RGB) of ({}) is : {:.2f} dB'.format(testset_name,  ave_psnr))


if __name__ == '__main__':

    main()
