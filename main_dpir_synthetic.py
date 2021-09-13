import os.path
import cv2
import logging

import numpy as np
from collections import OrderedDict

import torch
from torch import optim
import torch.nn.functional as F

from utils import utils_model
from utils import utils_mosaic
from utils import utils_logger
from utils import utils_pnp as pnp
from utils import utils_image as util
from utils import utils_synthetic

from scipy import sparse
from scipy.sparse.linalg import lsqr
import cvxpy as cp
from cvxpy.atoms.affine.reshape import reshape as cp_reshape
from cvxpy.atoms.affine.vstack import vstack as cp_vstack
from cvxpy.expressions.expression import Expression
from cvxpy.atoms.affine.sum import sum as cp_sum
from cvxpy.atoms.norm import norm as cp_norm
import pylops


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

    noise_level_img = 2.0/255.0            # set AWGN noise level for LR image, default: 0
    noise_level_model = noise_level_img  # set noise level of model, default: 0
    model_name = 'drunet_gray'           # set denoiser, 'drunet_color' | 'ircnn_color'
    testset_name = 'set3c'               # set testing set,  'set18' | 'set24'
    x8 = True                            # set PGSE to boost performance, default: True
    iter_num = 8                        # set number of iterations, default: 40 for demosaicing
    modelSigma1 = 10                     # set sigma_1, default: 49
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
        # col_idx = np.linspace(0, width_ - 1, num=width_)
        # row_idx = np.linspace(0, height_ - 1, num=height_)
        # mx, my = np.meshgrid(col_idx, row_idx)
        # indices = np.zeros((1, 2, height_, width_))
        # indices[:, 0, :, :] = my
        # indices[:, 1, :, :] = mx
        # indices = torch.from_numpy(indices).float().to(device)
        # warped_y = indices[:, 0:1, :, :]
        # warped_x = indices[:, 1:2, :, :]

        warped_y = y_cord.reshape(1, height_, width_)
        warped_x = x_cord.reshape(1, height_, width_)
        warped_y = 2 * warped_y / (height_ - 1) - 1
        warped_x = 2 * warped_x / (width_ - 1) - 1
        grid_y = torch.from_numpy(warped_y).float().unsqueeze(0)
        grid_x = torch.from_numpy(warped_x).float().unsqueeze(0)
        #grid_pos = torch.cat([warped_x, warped_y], dim=1).permute(0, 2, 3, 1)
        grid_pos = torch.cat([grid_x, grid_y], dim=1).permute(0, 2, 3, 1).to(device)
        H_mat, H_mat_T = utils_synthetic.calc_A_T(flow_y, flow_x, smooth=False)
        b = H_mat @ img_H.ravel()
        # iwe_abs = np.abs(img_iwe)
        # ret, binary_img = cv2.threshold(iwe_abs, 0.0, 1, cv2.THRESH_BINARY)
        # binary_mask = sparse.diags((binary_img).ravel())
        # H_mat = binary_mask @ H_mat
        # b_tmp = b / 255.0
        # x_init = np.ones(height_ * width_)
        # x_best, istop, itn, r1norm = lsqr(H_mat, b_tmp, x0=x_init, atol=1e-6, btol=1e-6, iter_lim=400, show=True)[:4]
        # util.imshow(x_best.reshape(1, height_, width_),cbar=True)
        # b = cv2.normalize(b, None, 0.0, 255.0, cv2.NORM_MINMAX)
        img_L = b.reshape(height_, width_, 1)
        util.imshow(utils_synthetic.padding(b.reshape(height_, width_), 20), cbar=True)
        img_tmp = torch.from_numpy(img_H.reshape(1, 1, height_, width_).astype(np.float32)).to(device)
        warped_img = img_tmp - F.grid_sample(img_tmp, grid_pos, mode="bilinear", padding_mode="zeros", align_corners = True)
        util.imshow(warped_img.data.squeeze().float().cpu().numpy(), cbar=True)
        util.imshow(warped_img.data.squeeze().float().cpu().numpy() - b.reshape(1, height_, width_), cbar=True)
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
        img_L = img_iwe * center_mask / flow_mag * 2e3
        # U = cp.Variable(shape=(height_, width_))
        # U_mtx = Expression.cast_to_const(U)
        # U_ = cp_reshape(U, (height_*width_, 1), order='C')
        # diffs = [
        #     U_mtx[0:height_-1, 1:width_] - U_mtx[0:height_-1, 0:width_-1],
        #     U_mtx[1:height_, 0:width_-1] - U_mtx[0:height_-1, 0:width_-1]
        # ]
        # length = diffs[0].shape[0]*diffs[1].shape[1]
        # stacked = cp_vstack([cp_reshape(diff, (1, length)) for diff in diffs])
        # cost = cp.norm2(H_mat @ U_ - b.reshape(-1, 1) / 255.0)**2 + 0.003 * cp_sum(cp_norm(stacked, p=2, axis=0))
        # obj = cp.Minimize(cost)
        # prob = cp.Problem(obj)
        # prob.solve(verbose=True, eps = 1e-3, solver=cp.SCS, max_iters=100, warm_start=True)
        # img_init = U.value
        # util.imshow(img_init.reshape(1, height_, width_),cbar=True)
        img_init = np.ones_like(img_L) / 2.0
        Op = pylops.MatrixMult(H_mat)
        Dop = [pylops.FirstDerivative(height_ * width_, dims=(height_, width_), dir=0, edge=False),
               pylops.FirstDerivative(height_ * width_, dims=(height_, width_), dir=1, edge=False)]
        Dop2 = [pylops.SecondDerivative(height_ * width_, dims=(height_, width_), dir=0, edge=False),
               pylops.SecondDerivative(height_ * width_, dims=(height_, width_), dir=1, edge=False)]
        x_best = pylops.optimization.sparsity.SplitBregman(Op, Dop,
                                                        img_L.ravel() / 255.0,
                                                        niter_outer=10, niter_inner=5,
                                                        mu=1.5, epsRL1s=[6e-2, 6e-2],
                                                        tol=1e-4, tau=1., show=True,
                                                        x0=img_init.ravel(),
                                                        **dict(iter_lim=10, damp=1e-4))[0]
        # x_best, istop, itn, r1norm = lsqr(H_mat, img_L.ravel() / 255.0, x0=img_init.ravel(), atol=1e-6, btol=1e-6, iter_lim=400, show=True)[:4]
        img_init = x_best.reshape(height_, width_, 1)
        util.imshow(x_best.reshape(1, height_, width_),cbar=True)
        x = util.uint2tensor4(img_init).to(device) * 255.0
        # warped_x = F.grid_sample(x, grid_pos, mode="bilinear", padding_mode="zeros", align_corners = True)
        # img_res = x - warped_x - torch.from_numpy(img_L.reshape(1, height_, width_) / 255.0).float().unsqueeze(0).to(device)
        # util.imshow(img_res.data.squeeze().float().cpu().numpy(), cbar=True)
        # x = torch.from_numpy(np.ascontiguousarray(img_init)).permute(2, 0, 1).float().unsqueeze(0).to(device)
        z = x.clone()
        y = util.uint2tensor4(img_L).to(device)
        # y = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().unsqueeze(0).to(device)
        #mask = util.single2tensor4(mask.astype(np.float32)).to(device)

        # --------------------------------
        # (3) get rhos and sigmas
        # --------------------------------

        rhos_np, sigmas_np = pnp.get_rho_sigma(sigma=max(0.255/255., noise_level_img) * 1.2, iter_num=iter_num, modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
        rhos, sigmas = torch.tensor(rhos_np).to(device), torch.tensor(sigmas_np).to(device)

        # --------------------------------
        # (4) main iterations
        # --------------------------------
        lr = 0.05
        lr_step = 200
        grad_des_iters = 200
        lr_decay = 0.1
        loss = torch.nn.MSELoss()
    
        for i in range(iter_num):

            # --------------------------------
            # step 1, closed-form solution
            # --------------------------------
            z_k = z.data.squeeze().float().cpu().numpy().ravel()
            # #z_k = cv2.normalize(z_k, None, 0.0, 1.0, cv2.NORM_MINMAX).ravel()
            util.imshow(z_k.reshape(1, height_, width_), cbar=True)
            # z_k_img = cv2.normalize(z_k.reshape(height_, width_), None, 0, 255, cv2.NORM_MINMAX)
            # z_k_img = z_k_img.astype(np.uint8)
            # folder_name = os.path.join(E_path, "z_ks")
            # if not os.path.exists(folder_name):
            #     os.makedirs(folder_name)
            # z_k_name = "%04d.png" % i
            # cv2.imwrite(os.path.join(folder_name, z_k_name), z_k_img)
            y_k = y.data.squeeze().float().cpu().numpy().ravel()
            y_k = img_L.ravel() / 255.0
            # y_k = img_L.ravel()
            A_mat = H_mat.T @ H_mat + sparse.eye(height_ * width_) * rhos_np[i]
            b_vec = (H_mat.T @ y_k + rhos_np[i] * z_k)
            Op = pylops.MatrixMult(A_mat)
            x_best = pylops.optimization.sparsity.SplitBregman(Op, Dop2,
                                                        b_vec,
                                                        niter_outer=10, niter_inner=5,
                                                        mu=1.5, epsRL1s=[1e-3, 1e-3],
                                                        tol=1e-4, tau=1., show=False,
                                                        x0=img_init.ravel(),
                                                        **dict(iter_lim=5, damp=1e-4))[0]
            #x_best, istop, itn, r1norm = lsqr(A_mat, b_vec, x0=z_k, atol=1e-6, btol=1e-6, iter_lim=100)[:4]
            #x_best = cv2.normalize(x_best, None, 0.0, 1.0, cv2.NORM_MINMAX).ravel()
            #x_best = x_best - np.amin(x_best)
            util.imshow(x_best.reshape(1, height_, width_),cbar=True)
            # x_k_img = cv2.normalize(x_best.reshape(height_, width_), None, 0, 255, cv2.NORM_MINMAX)
            # x_k_img = x_k_img.astype(np.uint8)
            # folder_name = os.path.join(E_path, "x_ks")
            # if not os.path.exists(folder_name):
            #     os.makedirs(folder_name)
            # x_k_name = "%04d.png" % i
            # cv2.imwrite(os.path.join(folder_name, x_k_name), x_k_img)
            x = torch.from_numpy(np.ascontiguousarray(x_best.reshape(height_, width_, 1))).permute(2, 0, 1).float().unsqueeze(0).to(device)

            #x = (y+rhos[i].float()*z).div(mask+rhos[i])
            # first_term = y - x * mask
            # second_term = x - z_hat
            # x_k = x.clone().detach()
            # -------------------------------
            # x.requires_grad = True
            # optimizer = optim.Adam([x], lr=0.03)
            # # util.imshow(y.data.squeeze().float().cpu().numpy(), cbar=True)
            # #scheduler = optim.lr_scheduler.StepLR(optimizer, lr_step, lr_decay)
            # for it in range(grad_des_iters):
            #     optimizer.zero_grad()
            #     # img_dx = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
            #     # img_dy = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
            #     # tv_error = img_dx.sum() + img_dy.sum()
            #     img_dx = torch.square(x[:, :, :-1, :] - x[:, :, 1:, :])
            #     img_dy = torch.square(x[:, :, :, :-1] - x[:, :, :, 1:])
            #     square_error = img_dx.sum() + img_dy.sum()
            #     warped_x = F.grid_sample(x, grid_pos, mode="bilinear", padding_mode="zeros", align_corners = True)
            #     output = loss(y, (x - warped_x)) + rhos[i].float() * loss(x, z) + 5e-7 * square_error
            #     # print(output.item())
            #     try:
            #         output.backward()
            #     except Exception as e:
            #         print(e)
            #     optimizer.step()
            #     #scheduler.step()
            # x = x.detach()
            # x_k = x.data.squeeze().float().cpu().numpy().ravel()
            # util.imshow(x_k.reshape(1, height_, width_), cbar=True)
            # -----------------------------------------------------
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
        util.imshow(img_E.reshape(1, height_, width_),cbar=True)
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
