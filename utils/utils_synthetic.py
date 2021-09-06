import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
from numpy import linalg
import time
import torch
import torch.nn.functional as F
from scipy.sparse import dia_matrix
from scipy import sparse

def myimshow(src):
    dpi = 80
    h, w = src.shape
    fig = plt.figure(figsize=(h/float(dpi), w/float(dpi)), dpi=dpi)
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(src, extent=(0,w,h,0), interpolation=None, cmap='gray')
    ax.set_xticks([])  # remove xticks
    ax.set_yticks([])  # remove yticks
    ax.axis('off')

def minMaxLocRobust(img, percent):
    img_vec = img.ravel()
    img_vec = np.sort(img_vec, axis=None)
    idx_min = (0.5 * percent / 100.0) * len(img_vec)
    idx_max = (1.0 - 0.5 * percent / 100.0) * len(img_vec)
    robust_min = img_vec[int(idx_min)]
    robust_max = img_vec[int(idx_max)]
    print(robust_min, robust_max)
    return robust_min, robust_max

def normalizeRobust(img, percent):
    #The image should change to float datatype before calling this function
    robust_min, robust_max = minMaxLocRobust(img, percent)
    scale = (255.0 / (robust_max - robust_min) if (robust_max != robust_min) else 1.0)
    img_normalized = (img - robust_min) * scale
    #img_normalized.astype(np.uint8)
    return img_normalized

def load_cam_calib(path):
    calib_data = np.loadtxt(f'{path}/calib.txt')
    fx = calib_data[0]
    fy = calib_data[1]
    px = calib_data[2]
    py = calib_data[3]
    dist_co = calib_data[4:]
    return fx, fy, px, py, dist_co

def load_img_by_ref_time(data_path, ref_time):
    imgs = pd.read_csv(f'{data_path}/images.txt', sep=" ", header=None)
    imgs.columns = ["time", "img_name"]
    imgs_time = imgs["time"].to_numpy()
    imgs_name = imgs["img_name"]
    idx = (np.abs(imgs_time - ref_time)).argmin()
    img_time = imgs_time[idx]
    img_name = imgs_name[idx]
    img_raw = cv2.imread(f'{data_path}/{img_name}', cv2.IMREAD_GRAYSCALE)
    return img_time, img_raw

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def visualize_flow(flow_mag, flow_ang):
    img_height, img_width = flow_mag.shape
    hsv = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = (flow_ang + np.pi) * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(flow_mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def calc_dense_flow_rot(img_height, img_width, fx, fy, px, py, angular_velocity):
    map_y, map_x = np.mgrid[0:img_height, 0:img_width]
    map_y = map_y.astype(np.float32)
    map_x = map_x.astype(np.float32)
    cam_cord = np.ones((img_height, img_width, 3))
    v_bar = map_y - py
    u_bar = map_x - px
    v_dot = np.inner(np.dstack((v_bar*v_bar/fy + fy, -u_bar*v_bar/fx, -fy*u_bar/fx)), angular_velocity)
    u_dot = np.inner(np.dstack((u_bar*v_bar/fy, -u_bar*u_bar/fx - fx,  fx*v_bar/fy)), angular_velocity)
    flow_mag, flow_ang = cart2pol(u_dot, v_dot)
    flow_rgb = visualize_flow(flow_mag, flow_ang)
    # plt.imshow(flow_rgb)
    # plt.show()
    np.where(flow_mag > 0, flow_mag, 1.0)
    uni_flow_x, uni_flow_y = pol2cart(np.ones_like(flow_mag), flow_ang)
    pix_cord_y = uni_flow_y + map_y
    pix_cord_x = uni_flow_x + map_x
    # cam_cord[:, :, 1] = (map_y - py) / fy
    # cam_cord[:, :, 0] = (map_x - px) / fx
    # cam_dx = np.cross(angular_velocity, cam_cord)
    # max_dx = np.amax(linalg.norm(cam_dx, axis = 2))
    # # Divide by (max_dx * fx) to assume in a small time-intervel 
    # # so that optical flow of all the pixels are below 1
    # cam_cord = cam_cord + cam_dx / (max_dx * fx)
    # pix_cord_y = cam_cord[:, :, 1] / cam_cord[:, :, 2] * fy + py
    # pix_cord_x = cam_cord[:, :, 0] / cam_cord[:, :, 2] * fx + px
    # flow_mag, flow_ang = cart2pol(pix_cord_x - map_x, pix_cord_y - map_y)
    # flow_rgb = visualize_flow(flow_mag, flow_ang)
    # plt.imshow(flow_rgb)
    # plt.show()
    # flow_mag = np.ones_like(flow_mag)
    border_mask = np.zeros((img_height, img_width))
    border_mask[1:-1, 1:-1] = 1
    pix_cord_y = pix_cord_y * border_mask + map_y * (1 - border_mask)
    pix_cord_x = pix_cord_x * border_mask + map_x * (1 - border_mask)
    return pix_cord_y.astype(np.float32), pix_cord_x.astype(np.float32), flow_mag.astype(np.float32), flow_rgb

def calc_dense_flow_tra(img_height, img_width, linear_velocity):
    map_y, map_x = np.mgrid[0:img_height, 0:img_width]
    map_y = map_y.astype(np.float32)
    map_x = map_x.astype(np.float32)
    abs_velocity = linalg.norm(linear_velocity)
    flow_y = np.ones((img_height, img_width)) * linear_velocity[1] / abs_velocity
    flow_x = np.ones((img_height, img_width)) * linear_velocity[0] / abs_velocity
    map_y -= flow_y
    map_x -= flow_x
    return map_y, map_x

def undo_distortion(src, instrinsic_matrix, distco, width, height):
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(instrinsic_matrix, distco, (width, height), 0, (width, height))
    dst = cv2.undistortPoints(src, instrinsic_matrix, distco, None, newcameramtx)
    return dst
    
def load_dataset(name, path_dataset, sequence, ref_time, Ne):
    if name == "DAVIS_240C":
        calib_data = np.loadtxt('{}/{}/calib.txt'.format(path_dataset,sequence))
        events = pd.read_csv(
            '{}/{}/events_chunk/events_{}.txt'.format(path_dataset,sequence, int(ref_time)), sep=' ', header=None, engine='python')
        events.columns = ["ts", "x", "y", "p"]

        fx = calib_data[0]
        fy = calib_data[1]
        px = calib_data[2]
        py = calib_data[3]
        dist_co = calib_data[4:]
        height = 180
        width = 240
        instrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

        LUT = np.zeros([width, height, 2])
        for i in range(width):
            for j in range(height):
                LUT[i][j] = np.array([i, j])
        LUT = LUT.reshape((-1, 1, 2))
        LUT = undo_distortion(LUT, instrinsic_matrix, dist_co, width, height).reshape((width, height, 2))
        events_set = events.to_numpy()

        t_k = events_set[:, 0]
        idx = (np.abs(t_k - ref_time)).argmin()
        event_range = slice(idx - Ne // 2, idx + Ne // 2)
        #event_range = slice(idx - Ne     , idx)
        #event_range = slice(idx          , idx + Ne)
        #events_set = events_set[event_range, :]
        #events_set = events_set[1500000:1650000, :]
        events_set = events_set[1500000:, :]
        #events_set = events_set[2009000:, :]
        #events_set = events_set[335000:, :]
        #events_set = events_set[1050000:, :]
        #events_set = events_set[972000:, :]
        #events_set = events_set[1080000:, :]
        #events_set = events_set[::-1, :]

    print("Events total count: ", len(events_set))
    print("Time duration of the sequence: {} s".format(events_set[-1][0] - events_set[0][0]))
    return LUT, events_set, height, width, fx, fy, px, py

def undistortion(events_batch, LUT, t_ref):
    events_batch[:, 0] = events_batch[:, 0] - t_ref
    events_batch[:, 1:3] = LUT[(events_batch[:, 1]).type(torch.long), (events_batch[:, 2]).type(torch.long), :]
    return events_batch

def timer(func):
    def acc_time(self, *args, **kwargs):
        start = time.time()
        func(self, *args, **kwargs)
        end = time.time()
        main_duration = end - start
        print('The estimation lasts for {} h {} m {} s.'.format(int(main_duration/3600),int(main_duration%3600/60),int(main_duration%3600%60)))
    return acc_time

def undistort_img(src, fx, fy, px, py, distco=None):
    h, w = src.shape
    mtx = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distco, (w, h), 0, (w, h))
    dst = cv2.undistort(src, mtx, distco, None, newcameramtx)
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    return dst

def calc_A_T(flow_y, flow_x, smooth):
    rows, cols = flow_y.shape
    flow_y_flat = flow_y.ravel()
    flow_x_flat = flow_x.ravel()
    flow_y_plus = flow_y_flat >= 0.
    flow_x_plus = flow_x_flat >= 0.
    flow_y_minus = np.logical_not(flow_y_plus)
    flow_x_minus = np.logical_not(flow_x_plus)
    flow_y_abs = np.abs(flow_y_flat)
    flow_x_abs = np.abs(flow_x_flat)
    vote_00 = (1.-flow_y_abs) * (1.-flow_x_abs)
    vote_01 = (1.-flow_y_abs) *     flow_x_abs
    vote_10 =     flow_y_abs  * (1.-flow_x_abs)
    vote_11 =     flow_y_abs  *     flow_x_abs
    #t,b,l,r stands for top,bottom,left,right
    mask_tl = np.logical_and(flow_y_minus, flow_x_minus)
    mask_tr = np.logical_and(flow_y_minus, flow_x_plus)
    mask_br = np.logical_and(flow_y_plus , flow_x_plus)
    mask_bl = np.logical_and(flow_y_plus , flow_x_minus)
    #m,t,b,l,r stands for middle,top,bottom,left,right
    vote_tl = mask_tl * vote_11
    vote_tr = mask_tr * vote_11
    vote_br = mask_br * vote_11
    vote_bl = mask_bl * vote_11
    vote_tm = np.logical_or(mask_tl, mask_tr) * vote_10
    vote_bm = np.logical_or(mask_bl, mask_br) * vote_10
    vote_lm = np.logical_or(mask_tl, mask_bl) * vote_01
    vote_rm = np.logical_or(mask_tr, mask_br) * vote_01
    vote_mm = vote_00
    #For Scipy, dia_matrix will throw away the head elements for the upper diagonals.
    #and tail elements for the lower diagonals. In our setting, the head of the vote_tl
    #and the tail of the vote_br should be thrown away. Because the flow of the border 
    # values are zero, so both the head and tails elements of the vote_tl and vote_br 
    # are zeros. So the first and last columns of the matrix A are zeros.
    diagonals_T = np.array([vote_br, vote_bm, vote_bl,
                            vote_rm, vote_mm, vote_lm,
                            vote_tr, vote_tm, vote_tl])
    offsets_T = np.array([-cols-1, -cols, -cols+1,
                            -1,     0,       1,
                           cols-1,  cols,  cols+1])                
    A_T = dia_matrix((diagonals_T, offsets_T), shape=(rows*cols, rows*cols))
    A_ = A_T.T
    A_res = sparse.eye(rows*cols) - A_
    if smooth:
        border_mask = np.zeros((rows, cols))
        border_mask[1:-1, 1:-1] = 1
        #3x3 Gaussian kernel has the values [1/16, 1/8, 1/16; 1/8, 1/4, 1/8; 1/16, 1/8, 1/16]
        main_diag_ = (np.ones(rows*cols) * 0.25) * border_mask.ravel()
        midd_diag_ = (np.ones(rows*cols) * 0.125) * border_mask.ravel()
        edge_diag_ = (np.ones(rows*cols) * 0.0625) * border_mask.ravel()
        #Don't have to worry the missing elements of the diagonals because the borders
        #are set to zeros.
        blur_diags_ = np.array([edge_diag_, midd_diag_, edge_diag_,
                                midd_diag_, main_diag_, midd_diag_,
                                edge_diag_, midd_diag_, edge_diag_])
        blur_offsets = np.array([-cols-1, -cols, -cols+1,
                                    -1,     0,       1,
                                cols-1,  cols,  cols+1])
        blur_mtx = dia_matrix((blur_diags_, blur_offsets), shape=(rows*cols, rows*cols))
        A_res = sparse.eye(rows*cols) - blur_mtx @ A_
    
    return A_res, A_res.T

def calc_A_prependicular(flow_y_, flow_x_):
    flow_y = -flow_x_
    flow_x = flow_y
    rows, cols = flow_y.shape
    flow_y_flat = flow_y.ravel()
    flow_x_flat = flow_x.ravel()
    flow_y_plus = flow_y_flat >= 0.
    flow_x_plus = flow_x_flat >= 0.
    flow_y_minus = np.logical_not(flow_y_plus)
    flow_x_minus = np.logical_not(flow_x_plus)
    flow_y_abs = np.abs(flow_y_flat)
    flow_x_abs = np.abs(flow_x_flat)
    vote_00 = (1.-flow_y_abs) * (1.-flow_x_abs)
    vote_01 = (1.-flow_y_abs) *     flow_x_abs
    vote_10 =     flow_y_abs  * (1.-flow_x_abs)
    vote_11 =     flow_y_abs  *     flow_x_abs
    #t,b,l,r stands for top,bottom,left,right
    mask_tl = np.logical_and(flow_y_minus, flow_x_minus)
    mask_tr = np.logical_and(flow_y_minus, flow_x_plus)
    mask_br = np.logical_and(flow_y_plus , flow_x_plus)
    mask_bl = np.logical_and(flow_y_plus , flow_x_minus)
    #m,t,b,l,r stands for middle,top,bottom,left,right
    vote_tl = mask_tl * vote_11
    vote_tr = mask_tr * vote_11
    vote_br = mask_br * vote_11
    vote_bl = mask_bl * vote_11
    vote_tm = np.logical_or(mask_tl, mask_tr) * vote_10
    vote_bm = np.logical_or(mask_bl, mask_br) * vote_10
    vote_lm = np.logical_or(mask_tl, mask_bl) * vote_01
    vote_rm = np.logical_or(mask_tr, mask_br) * vote_01
    vote_mm = vote_00
    diagonals_T = np.array([vote_br, vote_bm, vote_bl,
                            vote_rm, vote_mm, vote_lm,
                            vote_tr, vote_tm, vote_tl])
    offsets_T = np.array([-cols-1, -cols, -cols+1,
                            -1,     0,       1,
                           cols-1,  cols,  cols+1])
    A_T_sparse = dia_matrix((diagonals_T, offsets_T), shape=(rows*cols, rows*cols))
    A_T_sparse = sparse.eye(rows*cols) - A_T_sparse
    return A_T_sparse.T, A_T_sparse

class MyTimer:  #@save

    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

def padding(img, margin):
    img_height, img_width = img.shape
    img_new = np.zeros((img_height + 2*margin, img_width + 2*margin))
    img_new[margin:-margin, margin:-margin] = img
    return img_new

def cropping(img, margin):
    return img[margin:-margin, margin:-margin]

def get_border_mask(height, width, margin, smooth=True):
    #Border handeling, choose sharp border or smooth border
    mask = np.zeros((height, width))
    mask[margin:-margin, margin:-margin] = 1
    if smooth == True:
        mask = cv2.GaussianBlur(mask,(margin*2+1, margin*2+1),0)
    return mask

def calc_R_matrix(rows, cols):
    diag = np.ones(rows)
    diag[0] = 0
    diag_tile = np.tile(diag, cols)
    diag_hon = np.array([diag_tile, -diag_tile])
    offsets_hon = np.array([0, -1])
    R_hon = dia_matrix((diag_hon, offsets_hon), shape=(rows*cols, rows*cols))
    
    diag = np.ones(rows*cols)
    diag[:cols] = 0
    diag_ver = np.array([diag, -diag])
    offsets_ver = np.array([0, -cols])
    R_ver = dia_matrix((diag_ver, offsets_ver), shape=(rows*cols, rows*cols))

    return R_hon.T @ R_hon + R_ver.T @ R_ver, R_hon, R_ver
