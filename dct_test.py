from scipy.fftpack import dct, idct
import cv2
import numpy as np
import matplotlib.pylab as plt

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
    print(scale)
    #img_normalized.astype(np.uint8)
    return img_normalized

def dct2(a):
    return dct(dct(a.T, type=2, norm='ortho').T, type=2, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, type=2, norm='ortho').T, type=2, norm='ortho')    

img = cv2.imread('testsets/set12/08.png', cv2.IMREAD_GRAYSCALE)
mosaic_gradx = cv2.imread('mosaic_gradx.exr', cv2.IMREAD_UNCHANGED)
mosaic_grady = cv2.imread('mosaic_grady.exr', cv2.IMREAD_UNCHANGED)

# img_lap = np.zeros_like(img_gradx)
mosaic_gradx2 = mosaic_gradx[:, 1:] - mosaic_gradx[:, :-1]
mosaic_grady2 = mosaic_grady[1:, :] - mosaic_grady[:-1, :]
mosaic_lap = mosaic_gradx2[:1023,:2047] + mosaic_grady2[:1023,:2047]

# img_gradx = normalizeRobust(img_gradx, 1.0)
# img_grady = normalizeRobust(img_grady, 1.0)
#img_lap = normalizeRobust(img_lap, 1.0)
#img_gradx = img_gradx.astype(np.uint8)
#img_grady = img_grady.astype(np.uint8)
# img_lap = img_lap.astype(np.uint8)
# plt.imshow(img_lap, cmap='gray')
# plt.show() 
#mosaic_lap = mosaic_lap[256:768, 768:1280]
#mosaic_lap = mosaic_lap[200:800, 500:1500]

#Nx, Ny = (512, 512)
Ny, Nx = mosaic_lap.shape
# Nx, Ny = img.shape
# grad_x = cv2.Sobel(img, cv2.CV_64F, 1,0)
# grad_y = cv2.Sobel(img, cv2.CV_64F, 0,1)
# img_lap = cv2.Laplacian(img, cv2.CV_64F)
# plt.imshow(img_lap, cmap='gray')
# plt.show() 
# imF = dct2(img)
lap_F = dct2(mosaic_lap)
kx = np.arange(Nx)
ky = np.arange(Ny)
factor_x = 2.0 * (np.cos((np.pi * kx) / Nx) - 1.0)
factor_y = 2.0 * (np.cos((np.pi * ky) / Ny) - 1.0)
mwx, mwy = np.meshgrid(factor_x, factor_y)
factor_div = mwx + mwy
factor_div = np.where(factor_div == 0.0, 1.0, factor_div)
lap_F2 = lap_F / factor_div
lap_F2[0, 0] = 0.0
for i in range(Ny):
    for j in range(Nx):
        factor = 2.0 * (np.cos((np.pi * i) / (Ny - 1)) + np.cos((np.pi * j) / (Nx - 1)) - 2.0)
        if factor != 0.0:
            lap_F[i, j] /= factor
        else:
            lap_F[i, j] = 0.0
# im1 = idct2(imF)
mosaic_rec = idct2(lap_F)
mosaic_rec2 = idct2(lap_F2)
mosaic_rec = normalizeRobust(mosaic_rec, 1.0)
mosaic_rec2 = normalizeRobust(mosaic_rec2, 1.0)
#img_rec = img_rec.astype(np.uint8)

plt.gray()
plt.subplot(121), plt.imshow(mosaic_rec), plt.axis('off'), plt.title('original image', size=20)
plt.subplot(122), plt.imshow(mosaic_rec2), plt.axis('off'), plt.title('reconstructed image (DCT+IDCT)', size=20)
plt.show()