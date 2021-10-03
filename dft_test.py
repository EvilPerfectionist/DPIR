import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftfreq

mosaic_gradx = cv2.imread('mosaic_gradx.exr', cv2.IMREAD_UNCHANGED)
mosaic_grady = cv2.imread('mosaic_grady.exr', cv2.IMREAD_UNCHANGED)

mosaic_gradx2 = np.roll(mosaic_gradx, -1, axis=1) - mosaic_gradx
mosaic_grady2 = np.roll(mosaic_grady, -1, axis=0) - mosaic_grady
mosaic_lap = mosaic_gradx2 + mosaic_grady2

Ny, Nx = mosaic_lap.shape

lap_F = fft2(mosaic_lap / 90.0)

kx = fftfreq(Nx)
ky = fftfreq(Ny)
factor_x = 2.0 * (np.cos(np.pi * kx) - 1.0)
factor_y = 2.0 * (np.cos(np.pi * ky) - 1.0)
mwx, mwy = np.meshgrid(factor_x, factor_y)
factor_div = mwx + mwy
factor_div = np.where(factor_div == 0.0, 1.0, factor_div)
lap_F2 = lap_F / factor_div
lap_F2[0, 0] = (1024.0 * 2048.0) / 2.0

mosaic_rec = np.real(ifft2(lap_F2))
# mosaic_rec = np.fft.ifft2(lap_F2)
plt.imshow(mosaic_rec, cmap='gray')
plt.colorbar()
plt.show()