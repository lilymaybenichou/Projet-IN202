import PIL
from PIL import Image
import numpy as np
import scipy as sp
import os
from math import log10, sqrt

def load(filename):
    toLoad = Image.open(filename)
    return np.asarray(toLoad)

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def dct2(a):
    return sp.fft.dct(sp.fft.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(a):
    return sp.fft.idct(sp.fft.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def Y(mat):
    MatY = np.empty((mat.shape[0], mat.shape[1], 1))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            MatY[i, j] = 0.299 * mat[i, j, 0] + 0.587 * mat[i, j, 1] + 0.114 * mat[i, j, 2]
    return MatY

def Cb(mat):
    MatCb = np.empty((mat.shape[0], mat.shape[1], 1))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            MatCb[i, j] = -0.1687 * mat[i, j, 0] - 0.3313 * mat[i, j, 1] + 0.5 * mat[i, j, 2] + 128
    return MatCb

def Cr(mat):
    MatCr = np.empty((mat.shape[0], mat.shape[1], 1))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            MatCr[i, j] = 0.5 * mat[i, j, 0] - 0.4187 * mat[i, j, 1] - 0.0813 * mat[i, j, 2] + 128
    return MatCr

def YCbCr(mat):
    MatYCbCr = np.empty((mat.shape[0], mat.shape[1], 3))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            R = 0.299 * mat[i, j, 0] + 0.587 * mat[i, j, 1] + 0.114 * mat[i, j, 2]
            Cb = -0.1687 * mat[i, j, 0] - 0.3313 * mat[i, j, 1] + 0.5 * mat[i, j, 2] + 128
            Cr = 0.5 * mat[i, j, 0] - 0.4187 * mat[i, j, 1] - 0.0813 * mat[i, j, 2] + 128
            MatYCbCr[i, j] = [R, Cb, Cr]
    return MatYCbCr

def RGB2(mat):
    MatRGB = np.empty((mat.shape[0], mat.shape[1], 3))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            R = mat[i, j, 0] + 1.402 * (mat[i, j, 2] - 128)
            G = mat[i, j, 0] - 0.34414 * (mat[i, j, 1] - 128) - 0.71414 * (mat[i, j, 2] - 128)
            B = mat[i, j, 0] + 1.772 * (mat[i, j, 1] - 128)
            MatRGB[i, j] = (np.uint8(np.clip(R, 0.0, 255.0)), np.uint8(np.clip(G, 0.0, 255.0)), np.uint8(np.clip(B, 0.0, 255.0)))
    return MatRGB

#Conversion de l'image en YCbCr 
image = load("150_210.png")
ycbcr_image = YCbCr(image)
converted_image = Image.fromarray(ycbcr_image.astype(np.uint8)).show()

#Conversion de l'image en RGB 
image = load("150_210.png")
rgb_image = RGB2(image)
converted_image = Image.fromarray(rgb_image.astype(np.uint8)).show()

def add_padding(image, pad_size):
    if isinstance(pad_size, int):
        pad_size = (pad_size, pad_size)  # Si un seul entier est donné, le même padding sera ajouté dans les deux dimensions

    padded_image = np.pad(image, ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (0, 0)), mode='constant')

    return padded_image


def remove_padding(padded_image, pad_size):
    if isinstance(pad_size, int):
        pad_size = (pad_size, pad_size)  # Si un seul entier est donné, le même padding sera éliminé dans les deux dimensions

    image = padded_image[pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1], :]

    return image



