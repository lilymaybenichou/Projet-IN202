
import PIL
from PIL import Image
import numpy as np
import scipy as sp
import os
from math import log10, sqrt

def load(filename):
    toLoad= Image.open(filename)
    return np.asarray(toLoad)


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def dct2(a):
    return sp.fft.dct( sp.fft.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return sp.fft.idct( sp.fft.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')


def Y(mat):
    MatY=np.empty((mat.shape[0],mat .shape [11,1]))
    for i in range(mat.shape [0]):
     for j in range(mat.shape [1]):
      MatY [i,j] = 0.299 *mat [i, j,0] + 0.587*mat [i, j,1]+ 0.114 *mat[i, j,2]
    return MatY

def Cb(mat):
   MatCb=np.empty((mat.shape[0],mat.shape[1],1))
   for i in range(mat.shape [0]):
    for j in range(mat. shape [1]):
     MatCb[1,j] = -0.1687 *mat [i,j,0] - 0.3313*mat[1,j,1]+ 0.5*mat[i,j,2] + 128
   return MatCb

def Cr(mat):
   MatCr = np.empty( (mat. shape [0] , mat. shape [1] ,1))
   for i in range(mat.shape [0]):
    for j in range(mat.shape [1]):
     MatCr[i,j] = 0.5 *mat [i,j,0] - 0.4187*mat [i,j,1]- 0.0813*mat [i,j,2] + 128
   return MatCr

def YCbCr (mat) :
   MatYCbCr = np.empty (mat.shape)
   for i in range(mat.shape [0]):
    for j in range(mat.shape [1]):
     R =0.299 *mat [i,j,0] + 0.587 *mat[i,j,1] + 0.114*mat [i,j,2]
     Cb = -0.1687*mat [1,j,0] - 0.3313 *mat[i,j,1] + 0.5*mat [1,j,2] + 128
     Cr = 0.5*mat [1,j,0] - 0.4187*mat [1,j,1] - 0.0813*mat [i,j,2] + 128
   print (MatYCbCr [0,0])
   return MatYCbCr

def RGB2 (mat):
   MatRGB = np.empty ( (mat.shape [0] , mat. shape [1],3) )
   for i in range(mat.shape [0]):
    for j in range(mat.shape [1]):
     R = mat [i,j,0] + 1.402 *(mat [i,j,2]-128)
     G = mat [i,j,0] - 0.34414 *(mat[i,j,1]-128) - 0.71414*(mat [i,j,2]-128)
     B = mat[i,j,0] + 1.772*(mat [i,j,1] - 128)
     MatRGB[i,j] = (np.uint8(np.clip (R, 0.0,255.0)),np.uint8(np.clip(G,0.0,255.0)),np.uint8(np.clip(B,0.0,255.0)))
   return MatRGB 


test = load("test.png")

# Conversion de l'image en YCbCr
test_yCbCr = YCbCr(test)

# Vérification de la modification de l'image
print("Image originale :")
print(test[0, 0])
print("Image convertie en YCbCr :")
print(test_yCbCr[0, 0])
