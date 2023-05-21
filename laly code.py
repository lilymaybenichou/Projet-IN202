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

test = load("test.png")

# Conversion de l'image en YCbCr et RGB
test_yCbCr = YCbCr(test)
test_RGB = RGB2(test_yCbCr)

test = load("test.png")
Image.fromarray(test,'RGB').show()
Image.fromarray(test_yCbCr,'YCbCr').show()
Image.fromarray(test_RGB,'RGB').show()


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

#question 4
def matrice_sousechantillon(mat):
    matrice_se = mat[::2, ::2]
    return matrice_se

#Question 5
def matrice_2d(matrice):
    matrice_double = np.repeat(matrice, 2, axis=1)
    return matrice_double

#Question 6
def get_block(mat):
    indice = 8
    Liste_final=[]
    if mat.shape[0]%indice!=0 and mat.shape[1]%indice!=0:
        return "matrice non divisible par 8"
    else :
        for i in range (0,mat.shape[1]-1,indice):
            for j in range (0,mat.shape[0]-1,indice):
                Liste_final.append(mat[i:i+indice,j:j+indice])
    Liste_final=np.array(Liste_final)
    return Liste_final


matrice = np.array([[1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
print(get_block(matrice))

#question7
def transform_frequence(liste_de_bloc):
    liste_final=[]
    for i in liste_de_bloc :
        liste_final.append(dct2(i))
    liste_final=np.array(liste_final)
    return liste_final


def detransform_frequence(liste_de_bloc):
    liste_final=[]
    for i in liste_de_bloc :
        liste_final.append(idct2(i))
    liste_final=np.array(liste_final)
    return liste_final

matrice = np.array([[1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
listb = get_block(matrice)
print(listb)
listt = transform_frequence(listb)
print(listt)
listdt = transform_frequence(listt)
print(listdt)

#Question 8 
def compress_mode0(image):
    yCbCr_image = YCbCr(image)
    padded_image = add_padding(yCbCr_image)
    blocks = get_block(padded_image)

    return blocks


def compress_mode1(image, seuil):
    yCbCr_image = YCbCr(image)
    padded_image = add_padding(yCbCr_image)
    blocks = get_block(padded_image)
    blocks[np.abs(blocks) < seuil] = 0

    return blocks


def compress_mode2(image, seuil):
    yCbCr_image = YCbCr(image)
    padded_image = add_padding(yCbCr_image)
    blocks = get_block(padded_image)
    blocks[np.abs(blocks) < seuil] = 0
    subsampled_Cb = matrice_sousechantillon(Cb(padded_image))
    subsampled_Cr = matrice_sousechantillon(Cr(padded_image))

    return blocks, subsampled_Cb, subsampled_Cr

# Charger l'image
image = load("test.png")

# Ajouter le padding
pad_size = 10
padded_image = add_padding(image, pad_size)

subsampled_yCbCr = matrice_sousechantillon(YCbCr(padded_image))
padded_yCbCr = YCbCr(padded_image)
restored_yCbCr = remove_padding(padded_yCbCr, pad_size)

subsampled_RGB = matrice_sousechantillon(RGB2(padded_image))
padded_RGB = RGB2(padded_image)
restored_RGB = remove_padding(padded_RGB, pad_size)

# Charger l'image
image = Image.open("test.png")
image_array = np.array(image)

# Sous-échantillonnage
subsampled_image = image_array[::2, ::2]

# Multiplier par deux la deuxième dimension
doubled_image = matrice_2d(subsampled_image)

# Afficher les images
Image.fromarray(image_array).show()
Image.fromarray(subsampled_image).show()
Image.fromarray(doubled_image).show()

#Compression
# Charger l'image
image = load("test.png")

# Mode 0 : blocs transformés tels quels
compressed_mode0 = compress_mode0(image)

# Mode 1 : seuil appliqué aux coefficients
seuil = 8
compressed_mode1 = compress_mode1(image, seuil)

# Mode 2 : seuil appliqué aux coefficients + sous-échantillonnage de la chrominance
seuil = 8
compressed_mode2, subsampled_Cb, subsampled_Cr = compress_mode2(image, seuil)

