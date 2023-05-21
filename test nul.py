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
    MatYCbCr = np.empty(mat.shape)
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

# yCbCr_image = YCbCr(test)
# RGB_image = RGB2(test)

# Afficher l'image convertie en YCbCr
# Image.fromarray(yCbCr_image.astype(np.uint8), 'YCbCr').show()

# Afficher l'image convertie en RGB
# Image.fromarray(RGB_image.astype(np.uint8), 'RGB').show()

# Charger l'image
# image = Image.open("test.png")
# image_array = np.array(image)


# question 3
def add_padding(image):
    x = image.shape[1]
    if image.shape[1] % 8 != 0:
        y = x
        while x % 8 != 0:
            x += 1
        pad_sizeL = x - y
        print(pad_sizeL)
    a = image.shape[0]
    if a % 8 != 0:
        b = a
        while a % 8 != 0:
            a += 1
        pad_sizel = a - b
        print(pad_sizel)
    else:
        return "pas besoin"

    pad_size = (pad_sizeL, pad_sizel)  # Si un seul entier est donné, le même padding sera ajouté dans les deux dimensions

    padded_image = np.pad(image, ((0, pad_size[1]), (0, pad_size[0]), (0, 0)), mode='constant')

    return padded_image


def remove_padding(padded_image, pad_size):
    if isinstance(pad_size, int):
        pad_size = (pad_size, pad_size)  # Si un seul entier est donné, le même padding sera éliminé dans les deux dimensions

    image = padded_image[pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1], :]

    return image


# Charger l'image
# image = load("test.png")
# Ajouter le padding
# pad_size = 10
# padded_image = add_padding(image, pad_size)

# padded_RGB = RGB2(padded_image)
# restored_RGB = remove_padding(padded_RGB, pad_size)

# padded_yCbCr = YCbCr(padded_image)
# restored_yCbCr = remove_padding(padded_yCbCr, pad_size)


# question 4
def matrice_sousechantillon(mat):
    matrice_se = mat[::2, ::2]
    return matrice_se


# Question 5
def matrice_2d(matrice):
    matrice_double = np.repeat(matrice, 2, axis=1)
    return matrice_double


# subsampled_yCbCr = matrice_sousechantillon(YCbCr(padded_image))
# subsampled_RGB = matrice_sousechantillon(RGB2(padded_image))

# Sous-échantillonnage
# subsampled_image = image_array[::2, ::2]

# Multiplier par deux la deuxième dimension
# doubled_image = matrice_2d(subsampled_image)

# Afficher les images
# Image.fromarray(image_array).show()
# Image.fromarray(subsampled_image).show()
# Image.fromarray(doubled_image).show()


# Question 6
def get_block(mat):
    indice = 8
    Liste_final = []
    if mat.shape[0] % indice != 0 and mat.shape[1] % indice != 0:
        return "matrice non divisible par 8"
    else:
        for i in range(0, mat.shape[1] - 1, indice):
            for j in range(0, mat.shape[0] - 1, indice):
                Liste_final.append(mat[i:i + indice, j:j + indice])
    Liste_final = np.array(Liste_final)
    return Liste_final


matrice = np.array(
    [[1, 2, 5, 6, 0, 0, 0, 0], [3, 4, 7, 8, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
print(get_block(matrice))

# Question 8
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


# Exemple d'utilisation
image = load("test.png")
seuil = 10
mode0 = compress_mode0(image)
mode1 = compress_mode1(image, seuil)
mode2 = compress_mode2(image, seuil)

# Question 9
def decompress_mode0(blocks):
    padded_image = np.zeros((blocks.shape[0] * 8, blocks.shape[1] * 8, 1))
    for i in range(0, padded_image.shape[1] - 1, 8):
        for j in range(0, padded_image.shape[0] - 1, 8):
            padded_image[i:i + 8, j:j + 8] = blocks[i // 8 * blocks.shape[1] // 8 + j // 8]

    image = remove_padding(padded_image, blocks.shape[1] % 8)

    return image


def decompress_mode1(blocks, subsampled_Cb, subsampled_Cr):
    padded_image = np.zeros((blocks.shape[0] * 8, blocks.shape[1] * 8, 1))
    for i in range(0, padded_image.shape[1] - 1, 8):
        for j in range(0, padded_image.shape[0] - 1, 8):
            padded_image[i:i + 8, j:j + 8] = blocks[i // 8 * blocks.shape[1] // 8 + j // 8]

    image = remove_padding(padded_image, blocks.shape[1] % 8)

    image[:, :, 1] = subsampled_Cb
    image[:, :, 2] = subsampled_Cr

    image = RGB2(image)

    return image


# Exemple d'utilisation
mode0_restored = decompress_mode0(mode0)
mode1_restored = decompress_mode1(mode1, subsampled_Cb, subsampled_Cr)


# Exemple d'utilisation
image = load("test.png")
restored_image = decompress_mode1(mode1, subsampled_Cb, subsampled_Cr)
 











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


# Exemple d'utilisation des fonctions de compression

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

# Afficher les résultats
print("Mode 0 - Compressed shape:", compressed_mode0.shape)
print("Mode 1 - Compressed shape:", compressed_mode1.shape)
print("Mode 2 - Compressed shape:", compressed_mode2.shape)
print("Subsampled Cb shape:", subsampled_Cb.shape)
print("Subsampled Cr shape:", subsampled_Cr.shape)

# Charger l'image
image = Image.open("test.png")
image_array = np.array(image)

# Charger l'image
image = Image.open("test.png")
image_array = np.array(image)


test = load("test.png")

#test_yCbCr = YCbCr(test)
#test_RGB = RGB2(test_yCbCr)


test = load("test.png")
#Image.fromarray(test,'RGB').show()
#Image.fromarray(test_yCbCr,'YCbCr').show()
#Image.fromarray(test_RGB,'RGB').show()
