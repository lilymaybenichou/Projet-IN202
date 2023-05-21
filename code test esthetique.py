import PIL
from PIL import Image
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
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

#Question 1 
def YCbCr(mat):
    MatYCbCr = np.empty((mat.shape[0], mat.shape[1], 3))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            R = 0.299 * mat[i, j, 0] + 0.587 * mat[i, j, 1] + 0.114 * mat[i, j, 2]
            Cb = -0.1687 * mat[i, j, 0] - 0.3313 * mat[i, j, 1] + 0.5 * mat[i, j, 2] + 128
            Cr = 0.5 * mat[i, j, 0] - 0.4187 * mat[i, j, 1] - 0.0813 * mat[i, j, 2] + 128
            MatYCbCr[i, j] = [R, Cb, Cr]
    return MatYCbCr
#Conversion de l'image en YCbCr 
image = load("test.png")
ycbcr_image = YCbCr(image)
converted_image = Image.fromarray(ycbcr_image.astype(np.uint8)).show()

#Question 2
def RGB2(mat):
    MatRGB = np.empty((mat.shape[0], mat.shape[1], 3))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            R = mat[i, j, 0] + 1.402 * (mat[i, j, 2] - 128)
            G = mat[i, j, 0] - 0.34414 * (mat[i, j, 1] - 128) - 0.71414 * (mat[i, j, 2] - 128)
            B = mat[i, j, 0] + 1.772 * (mat[i, j, 1] - 128)
            MatRGB[i, j] = (np.uint8(np.clip(R, 0.0, 255.0)), np.uint8(np.clip(G, 0.0, 255.0)), np.uint8(np.clip(B, 0.0, 255.0)))
    return MatRGB
#Conversion de l'image en RGB 
image = load("test.png")
rgb_image = RGB2(image)
converted_image = Image.fromarray(rgb_image.astype(np.uint8)).show()

#Question 3
def add_padding(image):
    x=image.shape[1]
    a=image.shape[0]
    pad_sizeL=0
    pad_sizel = 0
    if (x%8==0)and (a%8==0):
        return image , (pad_sizeL,pad_sizel)

    if image.shape[1]%8!=0:
        y=x
        while x%8!=0:
            x+=1
        pad_sizeL=x-y   
       
   
    if a%8!=0: 
        b=a
        while a%8!=0:
            a+=1
        pad_sizel=a-b 
              
    pad_size = (pad_sizeL, pad_sizel)  # Si un seul entier est donné, le même padding sera ajouté dans les deux dimensions
    padded_image = np.pad(image, ((0, pad_size[1]), (0, pad_size[0]), (0, 0)), mode='constant')

    return padded_image,pad_size
#Test validite du add padding
image = load("test.png")
image = np.array(image)
padded_image, pad_size = add_padding(image)
print(padded_image.shape)  # Affiche les dimensions de l'image 
Image.fromarray(padded_image,'RGB').show()

def remove_padding(padded_image, pad_size):
    if isinstance(pad_size, int):
        pad_size = (pad_size, pad_size)  # Si un seul entier est donné, le même padding sera éliminé dans les deux dimensions

    image = padded_image[pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1], :]

    return image
#Test validite du remove padding
restored_image = remove_padding(padded_image, pad_size)
print(restored_image.shape)  # Affiche les dimensions de l'image restaurée sans le padding

#Question 4
def matrice_sousechantillon2(mat):
    matfinal=[]
    for i in range(0,mat.shape[0]): 
        listemoy=[]
        for j in range(0,mat.shape[1],2):
            moyenne=(mat[i,j]+mat[i,j+1])/2
            listemoy.append(moyenne)
        matfinal.append(listemoy)
    return np.array(matfinal)
#Test matrice   
mat=np.array([[1,2,3,6],[3,5,10,0]])
print(matrice_sousechantillon2(mat))

#Question 5
def matrice_2d(matrice):
    matrice_double = np.repeat(matrice, 2, axis=1)
    return matrice_double
#Test matrice 2d
mat=np.array([[1,2,3,6],[3,5,10,0]])
print(matrice_sousechantillon2(mat))
print(matrice_2d(matrice_sousechantillon2(mat)))

#Test sous echantillonage
image = Image.open("test.png")
image_array = np.array(image)

# Sous-échantillonnage
subsampled_image = image_array[::2, ::2]

# 2 dimension
doubled_image = matrice_2d(subsampled_image)

Image.fromarray(image_array).show()
Image.fromarray(subsampled_image).show()
Image.fromarray(doubled_image).show()

#Question 6 
def get_block(mat):
    indice = 8
    Liste_final=[]
    if mat.shape[0]%indice!=0 and mat.shape[1]%indice!=0:
        return "matrice non divisible par 8"
    else :
        for j in range (0,mat.shape[1]-1,indice):
            for i in range (0,mat.shape[0]-1,indice):
                Liste_final.append(mat[i:i+indice,j:j+indice])
                
    Liste_final=np.array(Liste_final)
    return Liste_final



#Question 7 
matrice = np.array([[1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0]])

def transform_frequence(liste_de_bloc):
    liste_final=[]
    for i in liste_de_bloc :
        liste_final.append(dct2(i))
   
    liste_final=np.array(liste_final)
    return liste_final

def detransform_frequence(liste_de_bloc):
    liste_final=[]
    for i in liste_de_bloc :
        liste_final.append(int(idct2(i)))
    
    liste_final=np.array(liste_final)
    return liste_final

matrice = np.array([[1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0]])

#Test quest7
listb = get_block(matrice)
print(listb)
listt = transform_frequence(listb)
print(listt)
listdt = detransform_frequence(listt)
print(listdt)

#Question 8 
def filter_coeff(liste_de_bloc,seuil):
    liste_final=[]
    for b in liste_de_bloc :
        b[(b>0) & (b < seuil)] = 0 
        b[(b<0) & (b > -seuil)] = 0 
        liste_final.append(b)
    
    return liste_final

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

matrice = np.array([[1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                    [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                    [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                    [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                    [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                    [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                    [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                    [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0]])

listb = get_block(matrice)
print(listb)
listt = transform_frequence(listb)
print(listt)
listtcoeff= filter_coeff(listt,10)
print(listtcoeff)

#Question 9
def compress(image,mode,seuil):
    imagemat = load(image)
    imageYCbCr=YCbCr(imagemat)
    listblockY = get_block(imageYCbCr[:,:,0])
    listblockCb=get_block(imageYCbCr[:,:,1])
    listblockCr=get_block(imageYCbCr[:,:,2])
    if mode==0:
        return (listblockY, listblockCb, listblockCr)
    if mode==1 :
        listblockY=filter_coeff(listblockY,seuil)
        return (listblockY, listblockCb, listblockCr)
    
    if mode==2:
        listblockY=filter_coeff(listblockY,seuil)
        
        listblockCb_ech=matrice_sousechantillon2(imageYCbCr[:,:,1])
        listblockCb=filter_coeff(listblockCb_ech,seuil)

        listblockCr_ech=matrice_sousechantillon2(imageYCbCr[:,:,2])
        listblockCr=filter_coeff(listblockCr_ech,seuil)
        return (listblockY, listblockCb, listblockCr)    

imagecompress=compress("test.png",0,10)
print(imagecompress)
imagecompress=compress("test.png",1,10)
print(imagecompress)
imagecompress=compress("test.png",2,10)
print(imagecompress)

#Question 10
def write_im_header(pathTextFile,pathImageFile,mode,encoding):
    f=open(pathTextFile,"w")
    f.write("SJPG\n")
    image = load(pathImageFile)
    hauteur=str(image.shape[0])
    largeur=str(image.shape[1])
    f.write(hauteur+" "+largeur+"\n")
    f.write(mode+"\n")
    f.write(encoding+"\n")
    f.close()

write_im_header("txtFile.txt","150_210.png","mode 0","RLE")

#Question 11
def write_im_header_block(pathTextFile,pathImageFile,mode,encoding):
    seuil=10
    f=open(pathTextFile,"w")
    f.write("SJPG\n")
    image = load(pathImageFile)
    
    padded_image = add_padding(image)[0]
    hauteur=str(padded_image.shape[0])
    largeur=str(padded_image.shape[1])
    f.write(hauteur+" "+largeur+"\n")
    f.write(mode+"\n")
    f.write(encoding+"\n")
    imageY=Y(padded_image)
    imageCb=Cb(padded_image)
    imageCr=Cr(padded_image)
    listeblockY=get_block(imageY)
    listeblockCb=get_block(imageCb)
    listeblockCr=get_block(imageCr)
    #TRlisteblockY=transform_frequence(listeblockY)
    if mode=="mode 0":
        listeblockY=transform_frequence(listeblockY)
        listeblockCb=transform_frequence(listeblockCb)
        listeblockCr=transform_frequence(listeblockCr)
    if mode=="mode 1":
        listeblockY=transform_frequence(listeblockY)
        listeblockY=filter_coeff(listeblockY,seuil)
        listeblockCb=transform_frequence(listeblockCb)
        listeblockCb=filter_coeff(listeblockCb,seuil)
        listeblockCr=transform_frequence(listeblockCr)
        listeblockCr=filter_coeff(listeblockCr,seuil)
    if mode=="mode 2":
        listeblockY=transform_frequence(listeblockY)
        listeblockY=filter_coeff(listeblockY,seuil)

        imageCb=matrice_sousechantillon2(imageCb)
        imageCr = matrice_2d(imageCb)
        listeblockCb=get_block(imageCb)
        listeblockCb=transform_frequence(listeblockCb)
        listeblockCb=filter_coeff(listeblockCb,seuil)
       
        imageCr=matrice_sousechantillon2(imageCr)
        imageCr = matrice_2d(imageCr)
        listeblockCr=get_block(imageCr)
        listeblockCr=transform_frequence(listeblockCr)
        listeblockCr=filter_coeff(listeblockCr,seuil)
        

    for b in listeblockY:
        line=""
        for i in range(0,8):
            for j in range(0,8):
                line = line + str(int(b[i,j]))
                if(i!=8 and j!=8):
                    line = line + " "
        f.write( line +"\n")
    for b in listeblockCb:
        line=""
        for i in range(0,8):
            for j in range(0,8):
                line = line + str(int(b[i,j]))
                if(i!=8 and j!=8):
                    line = line + " "
        f.write( line +"\n")
    for b in listeblockCr:
        line=""
        for i in range(0,8):
            for j in range(0,8):
                line = line + str(int(b[i,j]))
                if(i!=8 and j!=8):
                    line = line + " "
        f.write( line +"\n")
    f.close()

write_im_header_block("txtFile.txt","test.png","mode 0","RLE")
write_im_header_block("txtFile1.txt","test.png","mode 1","RLE")
write_im_header_block("txtFile2.txt","test.png","mode 2","RLE")

#Question 12 
def run_length_encoding(block):
    encoded_block = []
    current_value = block[0]
    count = 1

    for i in range(1, len(block)):
        if block[i] == current_value:
            count += 1
        else:
            encoded_block.append((current_value, count))
            current_value = block[i]
            count = 1

    encoded_block.append((current_value, count))

    return encoded_block

def write_im_header_block(pathTextFile, pathImageFile, mode, encoding):
    f = open(pathTextFile, "w")
    f.write("SJPG\n")
    image=load(pathImageFile)
    padded_image=add_padding(image)[0]
    hauteur=str(padded_image.shape[0])
    largeur=str(padded_image.shape[1])
    f.write(hauteur + " " + largeur + "\n")
    f.write(mode + "\n")
    f.write(encoding + "\n")
    imageY=Y(padded_image)
    imageCb=Cb(padded_image)
    imageCr=Cr(padded_image)
    listeblockY=get_block(imageY)
    listeblockCb=get_block(imageCb)
    listeblockCr=get_block(imageCr)
    #TRlisteblockY=transform_frequence(listeblockY)
    if mode == "mode 0":
        if encoding == "RLE":
            listeblockY = run_length_encoding(listeblockY)
            listeblockCb = run_length_encoding(listeblockCb)
            listeblockCr = run_length_encoding(listeblockCr)
        listeblockYTR = transform_frequence(listeblockY)
        listeblockCbTR = transform_frequence(listeblockCb)
        listeblockCrTR = transform_frequence(listeblockCr)

        for b in listeblockYTR:
            line = ""
            for i in range(8):
                for j in range(8):
                    line += str(int(b[i, j]))
                    if i != 7 or j != 7:
                        line += " "
            f.write(line + "\n")

        for b in listeblockCbTR:
            line = ""
            for i in range(8):
                for j in range(8):
                    line += str(int(b[i, j]))
                    if i != 7 or j != 7:
                        line += " "
            f.write(line + "\n")

        for b in listeblockCrTR:
            line = ""
            for i in range(8):
                for j in range(8):
                    line += str(int(b[i, j]))
                    if i != 7 or j != 7:
                        line += " "
            f.write(line + "\n")

    f.close()

write_im_header_block("txtFile.txt","test.png","mode 0","RLE")

#Question 13