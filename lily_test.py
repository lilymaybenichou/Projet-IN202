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
            MatYCbCr[i, j] = (np.uint8(R), np.uint8(Cb), np.uint8(Cr))
            
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


#question1
#test = load("test.png")
#Conversion de l'image en YCbCr
#test_yCbCr = YCbCr(test)

#question2
#test_RGB = RGB2(test_yCbCr)
#test = load("test.png")
#Image.fromarray(test,'RGB').show()
#Image.fromarray(test_yCbCr,'YCbCr').show()
#Image.fromarray(test_RGB,'RGB').show()


#question 3
def add_padding(image):
    x=image.shape[1]
    if image.shape[1]%8!=0:
        y=x
        while x%8!=0:
            x+=1
        pad_sizeL=x-y   
        print(pad_sizeL)
    a=image.shape[0]
    if a%8!=0: 
        b=a
        while a%8!=0:
            a+=1
        pad_sizel=a-b 
        print(pad_sizel)
    else:
        return("pas besoin")
          
    pad_size = (pad_sizeL, pad_sizel)  # Si un seul entier est donné, le même padding sera ajouté dans les deux dimensions

    padded_image = np.pad(image, ((0, pad_size[1]), (0, pad_size[0]), (0, 0)), mode='constant')

    return padded_image,pad_size


def remove_padding(padded_image, pad_size):
    if isinstance(pad_size, int):
        pad_size = (pad_size, pad_size)  # Si un seul entier est donné, le même padding sera éliminé dans les deux dimensions

    image = padded_image[pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1], :]

    return image

# Charger l'image
image = load("150_210.png")

#print(image.shape)

# Ajouter le padding
#padded_image = add_padding(image)
#Image.fromarray(padded_image,'RGB').show()

#padded_yCbCr = YCbCr(padded_image)
#restored_yCbCr = remove_padding(padded_yCbCr)

#padded_RGB = RGB2(padded_image)
#restored_RGB = remove_padding(padded_RGB)

#question4
def matrice_sousechantillon(mat):
    matrice_se = mat[::2, ::2]
    return matrice_se

#Question 5
def matrice_2d(matrice):
    matrice_double = np.repeat(matrice, 2, axis=1)
    return matrice_double

# Charger l'image
#image = Image.open("test.png")
#image_array = np.array(image)

# Sous-échantillonnage
#subsampled_image = image_array[::2, ::2]

# Multiplier par deux la deuxième dimension
#doubled_image = matrice_2d(subsampled_image)

# Afficher les images
#Image.fromarray(image_array).show()
#Image.fromarray(subsampled_image).show()
#Image.fromarray(doubled_image).show()

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
                
    #Liste_final=np.array(Liste_final)
    return Liste_final



#question7matrice = np.array([[1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
#                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
#                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
#                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
#                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
#                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
#                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
#                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0]]
#                     )

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

# matrice = np.array([[1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
#                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
#                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
#                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
#                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
#                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
#                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
#                     [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0]]
#                     )
# listb = get_block(matrice)
# print(listb)
# listt = transform_frequence(listb)
# print(listt)
# listdt = detransform_frequence(listt)
# print(listdt)

#question 8 
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


def compress_mode2(image, seuil):
    yCbCr_image = YCbCr(image)
    padded_image = add_padding(yCbCr_image)
    blocks = get_block(padded_image)
    blocks[np.abs(blocks) < seuil] = 0
    subsampled_Cb = matrice_sousechantillon(Cb(padded_image))
    subsampled_Cr = matrice_sousechantillon(Cr(padded_image))

    return blocks, subsampled_Cb, subsampled_Cr

matrice = np.array([[1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                    [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                    [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                    [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                    [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                    [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                    [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0],
                    [1,2,5,6,0,0,0,0,1,2,5,6,0,0,0,0],[3,4,7,8,0,0,0,0,3,4,7,8,0,0,0,0]]
                    )

# listb = get_block(matrice)
# print(listb)
# listt = transform_frequence(listb)
# print(listt)
# listtcoeff= filter_coeff(listt,10)
# print(listtcoeff)


#question 9 

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
        
        listblockCb_ech=matrice_sousechantillon(imageYCbCr[:,:,1])
        listblockCb=filter_coeff(listblockCb_ech,seuil)

        listblockCr_ech=matrice_sousechantillon(imageYCbCr[:,:,2])
        listblockCr=filter_coeff(listblockCr_ech,seuil)
        return (listblockY, listblockCb, listblockCr)    



# imagecompress=compress("test.png",0,10)
# print(imagecompress)
# imagecompress=compress("test.png",1,10)
# print(imagecompress)
# imagecompress=compress("test.png",2,10)
# print(imagecompress)


#question 10

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


# write_im_header("txtFile.txt","150_210.png","mode 0","RLE")

#question11
def write_im_header_block(pathTextFile,pathImageFile,mode,encoding):
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
        listeblockYTR=transform_frequence(listeblockY)
        listeblockCbTR=transform_frequence(listeblockCb)
        listeblockCrTR=transform_frequence(listeblockCr)



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
write_im_header_block("txtFile.txt","150_210.png", 0,"RLE")