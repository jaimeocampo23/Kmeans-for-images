#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 21:20:38 2020

@author: jaimecalderonocampo
"""
from skimage import io, morphology
import matplotlib.pyplot as plt 
import numpy as np 

Brain = io.imread('Brain.png')
X, Y = Brain.shape
Imagen = np.zeros([X, Y]) 
Seg = np.zeros([X, Y]) 
Seg = Seg.astype('uint8')
# =============================================================================
D = np.asmatrix(np.ravel(Brain))
Caracteristicas, N = D.shape
Cluster = 3
# centroides = np.round(np.random.rand(Cluster,Caracteristicas) * 255)
centroides = np.load('Centros.npy')
stop = 0
UFinal = np.zeros([Cluster, N])
# =============================================================================
Umatriz = np.zeros((Cluster, N)) 
UActualizada = np.zeros([Cluster, N])
         
while stop == 0:
    
    distancia = np.zeros([Cluster, N])
    for i in range(Cluster):
        for k in range(N):
            suma = 0
            for j in range(Caracteristicas):
                suma += (D[j, k] - centroides[i, j])**2
            distancia[i, k] = np.sqrt(suma)
              
    Umatriz = np.zeros((Cluster, N))           
    for k in range(N):
        indice = np.argmin(distancia[:, k])
        Umatriz[indice, k] = 1

    UActualizada = np.zeros([Cluster, N])
    for j in range(N):
        UFinal[:, j] = np.where(distancia[:, j] == np.min(distancia[:, j]), 1, 0)

    Suma = 0
    for i in range(Cluster):
        for j in range(N):
            if UFinal[i, j] != Umatriz[i, j]:
                Suma += 1
                    
    Umatriz = UFinal             
    if Suma == 0:
        stop = 1
        
    for i in range(Cluster):
        for j in range(Caracteristicas):
            S = 0
            for k in range(N):
                S += Umatriz[i, k] * D[j, k]
            centroides[i, j] = S / np.sum(Umatriz[i, :])

U = UFinal  
# np.save('Centros', centroides)

        
k = 0
for i in range(X):
    for j in range(Y):
        if np.argmax(U[:, k]) == 0:
            Imagen[i, j] = 0
        elif np.argmax(U[:, k]) == 1:
            Imagen[i, j] = 125
        else:
            Imagen[i, j] = 255
        k += 1
D = morphology.area_closing(Imagen)
plt.figure(0)
plt.imshow(Imagen, cmap='gray')
plt.figure(1)
plt.imshow(D, cmap='gray')
plt.ion()
