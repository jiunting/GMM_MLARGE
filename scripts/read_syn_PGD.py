#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Fen 8 11:35:19 2022

@author:Tim Lin
@email:jiunting@uoregon.edu
"""


import numpy as np
import obspy
import glob
import matplotlib.pyplot as plt
import pandas as pd

def D2PGD(E,N,Z):
    '''
        or PGV or PGA
    '''
    D=(E**2.0+N**2.0+Z**2.0)**0.5
    PGD=np.maximum.accumulate(D)
    return PGD

def WGRW12(y, mode):
    '''
        Compute MMI with Worden 2012 given either PGA or PGV
        Input:
        y:      Array - Ground-motion intensity measure in cm/s/s if PGA
        or cm/s if PGV
        mode:   Integer - 0 if PGA, 1 if PGV
        Returns:
        MMI:    Array - Intensity measure in modified mercalli intensity
        '''
    if (mode == 0):
        pgalen = len(y)
        MMI = np.zeros(pgalen)
        pgalarge = np.where(np.log10(y) >= 1.57)[0]
        pgasmall = np.where((np.log10(y) < 1.57) & (y > 0))[0]
        pgazero = np.where(y == 0)[0]
        
        MMI[pgalarge] = 3.70*np.log10(y[pgalarge])-1.60
        MMI[pgasmall] = 1.55*np.log10(y[pgasmall])+1.78
        MMI[pgazero] = -10
    else:
        pgvlen = len(y)
        MMI = np.zeros((pgvlen,1))
        pgvlarge = np.where(np.log10(y) >= 0.53)
        pgvsmall = np.where(np.log10(y) < 0.53)
        MMI[pgvlarge] = 3.16*np.log10(y[pgvlarge])+2.89
        MMI[pgvsmall] = 1.47*np.log10(y[pgvsmall])+3.78
    return(MMI)

rupt_case = "Chile_full_new_subduction.017291"


# load synthetic STA & fault
STA = pd.read_csv("/Users/jtlin/Documents/Project/GMM_MLARGE/data/Fakequakes/stainfo/chile_vs30_syntheticSTA.xyz",header=None,sep='\s+',names=['lon','lat','vs30'],skiprows=0)
fault = np.genfromtxt("/Users/jtlin/Documents/Project/GMM_MLARGE/data/Fakequakes/model_info/chile.fault")
rupt = np.genfromtxt("/Users/jtlin/Documents/Project/GMM_MLARGE/data/Fakequakes/output/ruptures/%s.rupt"%(rupt_case))

# the base dir of data
#sacs_dir = "/Users/jtlin/Documents/Project/GMM_MLARGE/data/Fakequakes/output/waveforms/Chile_full_new_subduction.024391"
sacs_dir = "/Users/jtlin/Documents/Project/GMM_MLARGE/data/Fakequakes/output/waveforms/%s"%(rupt_case)
# find all sacs
Z_files = glob.glob(sacs_dir+"/"+"*.bb.HNZ.sac")
Z_files.sort()

# loop through all the Z files, get PGA and MMI
T = np.arange(102)*5+5
sav_PGA = []
sav_MMI = []
sav_A = []
sav_dist = []
sav_idx = []
for Z_file in Z_files:
    staidx = int(Z_file.split("/")[-1].split(".")[0].replace("A",""))
    E_file = Z_file.replace("HNZ","HNE")
    N_file = Z_file.replace("HNZ","HNN")
    Z = obspy.read(Z_file)
    E = obspy.read(E_file)
    N = obspy.read(N_file)
    A = (Z[0].data**2 + E[0].data**2 + N[0].data**2) ** 0.5
    sav_A.append(np.interp(T,Z[0].times(),A)) #downsampling A
    MMI = WGRW12(A*100, mode=0) #A to MMI. Convert input unit m/s^2 to cm/s^2
    sav_MMI.append(max(MMI))
    sav_PGA.append(max(A))
    sav_idx.append(staidx)
    dist = obspy.geodetics.locations2degrees(lat1=Z[0].stats.sac.evla,long1=Z[0].stats.sac.evlo,lat2=Z[0].stats.sac.stla,long2=Z[0].stats.sac.stlo)
    sav_dist.append(dist)
    #plt.plot(T,np.interp(T,E[0].times(),E[0].data)/max(np.interp(T,E[0].times(),E[0].data))+dist,color=[0.5,0.5,0.5],linewidth=0.5)
    plt.plot(T,np.interp(T,E[0].times(),E[0].data)+dist,color=[0.5,0.5,0.5],linewidth=0.5)

plt.xlabel('Time (s)',fontsize=14)
plt.ylabel('Distance ($\degree$)',fontsize=14)

# plot dist v.s. PGA/MMI
plt.figure()
plt.subplot(1,2,1)
plt.plot(sav_dist,sav_PGA,'k.')
plt.xlabel('Distance ($\degree$)',fontsize=14)
plt.ylabel('PGA (m/s^2)',fontsize=14)
plt.subplot(1,2,2)
plt.plot(sav_dist,sav_MMI,'k.')
plt.xlabel('Distance ($\degree$)',fontsize=14)
plt.ylabel('MMI',fontsize=14)



for i in range(len(sav_dist)):
    plt.plot(T,sav_A[i]/max(sav_A[i])+sav_dist[i],color=[0.5,0.5,0.5],linewidth=0.5)

plt.xlabel('Time (s)',fontsize=14)
plt.ylabel('Distance ($\degree$)',fontsize=14)


# make a map with synthetic stations
plt.plot(fault[:,1],fault[:,2],'k.')
plt.plot(STA['lon'],STA['lat'],'r^')
plt.title('Synthetic Stations')
plt.show()



# plot rupt and PGA
# get slip
slip = (rupt[:,8]**2+rupt[:,9]**2)**0.5
plt.scatter(rupt[:,1],rupt[:,2],c=slip,cmap='magma')
plt.scatter(STA.iloc[sav_idx]['lon'],STA.iloc[sav_idx]['lat'],c=sav_PGA,cmap='jet')
plt.colorbar()
plt.show()
