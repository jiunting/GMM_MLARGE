#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:38:38 2022 

@author:Tim Lin
@email:jiunting@uoregon.edu


Plot synthetic SM data at 515 virtual stations
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import pandas as pd
import obspy
import glob

# set rupture case
base_path = "/Users/jtlin/Documents/Project/GMM_MLARGE/data/Fakequakes/output/"
rupt_file = "Chile_full_new_subduction.024513.rupt" #
wave_dir = base_path+"waveforms/"+rupt_file.replace('.rupt','')
rupt_file = base_path+"ruptures/"+rupt_file



# plotting basemap
plt.subplot(1,2,1)
map = Basemap(projection='cyl',resolution='f',llcrnrlon=-76.5,llcrnrlat=-43,urcrnrlon=-67,urcrnrlat=-21.5,fix_aspect=False)
map.bluemarble(alpha=0.5)
#fig = map.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= True)
#fig.set_alpha(0.5)
map.drawcoastlines(linewidth=0.5)
map.drawcountries(linewidth=0.4)
lats = map.drawparallels(np.arange(-90,90,5),labels=[1,0,0,1],color='w',linewidth=0.5)
lons = map.drawmeridians(np.arange(-180,180,5),labels=[1,0,0,1],color='w',linewidth=0.5)


# plot fault
rupt = np.genfromtxt(rupt_file)
SS = rupt[:,8]
DS = rupt[:,9]
slip = (SS**2+DS**2)**0.5
rupt_time = rupt[:,12]
idx = np.where(slip>0)[0]
idx_hypo = np.where((slip>0) & (rupt_time==0))[0][0]
#plt.scatter(rupt[idx,1],rupt[idx,2],c=slip[idx],s=8,marker='s', cmap='rainbow')
plt.scatter(rupt[idx,1],rupt[idx,2],c=slip[idx],s=6, vmax=50,marker='s', cmap='gnuplot2')
plt.plot(rupt[idx_hypo,1],rupt[idx_hypo,2],'r*',ms=15,markeredgecolor=[0,0,0],lw=0.8)
ax = plt.gca()
fig = plt.gcf()
cbaxes = fig.add_axes([0.13, 0.4, 0.024, 0.18 ])
clb=plt.colorbar(cax=cbaxes, orientation='vertical',label='slip (m)')
clb.set_label('slip(m)', labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=0.5,size=1.5)
#plt.colorbar()
#plt.show()


#plot virtual stations
sta = np.genfromtxt("/Users/jtlin/Documents/Project/GMM_MLARGE/data/Fakequakes/stainfo/chile_vs30_syntheticSTA.xyz")
sac_Zs = glob.glob(wave_dir+"/*bb*HNZ*sac")
sac_Zs.sort()
scale_lon = 150 # scale the time range by this value
scale_lat = 5
sav_lon = []
sav_lat = []
sav_PGA = []
sav_tr_E = []
sav_tr_N = []
sav_tr_Z = []
sav_i_sta = []
t = None
#plt.plot(sta[:,0], sta[:,1], '^', color=[0.8,0.8,0.8])
fig = plt.gcf()
for sac_Z in sac_Zs:
    i_sta = int(sac_Z.split('/')[-1].split('.')[0].replace('A',''))
    #plt.plot(sta[i_sta,0],sta[i_sta,1],'ro',ms=3,markeredgecolor='k')
    Z = obspy.read(sac_Z)
    E = obspy.read(sac_Z.replace('HNZ','HNE'))
    N = obspy.read(sac_Z.replace('HNZ','HNN'))
    PGA = max((E[0].data**2 + N[0].data**2 )**0.5) # acc on horizontal
    #plt.plot(E[0].times()/scale_lon+sta[i_sta,0], (E[0].data/max(E[0].data))/scale_lat+sta[i_sta,1], color='k', lw=0.1)
    #plt.plot(E[0].times()/scale_lon+sta[i_sta,0], (E[0].data/scale_lat)+sta[i_sta,1],color=[0.2,0.2,0.2], lw=0.1)
    sav_tr_E.append(E[0].data)
    sav_tr_N.append(N[0].data)
    sav_tr_Z.append(Z[0].data)
    if t is None:
        t = E[0].times()
    sav_lon.append(sta[i_sta,0])
    sav_lat.append(sta[i_sta,1])
    sav_i_sta.append(i_sta)
    sav_PGA.append(PGA)
    Z.clear()
    E.clear()
    N.clear()

non_sta = np.array(list(set(np.arange(525)).symmetric_difference(set(sav_i_sta))))
plt.sca(ax)
#plt.plot(sta[non_sta,0],sta[non_sta,1],'o',markerfacecolor='none',markeredgecolor=[0.3,0.3,0.3],lw=0.5,markersize=3.5)
plt.scatter(sav_lon, sav_lat, c=sav_PGA, marker='^', s=50, edgecolor='k',lw=0.5, cmap='jet')
#plt.colorbar()
fig = plt.gcf()
cbaxes = fig.add_axes([0.13, 0.65, 0.024, 0.18 ])
clb=plt.colorbar(cax=cbaxes, orientation='vertical',label=r'PGA (m/s$^2$)')
clb.set_label(r'PGA(m/s$^2$)',labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=0.5,size=1.5)

plt.subplot(1,2,2)
for i,lat in enumerate(sav_lat):
    plt.plot(t,(sav_tr_E[i]/scale_lat)+lat,color=[0.3,0.3,0.3], lw=0.05)
    plt.plot(t+512,(sav_tr_N[i]/scale_lat)+lat,color=[1,0.3,0.3], lw=0.05)
    plt.plot(t+1024,(sav_tr_Z[i]/scale_lat)+lat,color=[0.3,0.3,1], lw=0.05)

props = dict(boxstyle='round', facecolor='white', alpha=0.7)
plt.plot([512,512],[-43, -21.5],'k',lw=1.5)
plt.plot([1024,1024],[-43, -21.5],'k',lw=1.5)
plt.text(50,-23.5,'E',fontsize=10,bbox=props)
plt.text(50+512,-23.5,'N',fontsize=10,bbox=props)
plt.text(50+1024,-23.5,'Z',fontsize=10,bbox=props)
#plot scale
plt.errorbar(50,-42,3/scale_lat,capsize=5)
plt.text(80,-42,'3 (m/s$^2$)',verticalalignment='center')
#plt.title('Synthetic (E)')
plt.xticks([200,400, 200+512,400+512, 200+1024,400+1024],['200','400','200','400','200','400'])
plt.yticks([],[])
plt.xlabel('Time (s)',fontsize=14,labelpad=0)
plt.ylim([-43, -21.5])
plt.xlim([0, 512*3])
plt.subplots_adjust(left=0.08,top=0.95,right=0.98,bottom=0.095,wspace=0.07)
plt.savefig('../results/SM_024513.png',dpi=300)
plt.show()
