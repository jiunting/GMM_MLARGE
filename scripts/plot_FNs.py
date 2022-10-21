# Read predicted data and real data (assuming synthetic HF data are real data), get their MMI timeseries and calculate warning time.

import numpy as np
import pandas as pd
import glob
import obspy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


MMIs = np.arange(3,8+0.1,0.1)
dists = np.linspace(0,600,40)


N_stable = 0

# load data directly
Z_warning = np.load('Z_warning_N%d.npy'%(N_stable),allow_pickle=True)
Z_warning = Z_warning.item()
TPTN = np.load('TPTN_N%d.npy'%(N_stable),allow_pickle=True)
TPTN = TPTN.item()
TP = np.load('TP_N%d.npy'%(N_stable),allow_pickle=True)
TP = TP.item()
TOT = np.load('TOT_N%d.npy'%(N_stable),allow_pickle=True)
TOT = TOT.item()
TOT_P = np.load('TOTP_N%d.npy'%(N_stable),allow_pickle=True)
TOT_P = TOT_P.item()

FN = np.load('FN_N%d.npy'%(N_stable),allow_pickle=True)
FN = FN.item()
FN1 = np.load('FN1_N%d.npy'%(N_stable),allow_pickle=True)
FN1 = FN1.item()
FN2 = np.load('FN2_N%d.npy'%(N_stable),allow_pickle=True)
FN2 = FN2.item()
FN3 = np.load('FN3_N%d.npy'%(N_stable),allow_pickle=True)
FN3 = FN3.item()

#-------combine the Z_warning and calculate avg warning time and accuracy------
#import seaborn as sns
#sns.set()
plt.subplot(1,3,1)
Z_FN = np.array([[np.NaN]*len(dists)] * len(MMIs))
for k in FN.keys():
    Z_FN[k[1],k[0]] = FN[k]

contour_lines = [30,60,90,120]
vrange = (0,120)
plt.pcolor(dists,MMIs,Z_FN,cmap='jet')
plt.xlabel('Distance (km)',fontsize=14,labelpad=0)
plt.ylabel('MMI',fontsize=14,labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=0.5,length=0.5,size=1.2,labelsize=10)
clb = plt.colorbar(location='top',pad=0.01)
clb.ax.tick_params(pad=0.5,length=0.5,size=1.5,rotation=0,labelsize=10)
clb.set_label('All FN', rotation=0,labelpad=5,size=12)
#CS = plt.contour(dists,MMIs,Z, contour_lines,vmin=vrange[0],vmax=vrange[1], colors='k', linestyles='--')
#CS.clabel(inline=True,fontsize=10)
#clb.set_ticklabels(TICKS,length=0.5,size=0.5,labelsize=22)
#plt.savefig('testwarning.png',dpi=300)
#plt.close()

#-----calculate warning accuracy-----
#TP is when warning time>0
plt.subplot(1,3,2)
Z_FN2 = np.array([[np.NaN]*len(dists)] * len(MMIs))
#for k in TP.keys():
#    Z_acc[k[1],k[0]] = 100*(TPTN[k]/TOT[k])

for k in FN2.keys():
    Z_FN2[k[1],k[0]] = (FN2[k]/FN[k])*100

contour_lines = [30,60,90]
vrange = (0,100)
plt.pcolor(dists,MMIs,Z_FN2,cmap='inferno')
plt.xlabel('Distance (km)',fontsize=14,labelpad=0)
#plt.ylabel('MMI',fontsize=14,labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=0.5,length=0.5,size=1.2,labelsize=10)
clb = plt.colorbar(location='top',pad=0.01)
clb.ax.tick_params(pad=0.5,length=0.5,size=1.5,rotation=0,labelsize=10)
#clb.set_label('Accuracy (%)', rotation=0,labelpad=5,size=14)
clb.set_label('Too late warning(%)', rotation=0,labelpad=5,size=12)
#CS = plt.contour(dists,MMIs,Z_acc, contour_lines,vmin=vrange[0],vmax=vrange[1], colors='k', linestyles='--')
#CS.clabel(inline=True,fontsize=10)



#-----number of data-----
#TP is when warning time>0
plt.subplot(1,3,3)
Z_FN3 = np.array([[np.NaN]*len(dists)] * len(MMIs))
for k in FN3.keys():
    Z_FN3[k[1],k[0]] = (FN3[k]/FN[k])*100

plt.pcolor(dists,MMIs,Z_FN3,cmap='inferno')
plt.xlabel('Distance (km)',fontsize=14,labelpad=0)
#plt.ylabel('MMI',fontsize=14,labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=0.5,length=0.5,size=1.2,labelsize=10)
clb = plt.colorbar(location='top',pad=0.01)
clb.ax.tick_params(pad=0.5,length=0.5,size=1.5,rotation=0,labelsize=10)
#clb.set_label('Accuracy (%)', rotation=0,labelpad=5,size=14)
clb.set_label('Fail prediction(%)', rotation=0,labelpad=5,size=12)

#Make subplots closer/farer
plt.subplots_adjust(left=0.05,top=0.85,right=0.97,bottom=0.14,wspace=0.14)




plt.savefig('FNs%d.png'%(N_stable),dpi=300)
plt.show()
plt.close()

import sys
sys.exit()


#some test plot
Z_num = np.array([[np.NaN]*len(dists)] * len(MMIs))
for k in TOT.keys():
    Z_num[k[1],k[0]] = TOT[k]

plt.pcolor(dists,MMIs,Z_num,cmap='magma')
plt.xlabel('Distance (km)',fontsize=14,labelpad=0)
plt.ylabel('MMI',fontsize=14,labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=0.5,length=0.5,size=1.5,labelsize=12)
clb = plt.colorbar(location='top',pad=0.01)
clb.ax.tick_params(pad=0.5,length=0.5,size=1.5,labelsize=12)
clb.set_label('Accuracy (%)', rotation=0,labelpad=5,size=14)
plt.contour(dists,MMIs,Z_acc, contour_lines,vmin=vrange[0],vmax=vrange[1], colors='k', linestyles='--')
#plt.savefig('testwarning.png',dpi=300)
plt.show()
plt.close()


#quick plot and test
#get real MMI
r = [real['MMI'][a][-1] for a in avail_stas]

plt.subplot(1,2,1)
plt.scatter(tmp_pred['# lon'],tmp_pred['lat'],c=tmp_pred[metric],cmap='jet',vmin=0,vmax=7.0)
plt.colorbar()
plt.subplot(1,2,2)
plt.scatter(tmp_pred['# lon'],tmp_pred['lat'],c=r,cmap='jet',vmin=0,vmax=7.0)
plt.colorbar()
plt.savefig('testMMI.png',dpi=300)
plt.close()



