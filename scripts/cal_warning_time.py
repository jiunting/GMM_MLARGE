# Read predicted data and real data (assuming synthetic HF data are real data), get their MMI timeseries and calculate warning time.

import numpy as np
import pandas as pd
import glob
import obspy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

EQID = np.load("/home/jtlin/MLARGE_pred/data/Run031_test_EQID.npy") #size=10000
EQinfo = pd.read_csv("/home/jtlin/MLARGE_pred/data/Chile_full_small2_small3_SRC.EQinfo",sep='\s+',skiprows=0)
STA = pd.read_csv("/hdd/jtlin/Chile_FQ/Chile_SM/data/station_info/chile_vs30_syntheticSTA.gflist",header=None,sep='\s+',names=['stname','stlo','stla'],skiprows=1,usecols=[0,1,2])

metric = 'MMI_montalva17_wgrw12' #metric used for comparision

pred_dir = "/hdd/jtlin/GM/MLARGE_pred/Shakefile/Test031_Montalva17" #Shake based on MLARGE predicted fault
#e.g. Test031.case0000152.epo085.shake means shake for idx=152, 85th epoch, each file has 525 records

real_dir = "/hdd/jtlin/Chile_FQ/Chile_SM/output/waveforms" #Real shake from HF synthetic


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


def get_syn_MMI(fname):
    Zs = glob.glob(fname+"/*bb.HNZ.sac")
    Zs.sort()
    Data = {'t':None,'MMI':{}}
    for Z in Zs:
        sta = Z.split('/')[-1].split('.')[0] #e.g. A000.bb.HNZ.sac > A000
        E = Z.replace('bb.HNZ','bb.HNE')
        N = Z.replace('bb.HNZ','bb.HNN')
        # read data
        Z = obspy.read(Z)
        E = obspy.read(E)
        N = obspy.read(N)
        # get PGA timeseries, convert to MMI and interpolate to T
        A = (E[0].data**2+N[0].data**2)**0.5
        PGA = np.maximum.accumulate(A)
        MMI = WGRW12(PGA*100, mode=0)
        if Data['t'] is None:
            Data['t'] = E[0].times()
        Data['MMI'][sta] = MMI
    return Data
    
def add_one(k,d):
    if k in d:
        d[k] += 1
    else:
        d[k] = 1
    return d


MMIs = np.arange(3,8+0.1,0.1)
dists = np.linspace(0,600,40)


N_stable = 0 


Z_warning = {} # 
TP = {}
TPTN = {} #number of TP+TN
FP = {}
FN = {}
TN = {}
TOT = {} #total number
TOT_P = {} #total of positive number
TOT_N = {} #total of negative number
FN1 = {} #number of unstable warning
FN2 = {} #number of Too late warning
FN3 = {} #number of fail prediction

T = np.arange(102)*5+5

prepand_idx_st = {'Chile_full_new':0, 'Chile_small_new':27200, 'Chile_small_new2':36800, 'Chile_small_new3':46400} #start of the index in EQinfo. e.g. idx=10 for Chile_small_new is EQinfo[10+27200]

"""
#skip the loading step
for i,eqid in enumerate(EQID):
    if i%20==0:
        print('Now in:',i,len(EQID))
    #e.g. eqid='Chile_small_new3_149713' > Chile_small_new3_subduction.149713
    rnum = eqid.split('_')[-1]
    rprepand = '_'.join(eqid.split('_')[:3])
    real_idx = prepand_idx_st[rprepand]+int(rnum) # this is the real idx in EQinfo
    # load read data (from HF synthetic)
    fname = real_dir+'/'+'_'.join(eqid.split('_')[:-1]+['subduction'])+'.'+rnum
    real = get_syn_MMI(fname)
    avail_stas = list(real['MMI'].keys())
    # load ML predicted data e.g. Test031.case0000152.epo085.shake
    pred = {'t':T, 'MMI':{}, 'dist':{}, 'Rrupt':{}}
    for epo in range(102):
        tmp_pred_file = pred_dir+"/"+"Test031.case%07d.epo%03d.shake"%(i,epo)
        tmp_pred = pd.read_csv(tmp_pred_file) #prediction at this time
        tmp_pred = tmp_pred.join(STA['stname'])
        tmp_pred.set_index('stname',inplace=True)
        tmp_pred = tmp_pred.loc[avail_stas]
        for sta,row in tmp_pred.iterrows():
            Rrupt = row['Rrupt(km)']
            if sta in pred['MMI']:
                pred['MMI'][sta].append(row[metric])
                pred['Rrupt'][sta].append(Rrupt)
            else:
                # get EQinfo and dist info
                evlo = EQinfo.iloc[real_idx]['Hypo_lon[2]']
                evla = EQinfo.iloc[real_idx]['Hypo_lat[3]']
                stlo = STA.iloc[int(sta[1:])]['stlo']
                stla = STA.iloc[int(sta[1:])]['stla']
                dist,_,_ = obspy.geodetics.base.gps2dist_azimuth(lat1=evla,lon1=evlo,lat2=stla,lon2=stlo)
                dist = dist*1e-3 #km
                pred['MMI'][sta] = [row[metric]]
                pred['dist'][sta] = dist #dist always the same for different epo
                pred['Rrupt'][sta] = [Rrupt] #Rrupt change because predicted fault change
    #-----compare real with pred-----
    for sta in avail_stas:
        dist = pred['dist'][sta]
        dist_idx = np.where(dists>=dist)[0]
        if len(dist_idx)==0:
            continue #station too far
        dist_idx = dist_idx[0]
        MMI_real = real['MMI'][sta]
        MMI_pred = np.array(pred['MMI'][sta])
        for MMI_idx, thresh_MMI in enumerate(MMIs):
            # the corresponding grid for this (dist,thresh_MMI) i.e. dist_idx, MMI_idx
            TOT = add_one((dist_idx,MMI_idx),TOT)
            pred_t_pass_thresh = np.where(MMI_pred>=thresh_MMI)[0]
            true_t_pass_thresh = np.where(MMI_real>=thresh_MMI)[0]
            # get the time (second) when true/pred MMI pass the threshold
            pred_t_pass_thresh = T[pred_t_pass_thresh]
            true_t_pass_thresh = real['t'][true_t_pass_thresh]
            #number of positive or negative only
            if len(true_t_pass_thresh)!=0:
                TOT_P = add_one((dist_idx,MMI_idx), TOT_P)
            else:
                TOT_N = add_one((dist_idx,MMI_idx), TOT_N)
            # there are 4 cases, 1. len(true_t_pass_thresh)!=0 or ==0 and len(pred_t_pass_thresh)!=0 or ==0, consider the case that both !=0 and t_pred<t_true for warning time.
            if (len(true_t_pass_thresh)!=0 and len(pred_t_pass_thresh)!=0):
                # considering that prediction has to be stable, not just randomly guess for the next N iteration
                if len(pred_t_pass_thresh)<N_stable+1:
                    FN = add_one((dist_idx,MMI_idx), FN)
                    FN1 = add_one((dist_idx,MMI_idx), FN1)
                    continue #unstable warning, fail to predict True
                # find a stable prediction
                stable_detc = False
                #N_stable = 1 #define earlier
                for i_detc in range(N_stable,len(pred_t_pass_thresh)):
                    if pred_t_pass_thresh[i_detc]-pred_t_pass_thresh[i_detc-N_stable]==5*N_stable:
                        stable_detc = True
                        i_detc -= N_stable # current and the next has the same prediction, use the first detection
                        break
                if stable_detc:
                    # calculate warning time
                    warning_t = true_t_pass_thresh[0]-pred_t_pass_thresh[i_detc]
                else:
                    FN = add_one((dist_idx,MMI_idx), FN) #unstable prediction
                    FN1 = add_one((dist_idx,MMI_idx), FN1) #unstable prediction
                    continue
                if warning_t>=0: #successful warning
                    #TP += 1
                    if (dist_idx,MMI_idx) in Z_warning:
                        Z_warning[(dist_idx,MMI_idx)].append(warning_t)
                    else:
                        Z_warning[(dist_idx,MMI_idx)] = [warning_t]
                    TP = add_one((dist_idx,MMI_idx), TP)
                    TPTN = add_one((dist_idx,MMI_idx), TPTN)
                else: #too late warning
                    #warning_t = None
                    FN = add_one((dist_idx,MMI_idx), FN)
                    FN2 = add_one((dist_idx,MMI_idx), FN2)
                    continue
            elif (len(true_t_pass_thresh)!=0 and len(pred_t_pass_thresh)==0): # fail to predict the true MMI (i.e. FN)
                FN = add_one((dist_idx,MMI_idx), FN)
                FN3 = add_one((dist_idx,MMI_idx), FN3)
                continue
            elif (len(true_t_pass_thresh)==0 and len(pred_t_pass_thresh)!=0): # overestimation the true MMI (i.e. FP)
                #FP += 1
                FP = add_one((dist_idx,MMI_idx), FP)
                continue
            elif (len(true_t_pass_thresh)==0 and len(pred_t_pass_thresh)==0): # (i.e. TN)
                #TN += 1
                TN = add_one((dist_idx,MMI_idx), TN)
                TPTN = add_one((dist_idx,MMI_idx), TPTN)
                continue
    #if i==20:
    #    break


np.save('Z_warning_N%d.npy'%(N_stable),Z_warning)
np.save('TPTN_N%d.npy'%(N_stable),TPTN)
np.save('TP_N%d.npy'%(N_stable),TP)
np.save('TN_N%d.npy'%(N_stable),TN)
np.save('FP_N%d.npy'%(N_stable),FP)
np.save('FN_N%d.npy'%(N_stable),FN)
np.save('FN1_N%d.npy'%(N_stable),FN1)
np.save('FN2_N%d.npy'%(N_stable),FN2)
np.save('FN3_N%d.npy'%(N_stable),FN3)
np.save('TOT_N%d.npy'%(N_stable),TOT)
np.save('TOTP_N%d.npy'%(N_stable),TOT_P)
np.save('TOTN_N%d.npy'%(N_stable),TOT_N)
"""


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

#find warning time at dist=200 and 300 for all MMIs
d = 200 #300
d_idx = np.where(np.abs(dists-d)==np.min(np.abs(dists-d)))[0][0]
all_t = []
for mmi_idx,mmi in enumerate(MMIs):
    try:
        all_t += Z_warning[(d_idx,mmi_idx)]
    except:
        pass

print('Median warning time at %.1f km is:%fs'%(d,np.median(all_t)))

#find warning time at MMI4 and 5 for all distance
mmi = 5 #5
mmi_idx = np.where(np.abs(MMIs-mmi)==np.min(np.abs(MMIs-mmi)))[0][0]
all_t = []
for d_idx,d in enumerate(dists):
    try:
        all_t += Z_warning[(d_idx,mmi_idx)]
    except:
        pass

print('Median warning time at MMI=%.1f is:%f'%(mmi,np.median(all_t)))


#-------combine the Z_warning and calculate avg warning time and accuracy------
#import seaborn as sns
#sns.set()
plt.subplot(1,3,1)
Z = np.array([[np.NaN]*len(dists)] * len(MMIs))
for k in Z_warning.keys():
    Z[k[1],k[0]] = np.mean(Z_warning[k])

contour_lines = [30,60,90,120]
vrange = (0,120)
Z = np.array(Z)
plt.pcolor(dists,MMIs,Z,vmin=vrange[0],vmax=vrange[1],cmap='rainbow_r')
plt.xlabel('Distance (km)',fontsize=14,labelpad=0)
plt.ylabel('MMI',fontsize=14,labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=0.5,length=0.5,size=1.2,labelsize=10)
clb = plt.colorbar(location='top',pad=0.01)
clb.ax.tick_params(pad=0.5,length=0.5,size=1.5,rotation=0,labelsize=10)
clb.set_label('Warning Time (s)', rotation=0,labelpad=5,size=12)
CS = plt.contour(dists,MMIs,Z, contour_lines,vmin=vrange[0],vmax=vrange[1], colors='k', linestyles='--')
CS.clabel(inline=True,fontsize=10)
#clb.set_ticklabels(TICKS,length=0.5,size=0.5,labelsize=22)
#plt.savefig('testwarning.png',dpi=300)
#plt.close()

#-----calculate warning accuracy-----
#TP is when warning time>0
plt.subplot(1,3,2)
Z_acc = np.array([[np.NaN]*len(dists)] * len(MMIs))
#for k in TP.keys():
#    Z_acc[k[1],k[0]] = 100*(TPTN[k]/TOT[k])

for k in TP.keys():
    #Z_acc[k[1],k[0]] = 100*(TP[k]/TOT_P[k])
    if k in FN3:
        Z_acc[k[1],k[0]] = 100*(TP[k]/(TOT_P[k]-FN3[k]))
    else:
        Z_acc[k[1],k[0]] = 100*(TP[k]/TOT_P[k])


contour_lines = [30,60,90]
vrange = (0,100)
plt.pcolor(dists,MMIs,Z_acc,vmin=vrange[0],vmax=vrange[1],cmap='inferno')
#plt.pcolor(dists,MMIs,Z_acc,vmin=vrange[0],vmax=vrange[1],cmap='Blues_r')
plt.xlabel('Distance (km)',fontsize=14,labelpad=0)
#plt.ylabel('MMI',fontsize=14,labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=0.5,length=0.5,size=1.2,labelsize=10)
clb = plt.colorbar(location='top',pad=0.01)
clb.ax.tick_params(pad=0.5,length=0.5,size=1.5,rotation=0,labelsize=10)
#clb.set_label('Accuracy (%)', rotation=0,labelpad=5,size=14)
clb.set_label('Successful Warning (%)', rotation=0,labelpad=5,size=12)
CS = plt.contour(dists,MMIs,Z_acc, contour_lines,vmin=vrange[0],vmax=vrange[1], colors='k', linestyles='--')
CS.clabel(inline=True,fontsize=10)



#-----number of data-----
#TP is when warning time>0
plt.subplot(1,3,3)
Z_num = np.array([[np.NaN]*len(dists)] * len(MMIs))
for k in TOT_P.keys():
    if k in FN3:
        Z_num[k[1],k[0]] = TOT_P[k]-FN3[k]
    else:
        Z_num[k[1],k[0]] = TOT_P[k]



plt.pcolor(dists,MMIs,np.log10(Z_num),cmap='jet')
plt.xlabel('Distance (km)',fontsize=14,labelpad=0)
#plt.ylabel('MMI',fontsize=14,labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=0.5,length=0.5,size=1.2,labelsize=10)
clb = plt.colorbar(location='top',pad=0.01)
clb.ax.tick_params(pad=0.5,length=0.5,size=1.5,rotation=0,labelsize=10)
#clb.set_label('Accuracy (%)', rotation=0,labelpad=5,size=14)
clb.set_label('log(N)', rotation=0,labelpad=5,size=12)

#Make subplots closer/farer
plt.subplots_adjust(left=0.05,top=0.85,right=0.97,bottom=0.14,wspace=0.14)




plt.savefig('testwarning_N%d.png'%(N_stable),dpi=300)
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



