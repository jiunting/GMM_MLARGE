#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 5 15:30:19 2022 

@author:Tim Lin
@email:jiunting@uoregon.edu

"""
import obspy
import pandas as pd
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mlarge.scaling import make_linear_scale
import os
import seaborn as sns
sns.set()

#def D2PGD(E,N,Z):
#    '''
#        or PGV or PGA
#    '''
#    D=(E**2.0+N**2.0+Z**2.0)**0.5
#    PGD=np.maximum.accumulate(D)
#    return PGD
    
def D2PGD(E,N):
    '''
        or PGV or PGA
    '''
    D=(E**2.0+N**2.0)**0.5
    PGD=np.maximum.accumulate(D)
    return PGD

def get_tcs(EQtime,T,E_file,N_file,Z_file):
    #start calculate PGA from 3-comps, dealing with time
    #10/20: Change to horizontal component only, use all time
    '''
        EQtime: time in obspy UTCDateTime format
        T: interpolated time. 0 is the origin time
        E_file: the path of sac data
        '''
    D_E = obspy.read(E_file)
    DD_E = D_E.copy()
    DD_E.detrend('linear')
    DD_E.trim(starttime=EQtime,endtime=EQtime+T[-1]+10,pad=True,fill_value=0)
    D_N = obspy.read(N_file)
    DD_N = D_N.copy()
    DD_N.detrend('linear')
    DD_N.trim(starttime=EQtime,endtime=EQtime+T[-1]+10,pad=True,fill_value=0)
    #D_Z = obspy.read(Z_file)
    #DD_Z = D_Z.copy()
    #DD_Z.detrend('linear')
    #DD_Z.trim(starttime=EQtime,endtime=EQtime+T[-1]+10,pad=True,fill_value=0)
    #make the data=0 at 0 second
    DD_E[0].data = DD_E[0].data-DD_E[0].data[0]
    DD_N[0].data = DD_N[0].data-DD_N[0].data[0]
    #DD_Z[0].data = DD_Z[0].data-DD_Z[0].data[0]
    #Dsum = (DD_E[0].data**2+DD_N[0].data**2+DD_Z[0].data**2)**0.5
    #PGD = D2PGD(DD_E[0].data,DD_N[0].data,DD_Z[0].data)
    Dsum = (DD_E[0].data**2+DD_N[0].data**2)**0.5
    PGD = D2PGD(DD_E[0].data,DD_N[0].data)
    assert np.sum(DD_E[0].times() == DD_N[0].times())==len(DD_E[0].times()), "E-N are different"
    #assert np.sum(DD_E[0].times() == DD_Z[0].times())==len(DD_Z[0].times()), "E-Z are different"
    t = DD_E[0].times()
    interp_data_E = np.interp(T,t,DD_E[0].data)
    interp_data_N = np.interp(T,t,DD_N[0].data)
    #interp_data_Z = np.interp(T,t,DD_Z[0].data)
    interp_data_D = np.interp(T,t,Dsum)
    interp_data_PGD = np.interp(T,t,PGD)
    return interp_data_E,interp_data_N,interp_data_D,interp_data_PGD


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




# the base dir of data
eq_dir = "../data/testsuite_processed"

# list all the EQ folders under the eq_dir, same order as the MLARGE output
EQs = ["Illapel2015","Iquique2014","Maule2010_new","Melinka2016","Iquique_aftershock2014_new"]
titles = ["Illapel2015","Iquique2014","Maule2010","Melinka2016","Iquique_aftershock2014"]
EQt=[obspy.UTCDateTime(2015,9,16,22,54,33),obspy.UTCDateTime(2014,4,1,23,46,47),obspy.UTCDateTime(2010,2,27,6,34,14),obspy.UTCDateTime(2016,12,25,14,22,26),obspy.UTCDateTime(2014,4,3,2,43,13)]
#EQMw=[8.3,8.2,8.8,7.6,7.7]
hypos=[(-71.674,-31.573,22.4),(-70.769,-19.61,25.0),(-72.898,-36.122,22.9),(-73.941,-43.406,38.0),(-70.493,-20.571,22.4)]
T = np.arange(102)*5+5 # sampled data in MLARGE
Tacc = np.arange(0,511,0.01) #sampled data in accel tcs
# MMI threshold that we're interested
MMI_thres_for_cal = [3,4,5,6,7,8,9]
MMI_thres_for_plot = [3,4,5]

# save the detailed PGA/MMI figures
fig_out = "../results/details" #or =None to skip saving
if fig_out:
    if not(os.path.exists(fig_out)):
        os.makedirs(fig_out)

# loop for each earthquake
#for ieq in [0,1,3]:
# -----save the MMI, warning times for all events-----
all_sav_maxMMI_obs = []
all_sav_maxMMI_model = []
all_sav_Time_obs = {}   # time when the value passes the MMI threshold
all_sav_Time_model = {} # time when the value passes the MMI threshold
all_sav_warningTime = {} # warning time at different threshold (all_sav_Time_obs-all_sav_Time_model)
all_sav_stats = {}
for ieq in [0,1,2,3,4]:
#for ieq in [4]:
    # reset for a new event
    sav_warningTime = {} #warning time for different MMI threshold
    sav_stats = {} #model statistics(accuracy, precision, recall) for MMI threshold
    # get EQ info
    evlo,evla,evdep = hypos[ieq]
    # get station info
    sta_info = glob.glob(eq_dir+"/"+EQs[ieq]+"/*_sm.chan")[0]
    sta_info = pd.read_csv(sta_info,header=None,sep='\s+',names=['net','sta','loc','chan','lat','lon','elev','samplerate','gain','units'],skiprows=1)
    # find unique staion, lon, and lat from sta_info
    print(sta_info)
    tmp_uniq_sta, tmp_uniq_lon, tmp_uniq_lat = [],[],[]
    for ii in range(len(sta_info)):
        if sta_info['sta'][ii] not in tmp_uniq_sta:
            tmp_uniq_sta.append(sta_info['sta'][ii])
            tmp_uniq_lon.append(sta_info['lon'][ii])
            tmp_uniq_lat.append(sta_info['lat'][ii])
    # MLARGE shake prediction
    memo_sta_idx = {} # to memorize the location (index) of each station in the .shake grid
    MLARGE_predGM = glob.glob('/Users/jtlin/Documents/Project/MLARGE/results/GMTinp_Test031_GM_Pred/Test031.case000000%1d.epo*.shake'%(ieq))
    MLARGE_predGM.sort()
    # ===========for each time epoch===========
    # -part #1. read shake prediction by MLARGE and get all the predictions from file
    sav_MMI_model = {} #each station has a time series in list/array. Length=102
    for it,t in enumerate(T):
        shake_file = MLARGE_predGM[it]
        shake = pd.read_csv(shake_file)
        for sta,stlo,stla in zip(tmp_uniq_sta,tmp_uniq_lon,tmp_uniq_lat):
            if sta not in memo_sta_idx:
                dist = (shake['# lon']-stlo)**2+(shake['lat']-stla)**2 #get GM at the station
                memo_sta_idx[sta] = np.where(dist==dist.min())[0][0]
            staidx = memo_sta_idx[sta]
            # get GM by staidx
            pred = shake.iloc[staidx]['MMI_montalva17_wgrw12']
            if sta not in sav_MMI_model:
                sav_MMI_model[sta] = np.array([pred])
            else:
                sav_MMI_model[sta] = np.hstack([sav_MMI_model[sta],pred])
                #sav_MMI_model[sta].append(pred)
    # -part #2. deal with obs accel time series data
    sav_MMI_obs = {} #each station has a time series in list/array
    Zs = glob.glob(eq_dir+"/"+EQs[ieq]+"/accel/*.*Z.mseed")
    Zs.sort()
    # for this event, read all the data
    tmp_stainfo = {} # for this event-sta pair, get the info
    for i in Zs:
        E_file = i.replace('Z.','E.')
        N_file = i.replace('Z.','N.')
        Z_file = i
        sta = Z_file.split('/')[-1].split('.')[0]
        # get station gain and info by station name
        idx = sta_info[sta_info['sta']==sta].index
        assert len(idx)==3, "not 3 components!"
        assert sta_info['gain'][idx[0]]==sta_info['gain'][idx[1]]==sta_info['gain'][idx[2]], "different gain"
        stlo = sta_info.lon[idx[0]]
        stla = sta_info.lat[idx[0]]
        #gc = obspy.geodetics.locations2degrees(lat1=evla,long1=evlo,lat2=stla,long2=stlo)
        hypodist, _, _ = obspy.geodetics.base.gps2dist_azimuth(lat1=evla,lon1=evlo,lat2=stla,lon2=stlo)
        hypodist = hypodist*1e-3
        tmp_stainfo[sta] = {'dist':hypodist}
        # get data
        #E, N, Z, D, PGA = get_tcs(EQt[ieq],T,E_file,N_file,Z_file)
        E, N, D, PGA = get_tcs(EQt[ieq],Tacc,E_file,N_file,Z_file)
        # data divided by gain and convert MMI
        gain = sta_info['gain'][idx[0]]
        #E, N, Z, D, PGA = E/gain, N/gain, Z/gain, D/gain, PGA/gain #[cm/s2]
        E, N, D, PGA = E/gain, N/gain, D/gain, PGA/gain #[cm/s2]
        MMI_obs = WGRW12(PGA,0)
        #print('gain=',gain)
        #print(MMI_obs)
        sav_MMI_obs[sta] = MMI_obs
        # make some plot
        if fig_out:
            plt.plot(T,sav_MMI_model[sta],'r')
            plt.plot(Tacc,sav_MMI_obs[sta],'k')
            plt.xlabel('Time (s)',fontsize=14)
            plt.ylabel('MMI',fontsize=14)
            plt.ylim([0,10])
            plt.title('%s (%.1f km)'%(sta,hypodist),fontsize=14)
            plt.legend(['Prediction','Observation'])
            plt.savefig(fig_out+"/"+titles[ieq]+"_"+sta+".png",dpi=300)
            plt.close()
        #plt.show()
    # -part 3. deal with MMI threshold and timing
    for thres in MMI_thres_for_cal:
        sav_maxMMI_obs = []
        sav_maxMMI_model = []
        TP = TN = FP = FN = 0 # to do some stats
        for i_sac,sta in enumerate(sav_MMI_obs.keys()):
            sav_maxMMI_obs.append(max(sav_MMI_obs[sta]))
            sav_maxMMI_model.append(max(sav_MMI_model[sta]))
            #get warning time
            idx_t_obs = np.where(sav_MMI_obs[sta]>=thres)[0]
            idx_t_model = np.where(sav_MMI_model[sta]>=thres)[0]
            # only calculate when TP
            if len(idx_t_obs)!=0 and len(idx_t_model)!=0:
                #TP += 1
                idx_t_obs0 = idx_t_obs[0] #take the first one
                idx_t_model0 = idx_t_model[0]
                t_obs0 = Tacc[idx_t_obs0]
                t_model0 = T[idx_t_model0]
                #warningTime = (idx_t_obs0-idx_t_model0)*5 # sampling interval is 5 s
                warningTime = t_obs0-t_model0
                if warningTime>0:
                    TP += 1
                else:
                    FN += 1 #too late warning
                #saving warning time
                if thres in sav_warningTime:
                    sav_warningTime[thres]['time'].append(warningTime)
                    sav_warningTime[thres]['idx'].append(i_sac)
                else:
                    sav_warningTime[thres] = {'time':[warningTime],'idx':[i_sac]}
            elif len(idx_t_obs)==0 and len(idx_t_model)==0:
                TN += 1
            elif len(idx_t_obs)==0 and len(idx_t_model)!=0:
                FP += 1
            elif len(idx_t_obs)!=0 and len(idx_t_model)==0:
                FN += 1
            # saving the warning time for all events and thresholds
            if len(idx_t_obs)!=0 and len(idx_t_model)!=0:
                if thres in all_sav_warningTime:
                    all_sav_warningTime[thres]['time'].append(warningTime)
                    all_sav_Time_obs[thres]['time'].append(t_obs0)
                    all_sav_Time_model[thres]['time'].append(t_model0)
                else:
                    all_sav_warningTime[thres] = {'time':[warningTime]}
                    all_sav_Time_obs[thres] = {'time':[t_obs0]}
                    all_sav_Time_model[thres] = {'time':[t_model0]}
            else:
                if thres in all_sav_warningTime:
                    all_sav_warningTime[thres]['time'].append(np.nan)
                    all_sav_Time_obs[thres]['time'].append(np.nan)
                    all_sav_Time_model[thres]['time'].append(np.nan)
                else:
                    all_sav_warningTime[thres] = {'time':[np.nan]}
                    all_sav_Time_obs[thres] = {'time':[np.nan]}
                    all_sav_Time_model[thres] = {'time':[np.nan]}
        assert (TP+TN+FP+FN)==len(sav_maxMMI_obs),"TP,TN,FP,FN calculate incorrectly"
        accuracy = ((TP+TN)/(TP+TN+FP+FN))*100
        try:
            precision = (TP/(TP+FP))*100
        except:
            precision = np.nan
        try:
            recall = (TP/(TP+FN))*100
        except:
            recall = np.nan
        sav_stats[thres] = {'acc':accuracy, 'prec':precision, 'recall':recall, 'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}
    all_sav_maxMMI_obs += sav_maxMMI_obs
    all_sav_maxMMI_model += sav_maxMMI_model
    sav_maxMMI_obs = np.array(sav_maxMMI_obs)
    sav_maxMMI_model = np.array(sav_maxMMI_model)
    all_sav_stats[ieq] = sav_stats #save the stats
    print('Number of stations: %d'%(len(sav_maxMMI_obs)))
    print('Stats=',sav_stats)
    #=====make MMI threshold plot=====
    fig = plt.figure(figsize=(9.2,4))
    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    vmin, vmax = 0,120
    cmap = 'rainbow_r'
    #MMI_thres_for_plot = [3,4,5]
    for i,thres in enumerate(MMI_thres_for_plot):
        plt.subplot(1,3,i+1)
        idx_TP = np.array(sav_warningTime[thres]['idx'])
        idx_XTP = [ii for ii in range(len(sav_maxMMI_obs)) if ii not in idx_TP]
        # plot not TP as black dots
        plt.plot(sav_maxMMI_obs[idx_XTP],sav_maxMMI_model[idx_XTP],'ko')
        # if TP, show also the warning time
        tmp_warningTime = np.array(sav_warningTime[thres]['time'])
        plt.scatter(sav_maxMMI_obs[idx_TP],sav_maxMMI_model[idx_TP],c=tmp_warningTime,linewidth=1,edgecolor='k',cmap=cmap,vmin=vmin,vmax=vmax)
        plt.plot([thres,thres],[0,10],'r--')
        plt.plot([0,10],[thres,thres],'r--')
        plt.xlabel('Max Obs. MMI',fontsize=14,labelpad=0.1)
        ax1=plt.gca()
        ax1.tick_params(pad=0.5,length=0.1)
        plt.xticks([2,4,6,8,10],[2,4,6,8,10],fontsize=12)
        plt.yticks([2,4,6,8,10],[2,4,6,8,10],fontsize=12)
        if i in [0]:
            plt.ylabel('Max Pred. MMI',fontsize=14,labelpad=0.1)
        else:
            ax1.tick_params(labelleft=False)
        plt.xlim([1,10])
        plt.ylim([1,10])
        plt.text(1.5,9.0,"FP",bbox=props)
        plt.text(9.0,9.0,"TP",bbox=props,ha='right')
        plt.text(1.5,1.5,"TN",bbox=props)
        plt.text(9.0,1.5,"FN",bbox=props,ha='right')
        plt.title('MMI Threshold = %d'%(thres))
        plt.subplots_adjust(left=0.05,top=0.88,right=0.92,bottom=0.12,wspace=0.05)
    #add colormap/colorbar
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cmap.set_array([])
    cbaxes = fig.add_axes([0.93, 0.12, 0.02,0.76])
    clb = plt.colorbar(cmap,cax=cbaxes, orientation='vertical')
    clb.set_label('warning time (s)',fontsize=14,labelpad=0)
    ax1 = plt.gca()
    ax1.tick_params(pad=0.5,length=0.02)
    plt.suptitle(titles[ieq])
    plt.savefig('../results/%s_WarningTime.png'%(titles[ieq]),dpi=300)
    #plt.savefig('./GM_Pred_Test031/WarningTime.png',dpi=300)
    plt.show()
    #-----plot statistic-----
    all_acc = [sav_stats[thres]['acc'] for thres in MMI_thres_for_cal]
    all_prec = [sav_stats[thres]['prec'] for thres in MMI_thres_for_cal]
    all_recall = [sav_stats[thres]['recall'] for thres in MMI_thres_for_cal]
    plt.figure()
    plt.plot(MMI_thres_for_cal,all_acc,'bo-')
    plt.plot(MMI_thres_for_cal,all_prec,'k^-')
    plt.plot(MMI_thres_for_cal,all_recall,'r*-')
    plt.title(titles[ieq])
    plt.xlabel('MMI Threshold',fontsize=14)
    plt.ylabel('%',fontsize=14)
    plt.grid(True)
    plt.legend(['Accuracy','Precision','Recall'])
    plt.ylim([0,110])
    plt.xticks([3,4,5],fontsize=12)
    ax1=plt.gca()
    ax1.tick_params(pad=-1)
    plt.savefig('../results/%s_Statistic.png'%(titles[ieq]),dpi=300)
    plt.show()


# =====save the results in ../results/npy/*.npy=====
npy_out = "../results/npy"
if not(os.path.exists(npy_out)):
    os.makedirs(npy_out)
    
np.save(npy_out+"/all_sav_maxMMI_obs.npy",all_sav_maxMMI_obs)
np.save(npy_out+"/all_sav_maxMMI_model.npy",all_sav_maxMMI_model)
np.save(npy_out+"/all_sav_Time_obs.npy",all_sav_Time_obs)
np.save(npy_out+"/all_sav_Time_model.npy",all_sav_Time_model)
np.save(npy_out+"/all_sav_warningTime.npy",all_sav_warningTime)
np.save(npy_out+"/all_sav_stats.npy",all_sav_stats)

# ======load the data directly======

all_sav_maxMMI_obs = np.load(npy_out+"/all_sav_maxMMI_obs.npy")
all_sav_maxMMI_model = np.load(npy_out+"/all_sav_maxMMI_model.npy")
all_sav_Time_obs = np.load(npy_out+"/all_sav_Time_obs.npy",allow_pickle=True)
all_sav_Time_model = np.load(npy_out+"/all_sav_Time_model.npy",allow_pickle=True)
all_sav_warningTime = np.load(npy_out+"/all_sav_warningTime.npy",allow_pickle=True)
all_sav_stats = np.load(npy_out+"/all_sav_stats.npy",allow_pickle=True)

all_sav_Time_obs = all_sav_Time_obs.item()
all_sav_Time_model = all_sav_Time_model.item()
all_sav_warningTime = all_sav_warningTime.item()
all_sav_stats = all_sav_stats.item()

# ======confusion matrix calculation=======
MMI_thres_for_cal = [3,4,5,6,7,8,9]
confusion = {}
for thres in MMI_thres_for_cal:
    TP_idx = np.where((all_sav_maxMMI_obs>=thres) & (all_sav_maxMMI_model>=thres))[0]
    FN_idx = np.where((all_sav_maxMMI_obs>=thres) & (all_sav_maxMMI_model<thres))[0]
    FP_idx = np.where((all_sav_maxMMI_obs<thres) & (all_sav_maxMMI_model>=thres))[0]
    TN_idx = np.where((all_sav_maxMMI_obs<thres) & (all_sav_maxMMI_model<thres))[0]
    assert (len(TP_idx)+len(FN_idx)+len(FP_idx)+len(TN_idx))==len(all_sav_maxMMI_obs), "confusion calculation got wrong!"
    confusion[thres] = {'TP':len(TP_idx),'FN':len(FN_idx),'FP':len(FP_idx),'TN':len(TN_idx)}
    



# ======make warning time analysis plot======
# make colorbar for different groups
D = [k for k in all_sav_warningTime.keys()]
D = [3,4,5,6,7]
cm = plt.cm.magma_r(  plt.Normalize(min(D),max(D))(D)  )
norm = matplotlib.colors.Normalize(vmin=min(D), vmax=max(D))
cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap='magma_r')
cmap.set_array([])

# =======make histogram and CDF=======
fig = plt.figure()
ax1 = plt.gca()
ax2 = ax1.twinx() #twinx means same x axis (wanna plot different y)
for i,k in enumerate(D):
    h = ax1.hist(all_sav_warningTime[k]['time'],np.arange(0,200,10),alpha=0.85,color=cm[i])
    #x = (h[1][1:]+h[1][:-1])*0.5
    #ax1.plot(x,h[0],color=cm[i])
    # make cdf
    t = all_sav_warningTime[k]['time']
    t = np.array(t)
    idx = np.where((t>=0) &  (~np.isnan(t)))[0]
    t = t[idx]
    sor_idx = np.argsort(t)
    t = t[sor_idx]
    p = 1. * np.arange(len(t)) / (len(t) - 1)
    ax2.set_xlim(ax1.get_xlim())
    ax2.plot(t,p[::-1],color='k',linewidth=3)
    ax2.plot(t,p[::-1],color=cm[i],linewidth=2)
    ax2.grid(False)
    
ax1.invert_xaxis()
    
ax2.set_ylim([0,1])
ax1.set_xlabel('Warning Time (s)',fontsize=14,labelpad=0)
ax1.set_ylabel('Number of Alerts',fontsize=14,labelpad=0)
ax2.set_ylabel('CDF',fontsize=14,labelpad=0)
ax1.tick_params(pad=1.5,length=0,size=0,labelsize=12)
ax2.tick_params(pad=1.5,length=0,size=4,labelsize=12)
# add colorbar
cbaxes = fig.add_axes([0.18, 0.25, 0.06, 0.52])
clb = plt.colorbar(cmap,cax=cbaxes,alpha=1)
#clb = plt.colorbar(cmap)
clb.set_label('MMI Threshold', rotation=90,labelpad=1,size=14) #the smaller the labelpad is closer to the bar
clb.set_ticks(D)
clb.set_ticklabels(['III','IV','V','VI','VII+'])
clb.ax.tick_params(labelsize=12,size=0)
plt.savefig('../results/MMI_warningTime.png',dpi=300)
plt.show()



fig = plt.figure()
plt.subplot(1,2,1)
plt.plot(all_sav_maxMMI_obs,all_sav_Time_obs[3]['time'],'k.')
plt.xlim([3,9.5])
plt.ylim([0,250])
ax1 = plt.gca()
ax1.set_xlabel('Max obs. MMI',fontsize=14,labelpad=0)
ax1.set_ylabel('Time when MMI$\geq$3 (s)',fontsize=14,labelpad=0)
ax1.tick_params(pad=1.5,length=0,size=0,labelsize=12)
ax1.set_title('Observation',fontsize=14)
plt.subplot(1,2,2)
plt.plot(all_sav_maxMMI_obs,all_sav_Time_model[3]['time'],'r.')
plt.ylim([0,250])
ax1 = plt.gca()
ax1.set_xlabel('Max obs. MMI',fontsize=14,labelpad=0)
ax1.tick_params(pad=1.5,length=0,size=0,labelsize=12)
ax1.set_title('M-LARGE',fontsize=14)


fig = plt.figure()
plt.subplot(1,2,1)
plt.plot(all_sav_maxMMI_obs,all_sav_Time_obs[5]['time'],'k.')
plt.xlim([3,9.5])
plt.ylim([0,250])
ax1 = plt.gca()
ax1.set_xlabel('Max obs. MMI',fontsize=14,labelpad=0)
ax1.set_ylabel('Time when MMI$\geq$5 (s)',fontsize=14,labelpad=0)
ax1.tick_params(pad=1.5,length=0,size=0,labelsize=12)
ax1.set_title('Observation',fontsize=14)
plt.subplot(1,2,2)
plt.plot(all_sav_maxMMI_obs,all_sav_Time_model[5]['time'],'r.')
plt.xlim([3,9.5])
plt.ylim([0,250])
ax1 = plt.gca()
ax1.set_xlabel('Max obs. MMI',fontsize=14,labelpad=0)
ax1.tick_params(pad=1.5,length=0,size=0,labelsize=12)
ax1.set_title('M-LARGE',fontsize=14)
