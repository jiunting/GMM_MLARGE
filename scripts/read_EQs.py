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
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mlarge.scaling import make_linear_scale


def D2PGD(E,N,Z):
    '''
        or PGV or PGA
    '''
    D=(E**2.0+N**2.0+Z**2.0)**0.5
    PGD=np.maximum.accumulate(D)
    return PGD

def get_tcs(EQtime,T,E_file,N_file,Z_file):
    #start calculate PGA from 3-comps, dealing with time
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
    D_Z = obspy.read(Z_file)
    DD_Z = D_Z.copy()
    DD_Z.detrend('linear')
    DD_Z.trim(starttime=EQtime,endtime=EQtime+T[-1]+10,pad=True,fill_value=0)
    #make the data=0 at 0 second
    DD_E[0].data = DD_E[0].data-DD_E[0].data[0]
    DD_N[0].data = DD_N[0].data-DD_N[0].data[0]
    DD_Z[0].data = DD_Z[0].data-DD_Z[0].data[0]
    Dsum = (DD_E[0].data**2+DD_N[0].data**2+DD_Z[0].data**2)**0.5
    PGD = D2PGD(DD_E[0].data,DD_N[0].data,DD_Z[0].data)
    assert np.sum(DD_E[0].times() == DD_N[0].times())==len(DD_E[0].times()), "E-N are different"
    assert np.sum(DD_E[0].times() == DD_Z[0].times())==len(DD_Z[0].times()), "E-Z are different"
    t = DD_E[0].times()
    interp_data_E = np.interp(T,t,DD_E[0].data)
    interp_data_N = np.interp(T,t,DD_N[0].data)
    interp_data_Z = np.interp(T,t,DD_Z[0].data)
    interp_data_D = np.interp(T,t,Dsum)
    interp_data_PGD = np.interp(T,t,PGD)
    return interp_data_E,interp_data_N,interp_data_Z,interp_data_D,interp_data_PGD


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




# the base dir
eq_dir = "../data/testsuite_processed"

# list all the EQ folders under the eq_dir, same order as the MLARGE output
EQs = ["Illapel2015","Iquique2014","Maule2010","Melinka2016","Iquique_aftershock2014"]
EQt=[obspy.UTCDateTime(2015,9,16,22,54,33),obspy.UTCDateTime(2014,4,1,23,46,47),obspy.UTCDateTime(2010,2,27,6,34,14),obspy.UTCDateTime(2016,12,25,14,22,26),obspy.UTCDateTime(2014,4,3,2,43,13)]
#EQMw=[8.3,8.2,8.8,7.6,7.7]
#hypos=[(-71.674,-31.573,22.4),(-70.769,-19.61,25.0),(-72.898,-36.122,22.9),(-73.941,-43.406,38.0),(-70.493,-20.571,22.4)]
T = np.arange(102)*5+5 # sampled data in MLARGE and accel tcs

# MMI threshold that we're interested
MMI_thres = [3,4,5]

# loop for each earthquake
for ieq in [0,1,3]:
    # reset for a new event
    sav_warningTime = {} #warning time for different MMI threshold
    # get station info
    sta_info = glob.glob(eq_dir+"/"+EQs[ieq]+"/*_sm.chan")[0]
    sta_info = pd.read_csv(sta_info,header=None,sep='\s+',names=['net','sta','loc','chan','lat','lon','elev','samplerate','gain','units'],skiprows=1)
    # find unique staion, lon, and lat from sta_info
    tmp_uniq_sta, tmp_uniq_lon, tmp_uniq_lat = [],[],[]
    for ii in range(len(sta_info)):
        if sta_info['sta'][ii] not in tmp_uniq_sta:
            tmp_uniq_sta.append(sta_info['sta'][ii])
            tmp_uniq_lon.append(sta_info['lon'][ii])
            tmp_uniq_lat.append(sta_info['lat'][ii])
    # MLARGE shake prediction
    memo_sta_idx = {} # to memorize the location (index) of each station in the .shake grid
    MLARGE_predGM = glob.glob('/Users/timlin/TEST_MLARGE/GMTinp_Test031_GM_Pred/Test031.case000000%1d.epo*.shake'%(ieq))
    MLARGE_predGM.sort()
    # ===========for each time epoch===========
    # -part #1. read shake prediction by MLARGE and get all the predictions from file
    sav_MMI_model = {} #each station has a time series in list/array
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
    # -part #2. deal with accel time series data
    sav_MMI_obs = {} #each station has a time series in list/array
    Zs = glob.glob(eq_dir+"/"+EQs[ieq]+"/accel/*.*Z.mseed")
    Zs.sort()
    # for this event, read all the data
    for i in Zs:
        E_file = i.replace('Z','E')
        N_file = i.replace('Z','N')
        Z_file = i
        sta = Z_file.split('/')[-1].split('.')[0]
        # get station info by station name
        idx = sta_info[sta_info['sta']==sta].index
        assert len(idx)==3, "not 3 components!"
        assert sta_info['gain'][idx[0]]==sta_info['gain'][idx[1]]==sta_info['gain'][idx[2]], "different gain"
        # get data
        E, N, Z, D, PGA = get_tcs(EQt[ieq],T,E_file,N_file,Z_file)
        # data divided by gain and convert MMI
        gain = sta_info['gain'][idx[0]]
        E, N, Z, D, PGA = E/gain, N/gain, Z/gain, D/gain, PGA/gain #[cm/s2]
        MMI_obs = WGRW12(PGA,0)
        sav_MMI_obs[sta] = MMI_obs
        #plt.plot(T,sav_MMI_model[sta],'r')
        #plt.plot(T,MMI_obs,'k')
        #plt.show()
    # -part 3. deal with MMI threshold and timing
    for thres in MMI_thres:
        sav_maxMMI_obs = []
        sav_maxMMI_model = []
        for i_sac,sta in enumerate(sav_MMI_obs.keys()):
            sav_maxMMI_obs.append(max(sav_MMI_obs[sta]))
            sav_maxMMI_model.append(max(sav_MMI_model[sta]))
            idx_t_obs = np.where(sav_MMI_obs[sta]>=thres)[0]
            idx_t_model = np.where(sav_MMI_model[sta]>=thres)[0]
            # only calculate when TP
            if len(idx_t_obs)!=0 and len(idx_t_model)!=0:
                idx_t_obs = idx_t_obs[0]
                idx_t_model = idx_t_model[0]
                warningTime = (idx_t_obs-idx_t_model)*5 # sampling interval is 5 s
                if thres in sav_warningTime:
                    sav_warningTime[thres]['time'].append(warningTime)
                    sav_warningTime[thres]['idx'].append(i_sac)
                else:
                    sav_warningTime[thres] = {'time':[warningTime],'idx':[i_sac]}
    sav_maxMMI_obs = np.array(sav_maxMMI_obs)
    sav_maxMMI_model = np.array(sav_maxMMI_model)
    #=====make MMI threshold plot=====
    fig = plt.figure(figsize=(8.5,4))
    props = dict(boxstyle='round', facecolor='white', alpha=0.1)
    vmin, vmax = 0,120
    cmap = 'rainbow_r'
    for i,thres in enumerate(MMI_thres):
        plt.subplot(1,3,i+1)
        idx_TP = np.array(sav_warningTime[thres]['idx'])
        idx_XTP = [ii for ii in range(len(sav_maxMMI_obs)) if ii not in idx_TP]
        # plot not TP as dots
        plt.plot(sav_maxMMI_obs[idx_XTP],sav_maxMMI_model[idx_XTP],'ko')
        # if TP, show also the warning time
        tmp_warningTime = np.array(sav_warningTime[thres]['time'])
        plt.scatter(sav_maxMMI_obs[idx_TP],sav_maxMMI_model[idx_TP],c=tmp_warningTime,linewidth=1,edgecolor='k',cmap=cmap,vmin=vmin,vmax=vmax)
        plt.plot([thres,thres],[0,10],'r--')
        plt.plot([0,10],[thres,thres],'r--')
        plt.xlabel('Obs. MMI',fontsize=14,labelpad=0.5)
        ax1=plt.gca()
        ax1.tick_params(pad=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if i in [0]:
            plt.ylabel('Pred. MMI',fontsize=14,labelpad=0.5)
        else:
            ax1.tick_params(labelleft=False)
        plt.xlim([1,9.5])
        plt.ylim([1,9.5])
        plt.text(1.5,9.0,"FP",bbox=props)
        plt.text(9.0,9.0,"TP",bbox=props,ha='right')
        plt.text(1.5,1.5,"TN",bbox=props)
        plt.text(9.0,1.5,"FN",bbox=props,ha='right')
        plt.title('MMI Threshold = %d'%(thres))
        plt.subplots_adjust(left=0.05,top=0.88,right=0.92,bottom=0.12,wspace=0.05)
    #add colormap
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cmap.set_array([])
    cbaxes = fig.add_axes([0.93, 0.12, 0.02,0.76])
    clb = plt.colorbar(cmap,cax=cbaxes, orientation='vertical')
    clb.set_label('warning time (s)',fontsize=12,labelpad=0,size=14)
    ax1 = plt.gca()
    ax1.tick_params(pad=1)
    #plt.savefig('./GM_Pred_Test030/WarningTime.png',dpi=300)
    #plt.savefig('./GM_Pred_Test031/WarningTime.png',dpi=300)
    plt.show()











