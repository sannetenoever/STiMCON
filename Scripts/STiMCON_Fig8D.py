
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:36:03 2020

@author: sanoev

Fitting of the da/ga data using the relative node activation as output (Figure 8D)

"""
import math
import numpy as np
import STiMCON_core
import STiMCON_sen
import matplotlib.pyplot as plt
import ColorScheme as cs
from lmfit import Minimizer, Parameters
import pickle

cmaps = cs.CCcolormap()
bfi = cs.baseFigInfo()        
                                        
#%% load in the matlab data 
class fitdata(object):
    def __init__(self, rawdata):
        self.delv = rawdata[0][0][4][0]
        self.dimord = rawdata[0][0][0][0]
        self.ISIs = rawdata[0][0][1]
        self.exp = rawdata[0][0][2]
        self.freq = rawdata[0][0][3][0]
        self.dataF = rawdata[0][0][5]
        self.orirsq = rawdata[0][0][6]        
        
from scipy.io import loadmat
Rawfitdata = loadmat('./Data/DatafitdataAVG.mat')['fitdata']
Mfitd = fitdata(Rawfitdata)
        
#%% functions ##
def rsq_fun(fit, ydata):    
    rsq = 1-np.sum((fit-ydata)**2)/np.sum((ydata-np.mean(ydata))**2)           
    return rsq

def StimFit2(params,delays,data):
    fs = 1000
    Nnodes = 4 
    Freq = params['Freq']    
    prop = params['prop']
    fbdel = params['fbdel']
    fbdec = params['fbdec']
    toff = params['toff']
    inhib = params['inhib']
    osamp = params['osamp']
    fbmat = params['fbmat']   

    LMnames = np.array(['I','eat','da','ga'])
    feedbackmat = np.zeros([4,4])    
    if fbmat:
        feedbackmat[0] = [0, 1, 0, 0]
        feedbackmat[1] = [0,0,0.2,0.1]
        feedbackmat[2] = [0, 0, 0, 0]
        feedbackmat[3] = [0, 0, 0, 0]
    else:
        feedbackmat[0] = [0, 1, 0, 0]
        feedbackmat[1] = [0,0,0.15,0.15]
        feedbackmat[2] = [0, 0, 0, 0]
        feedbackmat[3] = [0, 0, 0, 0]
    
    parameters = {"Nnodes": Nnodes,
       "OsFreq": Freq,
       "OsAmp": osamp,
       "OsOffset": 0.25*math.pi,
       "activation_threshold": 1,
       "feedbackmat": feedbackmat,
       "feedbackinf": 1.5,
       "feedbackdecay": fbdec,
       "feedbackdelay": int(fbdel/Freq*fs),
       "latinhibstrength": 0,
       "selfexitation": 0,
       "Inhib": inhib,
       "fs": fs,
       'LMnames': LMnames}
    
    delays = delays+toff*fs     
    ndel = len(delays)
        
    MeanActivation = np.zeros([ndel])
    
    intensity = np.zeros([Nnodes,3]) 
    intensity[0,0] = 1
    intensity[1,1] = 1
    intensity[2,-1] = 1*prop
    intensity[3,-1] = 1-intensity[2,-1]       

    ## set all the parameters
    stimpara = {'word_duration': int(0.5/Freq*fs),
                    'onsetdelay': int(0.5/Freq*fs),
                    'Nnodes': Nnodes}   
    peak = (stimpara['word_duration']+stimpara['onsetdelay'])/fs
    parameters['OsOffset'] = peak*Freq*(2*math.pi)
    
    senObj = STiMCON_sen.modelSen(stimpara,parameters)        
    for cntDel in range(len(delays)):        
            lat = np.linspace(0,2/Freq,3)*fs
            lat[-1] = lat[-2]+delays[cntDel]              
            seninput = {'stim_ord': list(),
                        'intensity': intensity,
                        'stim_time': lat,            
                        'tot_length': 5/Freq*fs}
            sensory_input = senObj.create_stim_vartimethres(seninput)    
            sensory_input = np.concatenate((sensory_input, np.zeros((Nnodes,500))),axis=1)
            STiMCON_var = STiMCON_core.modelPara(parameters)            
            out = STiMCON_var.runsingle(sensory_input)
            
            I = np.arange(lat[-1],lat[-1]+500).astype(int)
            meanN1 = np.nanmean(out['activation'][2,I])
            meanN2 = np.nanmean(out['activation'][3,I])
            
            Fact = (meanN1-meanN2)/(meanN1+meanN2)
            if np.isnan(Fact):
                MeanActivation[cntDel]= 0.5
            else:
                MeanActivation[cntDel] = Fact          
    MeanActivation = MeanActivation-min(MeanActivation)
    if max(MeanActivation) > 0:
        MeanActivation = MeanActivation/max(MeanActivation)
    if params['fit']==True:
        return MeanActivation-data
    else:            
        return MeanActivation

#%%  
# create a set of Parameters
params = Parameters()
params.add('Freq', value=6.25, vary = False)
params.add('prop', value=0.2, min=0.1, max=0.8, brute_step=0.05, vary = True)
params.add('fbdel', value=0.22, min=0.1, max=1.0, brute_step=0.1, vary = True)
params.add('fbdec', value=0.022, min=0.0, max=0.1, brute_step=0.01, vary = True)
params.add('toff', value=-0.027, min=-0.05, max=0.05, brute_step=0.01, vary = True)
params.add('fit', value=False, vary = False)   
    
rrs = np.zeros(3)
modelname = ['regular','nofb2','noinhib','noos']
        
for fittype in range(4):    
    if fittype == 0: # normal
        params.add('inhib', value = -0.2, vary=False)
        params.add('osamp', value = 1, vary=False)
        params.add('fbmat', value = True, vary=False)
    elif fittype == 1: # no feedback
        params.add('inhib', value = -0.2, vary=False)
        params.add('osamp', value = 1, vary=False)
        params.add('fbmat', value = False, vary=False)
    elif fittype == 2: # no inhibition
        params.add('inhib', value = 0, vary=False)
        params.add('osamp', value = 1, vary=False)
        params.add('fbmat', value = True, vary=False)
    elif fittype == 3: # no oscillation
        params.add('inhib', value = -0.2, vary=False)
        params.add('osamp', value = 0, vary=False)
        params.add('fbmat', value = True, vary=False) 
        
    for fr in range(3):  
        #filename = saveloc + '/Brute_DaGa_' + modelname[fittype] + '_' + Mfitd.exp[fr][0][0]                  
        #if os.path.exists(filename)==False:
        params['Freq'].value = Mfitd.freq[fr]
        params['fit'].value = True;
        delays = Mfitd.delv[fr][0]*1000
        d = Mfitd.dataF[fr,3,:]
        d = (d-min(d))
        ydataF = d/max(d)               
        minner = Minimizer(StimFit2, params, fcn_args=(delays, ydataF), workers = -1)
        result = minner.minimize(method='brute') 
        #filename = saveloc + '/Brute_DaGa_' + modelname[fittype] + '_' + Mfitd.exp[fr][0][0]                  
        #with open(filename, 'wb') as f:
        #    pickle.dump([result, minner],f)

#%% load data and plot
foi = [0, 2]
fig, ax = plt.subplots(2,4, figsize = (bfi.figsize.Col2,3)) 
rsqs = np.zeros((4,len(foi)))
aics = np.zeros((4,len(foi)))

modelname = ['regular','noinhib','nofb2','noos']
for fittype in range(4):    
    for frit,fr in enumerate(foi): 
        filename = './Data/DaGaFitRelNodeActivation/DaGa_' + modelname[fittype] + '_' + Mfitd.exp[fr][0][0]                  
        with open(filename, 'rb') as f:
            x = pickle.load(f)
        result = x[0]               
        
        d = Mfitd.dataF[fr,3,:]
        d = (d-min(d))
        ydataF = d/max(d)
        
        delays = Mfitd.delv[fr][0]*1000
        params = result.params
        params['fit'].value = False
        fitTC = StimFit2(params,delays,ydataF) 
        rsq = 1-np.sum((fitTC-ydataF)**2)/np.sum((ydataF-np.mean(ydataF))**2) 
        rsqs[fittype,frit] = rsq
        aics[fittype,frit] = result.aic
        
        delays = delays/1000
        ax[frit,fittype].plot(delays, ydataF, '+', color = cs.cols[0])
        ax[frit,fittype].plot(delays,fitTC, color = cs.cols[1])        
        ax[frit,fittype].set_title(modelname[fittype] + ' %.2f' %Mfitd.freq[foi[frit]] + 'Hz')
plt.tight_layout()
plt.show()
#fig.savefig(figloc + 'DaGaFit_DifPar_timecourses.pdf', format='pdf')   

#%% plot rsq only
foi = [0, 2]
fig, ax = plt.subplots(1, figsize = (bfi.figsize.Col1/2,2)) 
plt.bar(modelname, np.mean(rsqs,axis=1))
plt.ylabel('R-squared')
plt.ylim([0.7, 0.85])
plt.xticks(rotation=90)
plt.show()
#fig.savefig(figloc + 'DaGaFit_DifPar_rsq.pdf', format='pdf')   

#%% rsq and relevant time courses:
axs = list()
fig = plt.figure(constrained_layout=True, figsize = (bfi.figsize.Col2/2,3))
gs = fig.add_gridspec(4, 2)
axs.append(fig.add_subplot(gs[0:2,0]))
plt.xticks(rotation=90)
axs[-1].bar(modelname, np.mean(rsqs,axis=1))
axs[-1].set_ylabel('R$^2$')
axs[-1].set_ylim([0.7, 0.85])
fittype = 0
for frit,fr in enumerate(foi): 
    axs.append(fig.add_subplot(gs[frit,1]))
    filename = './Data/DaGaFitRelNodeActivation/DaGa_' + modelname[fittype] + '_' + Mfitd.exp[fr][0][0]                  
    with open(filename, 'rb') as f:
        x = pickle.load(f)
    result = x[0]
    d = Mfitd.dataF[fr,3,:]
    d = (d-min(d))
    ydataF = d/max(d)
    
    delays = Mfitd.delv[fr][0]*1000
    params = result.params
    params['fit'].value = False
    fitTC = StimFit2(params,delays,ydataF) 
    rsq = 1-np.sum((fitTC-ydataF)**2)/np.sum((ydataF-np.mean(ydataF))**2) 
    rsqs[fittype,frit] = rsq
    aics[fittype,frit] = result.aic
    
    delays = delays/1000
    axs[-1].plot(delays, ydataF, '+', color = cs.cols[0])
    axs[-1].plot(delays,fitTC, color = cs.cols[1])    
    axs[-1].set_title(modelname[fittype] + ' %.2f' %Mfitd.freq[foi[frit]] + 'Hz')
#fig.savefig(figloc + 'DaGaFit_DifPar_Comb.pdf', format='pdf')   