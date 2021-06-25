
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:36:03 2020

@author: sanoev

Shows the basic behavior of STiMCON (Figure 4), the thresholds/timing of activation (Figure 5) and ambiguous daga overall simulations (Figure 8A)

"""
import math
import numpy as np
import STiMCON_core
import STiMCON_plot
import STiMCON_sen
import matplotlib.pyplot as plt
import ColorScheme as cs
cmaps = cs.CCcolormap()
bfi = cs.baseFigInfo()        
                                        
#%% basic information
fs = 1000;
Freq = 4.0;

#%% create the language model constrainst:
# model is: I eat very nice cake
LMnames = np.array(['I','eat','very','nice','cake'])
feedbackmat = np.zeros([5,5])
feedbackmat[0] = [0, 1, 0, 0, 0]
feedbackmat[1] = [0, 0, 0.2, 0.3, 0.5]
feedbackmat[2] = [0, 0, 0, 1, 0]
feedbackmat[3] = [0, 0, 0, 0, 1]
feedbackmat[4] = [0, 0, 0, 0, 0]

Nnodes = len(feedbackmat)

#%% define the parameters of the model
parameters = {"Nnodes": Nnodes,
       "OsFreq": Freq,
       "OsAmp": 1,
       "OsOffset": 0.25*math.pi,
       "activation_threshold": 1,
       "feedbackmat": feedbackmat,
       "feedbackinf": 1.5,
       "feedbackdecay": 0.01,
       "feedbackdelay": int(0.9/Freq*fs),
       "latinhibstrength": 0,
       "selfexitation": 0,
       "Inhib": -0.2,
       "fs": fs,
       'LMnames': LMnames}

#%% define the parameters for the sensory input
stimpara = {'word_duration': int(0.5/Freq*fs),
            'onsetdelay': int(0.5/Freq*fs),
            'Nnodes': Nnodes}
#% adjust OsOffset based on the onsetdelay and word_duration:
peak = (stimpara['word_duration']+stimpara['onsetdelay'])/fs
parameters['OsOffset'] = peak*Freq*(2*math.pi)

## set all the parameters
senObj = STiMCON_sen.modelSen(stimpara,parameters)

#%%########################################
### MODEL DEMONSTARTIONS - Figure 4 #######
##########################################

# create the sensory input
seninputA = list()
filenames = list()

#%% I eat
seninput = {'stim_ord': np.array([0,1]),
            'stim_time': np.linspace(0,1/Freq,2)*fs,
            'tot_length': 5/Freq*fs}
sensory_input = senObj.create_stim(seninput)
filename = 'STiMCON_5node_Ieat'
filenames.append(filename);seninputA.append(sensory_input)

#%% I eat very nice cake
seninput = {'stim_ord': np.array([0,1,2,3,4]),
          'stim_time': np.linspace(0,4/Freq,5)*fs,
          'tot_length': 5/Freq*fs}     
sensory_input = senObj.create_stim(seninput)   
filename = 'STiMCON_5node_Ieatverynicecake' 
filenames.append(filename);seninputA.append(sensory_input)

#%% I eat cake
seninput = {'stim_ord': np.array([0,1,4]),
            'stim_time': np.linspace(0,2/Freq,3)*fs,
            'tot_length': 5/Freq*fs}
sensory_input = senObj.create_stim(seninput)
filename = 'STiMCON_5node_Ieatcake' 
filenames.append(filename);seninputA.append(sensory_input)

#%% I eat cake
seninput = {'stim_ord': np.array([0,1,4]),
            'stim_time': np.linspace(0,2/Freq,3)*fs,
            'tot_length': 5/Freq*fs}
seninput['stim_time'][-1] = seninput['stim_time'][-1]-0.05*fs
sensory_input = senObj.create_stim(seninput)
filename = 'STiMCON_5node_Ieatcake_offset' 
filenames.append(filename);seninputA.append(sensory_input)

#%% Figure 4 
for it in range(len(filenames)):
    STiMCON_var = STiMCON_core.modelPara(parameters)
    out = STiMCON_var.runsingle(seninputA[it])
    
    # plot the model
    plObj = STiMCON_plot.PlotObj(out,parameters)
    figTC = plObj.timecourse(seninputA[it], fsize = [bfi.figsize.Col2/3,4])
    figAF = plObj.axfig(seninputA[it], fsize = [bfi.figsize.Col2/3,4])
    
    #figTC.savefig(figloc + filenames[it] + '_TC.pdf', format='pdf')
    #figAF.savefig(figloc + filenames[it] + '_AX.pdf', format='pdf')
    
#%%
#######################################
# THRESHOLD DETERMINATION - Figure 5###
#######################################

# iterate through thresholds and timing delays for a input and see effects
threshold = np.linspace(0,1.0,20)
delays = np.linspace(-0.5,0.5,20)/Freq*fs
fininput = np.array([0,2,3,4])

AllFirstSpTime = np.empty([len(threshold),len(delays),len(fininput)])
AllFirstSpTime_relStimOnset = np.empty([len(threshold),len(delays),len(fininput)])
for cntThr in range(len(threshold)):
    for cntDel in range(len(delays)):
        for cntFi in range(len(fininput)):
            senoi = fininput[cntFi]
            lat = np.linspace(0,2/Freq,3)*fs
            lat[-1] = lat[-1]+delays[cntDel]            
            seninput = {'stim_ord': np.array([0,1,senoi]),
                        'intensity': np.array([1,1,threshold[cntThr]]),
                        'stim_time': lat,            
                        'tot_length': 5/Freq*fs}
            sensory_input = senObj.create_stim_vartimethres(seninput)        
            STiMCON_var = STiMCON_core.modelPara(parameters)            
            out = STiMCON_var.runsingle(sensory_input)
            
            onsetLast = int(lat[-1:]+stimpara['onsetdelay'])
            inxpl = np.where(out['spiketimes'][senoi][onsetLast:]==2)
            if len(inxpl[0])>0:
                AllFirstSpTime[cntThr,cntDel,cntFi] = inxpl[0][0]+onsetLast - len(lat)/Freq*fs # time relative to isochrenous (i.e. phase)
                AllFirstSpTime_relStimOnset[cntThr,cntDel,cntFi] = inxpl[0][0]
            else:
                AllFirstSpTime[cntThr,cntDel,cntFi] = np.nan
                AllFirstSpTime_relStimOnset[cntThr,cntDel,cntFi] = np.nan
            
#%% plot first activation, Figure 5B
axs = list()
fig = plt.figure(constrained_layout=True, figsize = (bfi.figsize.Col2/3,6))
gs = fig.add_gridspec(20, 1)

for cntFi in range(len(fininput)):
    topl = AllFirstSpTime_relStimOnset[:,:,cntFi]
    axs.append(fig.add_subplot(gs[cntFi*4:cntFi*4+4,0])) 
    pos = axs[cntFi].imshow(topl, aspect='auto', interpolation='none', origin = 'lower',extent=[delays[0],delays[-1],threshold[0],threshold[-1]], cmap = cmaps.cmap3)
    pos.set_clim(25, 120)
    axs[cntFi].plot([-50,-50],[0,1],'k:')
    cbar = fig.colorbar(pos, ax=axs[cntFi])
    cbar.set_label('supra time')
    axs[cntFi].set_title(LMnames[fininput[cntFi]])   
    axs[cntFi].set_ylabel('stimulus intensity')   
    if cntFi == len(fininput)-1:
        axs[cntFi].set_xlabel('stim time relative\nto isochronous')
    else:
        axs[cntFi].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

thr = 10
print(threshold[thr])

axs.append(fig.add_subplot(gs[4*4:4*4+4,0])) 
axs[-1].plot(delays, AllFirstSpTime[thr,:,:])
axs[-1].set_xlabel('stim time relative\nto isochronous')
axs[-1].set_ylabel('supra time relative\nto isochronous')
axs[-1].legend(LMnames[fininput])
plt.tight_layout()
plt.show()

#fig.savefig(figloc + 'ProcEff.pdf', format='pdf')
   
#%%
#######################################
###ABMIGUOUS STIMULATION - Figure 8A###
#######################################
    
#%% show ambiguous stimulus    
feedbackmat = np.zeros([5,5])
feedbackmat[0] = [0, 1, 0, 0, 0]
feedbackmat[1] = [0, 0, 0.2, 0.1, 0.0]
feedbackmat[2] = [0, 0, 0, 1, 0]
feedbackmat[3] = [0, 0, 0, 0, 0]
feedbackmat[4] = [0, 0, 0, 0, 0]
LMnames = np.array(['I','eat','da','ga','cake'])
parameters['feedbackmat'] = feedbackmat
parameters['LMnames'] = LMnames

#%% iterate through thresholds and timing delays for a input
prop = np.linspace(0,1,12)
delays = np.linspace(-0.5,0.5,20)/Freq*fs

AllFirstSpTime = np.zeros([len(prop),len(delays),Nnodes])
AllFirstSpTime_relStimOnset = np.zeros([len(prop),len(delays),Nnodes])
FirstActive = np.zeros([len(prop),len(delays)])
intensity = np.zeros([Nnodes,3]) 
intensity[0,0] = 1; intensity[1,1] = 1
for cntProp in range(len(prop)):
    for cntDel in range(len(delays)):
            lat = np.linspace(0,2/Freq,3)*fs
            lat[-1] = lat[-1]+delays[cntDel]            
            intensity[2,-1] = 1*prop[cntProp] 
            intensity[3,-1] = 1-intensity[2,-1]
            seninput = {'stim_ord': list(),
                        'intensity': intensity,
                        'stim_time': lat,            
                        'tot_length': 5/Freq*fs}
            sensory_input = senObj.create_stim_vartimethres(seninput)        
            STiMCON_var = STiMCON_core.modelPara(parameters)            
            out = STiMCON_var.runsingle(sensory_input)
            
            onsetLast = int(lat[-1:]+stimpara['onsetdelay'])
            for senoi in range(Nnodes):
                inxpl = np.where(out['spiketimes'][senoi][onsetLast:]==2)
                if len(inxpl[0])>0:
                    AllFirstSpTime[cntProp,cntDel,senoi] = inxpl[0][0]+onsetLast - len(lat)/Freq*fs # time relative to isochrenous (i.e. phase)
                    AllFirstSpTime_relStimOnset[cntProp,cntDel,senoi] = inxpl[0][0]
                else:
                    AllFirstSpTime[cntProp,cntDel,senoi] = np.nan
                    AllFirstSpTime_relStimOnset[cntProp,cntDel,senoi] = np.nan
            try:
                FirstActive[cntProp,cntDel] = np.nanargmin(AllFirstSpTime_relStimOnset[cntProp,cntDel,:])
                val = AllFirstSpTime_relStimOnset[cntProp,cntDel,int(FirstActive[cntProp,cntDel])]
                checkdouble = np.argwhere(AllFirstSpTime_relStimOnset[cntProp,cntDel,:] == val)
                if len(checkdouble)>1:
                    aclev = np.zeros(len(checkdouble))
                    for it in range(len(checkdouble)):                        
                        inx = AllFirstSpTime_relStimOnset[cntProp,cntDel,checkdouble[it]]
                        aclev[it] = out['activation'][checkdouble[it],onsetLast+int(inx)]
                    FirstActive[cntProp,cntDel] = checkdouble[np.argmax(aclev)]
                    if aclev[0]==aclev[1]:
                        FirstActive[cntProp,cntDel]=np.nan 
            except:
                FirstActive[cntProp,cntDel] = np.nan
                
#%% plot
from matplotlib.colors import LinearSegmentedColormap
CH = plt.cm.get_cmap('cubehelix', 3)
cosCB = CH(np.linspace(0, 1, 3))
cosCB[0] = [0.8, 0, 0.8, 1]
cosCB[0] = [0.8,0.8,0.8,1]
cosCB[1] = [0.4, 0.2, 0, 1] 
cosCB[1] = [0.2,0.2,0.2,1]
CH = LinearSegmentedColormap.from_list('mycmap', cosCB,3)

x = FirstActive-2 # adjust to only plot
x = np.nan_to_num(x,nan=3.0,copy=True)
FirstActiveIt = x.astype(int)
axs = list()
fig = plt.figure(constrained_layout=True, figsize = (bfi.figsize.Col1,2.2))
gs = fig.add_gridspec(3, 9)
axs.append(fig.add_subplot(gs[0:3,0:5]))
axs[0].imshow(FirstActiveIt)
pos = axs[0].imshow(FirstActiveIt, aspect='auto', interpolation='none', origin='lower',
         cmap=CH)
x = np.linspace(delays[0],delays[-1],5).astype(int)
axs[0].set_xticks(np.linspace(0,len(delays),len(x)))
axs[0].set_xticklabels(x)
y = np.around(np.linspace(prop[0],prop[-1],6),1)
axs[0].set_yticks(np.linspace(0,len(prop),len(y)))
axs[0].set_yticklabels(y)
axs[0].set_title('First node active', fontsize = bfi.fontsize.axes)
axs[0].set_xlabel('stim time relative')
axs[0].set_ylabel('proportion /da/ stimulus') 
  
proptopl = [8,4,1]
Npl = [2,3]
for p,ppl in enumerate(proptopl): 
    axs.append(fig.add_subplot(gs[p,5:]))   
    topl = AllFirstSpTime_relStimOnset[ppl,:,Npl]
    pos=axs[p+1].imshow(topl,aspect='auto',interpolation='none',origin='lower',cmap = cmaps.cmap3)
    pos.set_clim(25, 120)
    x = np.linspace(delays[0],delays[-1],5).astype(int)   
    axs[p+1].set_xticks(np.linspace(0,len(delays),len(x)))        
    axs[p+1].set_yticks(np.arange(-1,3))
    axs[p+1].set_yticklabels(['',LMnames[2],LMnames[3],''])
    cbar = fig.colorbar(pos, ax=axs[p+1])
    cbar.set_label('supra time')
    axs[p+1].set_title('Prop: '+ '{0:.2f}'.format(prop[ppl]), fontsize = bfi.fontsize.axes)
    if p == 2:
        axs[p+1].set_xticklabels(x)
        axs[p+1].set_xlabel('stim time relative')   
    else:
        axs[p+1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    
    cntam = -1
    Rcurmin = 100
    contourvals = list()
    cnode = list()
    for it in range(len(topl[0])):
        curmin = FirstActive[ppl,it]
        if np.isnan(curmin):
            curmin = 10
        if curmin == Rcurmin:
            contourvals[cntam].append(it)
        else:
            cntam = cntam+1
            contourvals.append([it])
            cnode.append(curmin)                
        Rcurmin = curmin            
    for cntam in range(len(cnode)):
        if (cnode[cntam] < 10):
            if cnode[cntam] == 2.0:
                col = cosCB[0]
            else:
                col = cosCB[1]               
            axs[p+1].plot([contourvals[cntam][0]-0.5, contourvals[cntam][0]-0.5],
               [cnode[cntam]-0.5-2, cnode[cntam]+0.5-2],color=col)
            axs[p+1].plot( [contourvals[cntam][-1]+0.5, contourvals[cntam][-1]+0.5],
               [cnode[cntam]-0.5-2, cnode[cntam]+0.5-2],color=col)
            axs[p+1].plot( [contourvals[cntam][0]-0.5, contourvals[cntam][-1]+0.5],
               [cnode[cntam]-0.5-2, cnode[cntam]-0.5-2],color=col)
            axs[p+1].plot( [contourvals[cntam][0]-0.5, contourvals[cntam][-1]+0.5],
               [cnode[cntam]+.5-2, cnode[cntam]+0.5-2],color=col)          
plt.show()
#fig.savefig(figloc + 'Ambiguous.pdf', format='pdf')



