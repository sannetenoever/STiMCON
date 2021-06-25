"""
Created on Fri May  1 10:36:03 2020

@author: sanoev

Shows how acoustic time and stimulus time is not the same in STiMCON (Figure 6)

"""
import math
import numpy as np
import STiMCON_core
import STiMCON_sen
import matplotlib.pyplot as plt
import ColorScheme as cs
cmaps = cs.CCcolormap()
bfi = cs.baseFigInfo()   
            
#%% basic information
fs = 1000;
Freq = 4.0;

#%% create the language model constrainst:
# model is: I do/I eat/I go
LMnames = np.array(['a','b','c','d','e'])
feedbackmat = np.zeros([5,5])
feedbackmat[0] = [0, 0.0, 0.8, 0.8, 0.0]
feedbackmat[1] = [0.0, 0, 0, 0,0]
feedbackmat[2] = [0.8, 0, 0, 0,0]
feedbackmat[3] = [0.0, 0, 0, 0,0]
feedbackmat[4] = [0.8, 0, 0, 0,0]

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

#%% things for fft calculations
def calcFFT(tofft, fs = 1000, pad = 0):
    tofft = (tofft-np.mean(tofft))
    pad = int(0*fs)
    tofft = np.concatenate((np.zeros(pad),tofft,np.zeros(pad)),axis=0)
    N = tofft.shape[-1]
    Han = np.hanning(N)
    tofft = tofft*Han
    freq = np.fft.fftfreq(N, 1/fs)
    FourSen = np.fft.fft(tofft)    
    return FourSen, freq

#%% create itterations of sensory input at different temporal offsets
import numpy.matlib
timeoff = np.linspace(-0.1,0.1,19)
thres = np.linspace(0.2,1.5,15)
FI =  np.arange(0,100)
stimnames = ['low low','high high','low_high','high_low']
SEN = np.zeros([len(stimnames),len(timeoff),len(thres),len(FI)])
ACT = np.zeros([len(stimnames),len(timeoff),len(thres),len(FI)])
SUPRA = np.zeros([len(stimnames),len(timeoff),len(thres),len(FI)])
FIRSTSUPRA = np.zeros([len(stimnames),len(timeoff),len(thres),len(FI)])
SenType = ''

for stim in range(4):    
    if stim == 0: # high-high
        v = 1
    elif stim == 1: # low-low
        v = 2
    elif stim == 2: # high low
        v= 3  
    elif stim == 3: # low high
        v= 4     
    Ust = np.array([0,v])
    st = numpy.matlib.repmat(Ust,1,5)[0]
    for th in range(len(thres)):
        for it in range(len(timeoff)):
            stimtime= np.linspace(0,(len(st)-1)/Freq,len(st))*fs
            j = np.arange(1,len(stimtime),2)
            stimtime[j] = stimtime[j] + timeoff[it]*fs
           
            if SenType == 'linear':
                seninput = {'stim_ord': st,
                  'intensity': np.zeros(len(st))+thres[th],
                  'stim_time': stimtime,
                  'tot_length': (len(st)+1)/Freq*fs}                    
                sensory_input = senObj.create_stim_vartimethres(seninput) 
            else:            
                seninput = {'stim_ord': st,
                  'intensity': np.zeros(len(st))+thres[th],
                  'stim_time': stimtime,
                  'tot_length': (len(st)+1)/Freq*fs} 
                sensory_input = senObj.create_stim_vartimethresG(seninput,GausW=3)   
                SenType = ''
            STiMCON_var = STiMCON_core.modelPara(parameters)
            out = STiMCON_var.runsingle(sensory_input)  
            # ignore the first and last big (only sustained bit..)
            I = np.arange(500,len(sensory_input[0])-500)
            F,freq = calcFFT(np.mean(sensory_input[:,I],0))
            Fact,freq = calcFFT(np.mean(out['activation'][Ust,:][:,I],0)) # overall activation
            FsupraSen,freq = calcFFT(np.mean(out['spiketimes'][:,I]==2,0)) # overall activation
            idx= np.abs(freq-Freq).argmin()
            
            SEN[stim,it,th,:] = abs(F[FI])
            ACT[stim,it,th,:] = abs(Fact[FI])
            SUPRA[stim,it,th,:]= abs(FsupraSen[FI])
            curnodac = 10
            tempFA = np.zeros(out['spiketimes'].shape)
            for T in range(out['spiketimes'].shape[1]):
                if sum(out['spiketimes'][:,T]==2) > 0:
                    idx2 = np.argmax(out['spiketimes'][:,T]==2)
                    if idx2 != curnodac:
                        tempFA[idx2,T] = 1
                        curnodac = idx2                    
            FIRSTSUPRA[stim,it,th,:] = abs(calcFFT(np.mean(tempFA[:,I],0))[0])[FI]

   
#%% now plot
plot = ''
import copy
axs = list()
fig = plt.figure(constrained_layout=True, figsize = (bfi.figsize.Col2/3*2,6))
gs = fig.add_gridspec(20, 5)

ths=np.arange(0,15)
for stim in range(4):
    D = SUPRA[stim,:,:,idx]
    if plot == 'SEN':
        D = copy.deepcopy(SEN[stim,:,:,idx]).transpose()**2
        D = D/1000
    else:
        D = copy.deepcopy(ACT[stim,:,:,idx]).transpose()**2
        D = D/100000
    rowmean = D.mean(axis=0)
    rowstd = D.std(axis=0)
   
    axs.append(fig.add_subplot(gs[stim*5:stim*5+5,0:3]))    
    pos = axs[stim].imshow(D, aspect = 'auto', origin = 'lower',cmap = 'OrRd')
    axs[stim].set_title(stimnames[stim])
    x = np.linspace(timeoff[0]*1000,timeoff[-1]*1000,5).astype(int)
    axs[stim].set_xticks(np.linspace(0,len(timeoff),len(x)))            
    y = np.around(np.linspace(thres[0],thres[-1],5),1)
    axs[stim].set_yticks(np.linspace(0,len(thres),len(y)))
    axs[stim].set_yticklabels(y)    
    cbar = fig.colorbar(pos, ax=axs[stim])
    cbar.set_label('power')
    if stim == 3:
        axs[stim].set_xlabel('odd word offset (ms)')
        axs[stim].set_xticklabels(x)          
    else:
        axs[stim].tick_params(axis='x',  which='both', bottom=True, labelbottom=False)
    # power
    axs[stim].set_ylabel('stimulus intensity')
    if plot != 'SEN':
        cbar.mappable.set_clim(2,4.8)

#% slice at 0.8 and 1.0
ths = [9, 6]
for it in range(len(ths)):
    th = len(thres)-ths[it]
    axs.append(fig.add_subplot(gs[it*4:it*4+4,3:6]))       
    for stim in range(4):
        if plot == 'SEN':
            D = copy.deepcopy(SEN[stim,:,th,idx])**(2)
        else:
            D = copy.deepcopy(ACT[stim,:,th,idx])**(2)
        tit = 'power at intensity ' + '{:.1f}'.format(thres[th])
        axs[-1].plot(timeoff*1000, D)      
    axs[-1].set_title(tit)
    if plot != 'SEN':
        axs[-1].set_ylabel('power ($x10^{5}$)')
        axs[-1].set_xticks(np.arange(-100,150,50))    
        axs[-1].set_yticks([3e5,4e5])
        axs[-1].set_yticklabels([3,4])
    else:
        axs[-1].set_ylabel('power ($x10^{3}$)')
        axs[-1].set_yticks([1e3,2e3])
        axs[-1].set_yticklabels([1,2])
    if it == 0:
        axs[-1].tick_params(axis='x', which='both', bottom=True, labelbottom=False)
    else:
        axs[-1].set_xlabel('odd word offset (ms)')

to = [4,9,14]
th = len(thres)-ths[1]
for it in range(len(to)):
    axs.append(fig.add_subplot(gs[(it+3)*3+2:(it+3)*3+3+2,3:6]))    
    for stim in range(4):
        if plot == 'SEN':
            D = copy.deepcopy(SEN[stim,to[it],th,:])**(20)
        else:
            D = copy.deepcopy(ACT[stim,to[it],th,:])**(20)       
        D = D/1e56
        axs[-1].plot(freq[FI[1:-80]], D[1:-80])                 
    axs[-1].set_ylabel('$ampl^{10}(x10^{56})$')
    axs[-1].set_title('odd word offset ' '{:.1f}'.format(timeoff[to[it]]*1000))
    axs[-1].set_xticks(np.arange(2,10,2))
    if it < len(to)-1:
        axs[-1].tick_params(axis='x', which='both', bottom=True, labelbottom=False)
    else:
        axs[-1].set_xlabel('frequency (Hz)')         
plt.show()
#fig.savefig(figloc + 'PredRhythm' + SenType + plot + '.pdf', format='pdf')
            
