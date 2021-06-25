#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:26:42 2020

@author: sanoev

This script extracts and plot the basic tmeporal variation in the syllables and words of the CGN related to Figure 2 of the main manuscript.
"""
base = ''

#%% locs
ID = 'S05_00'
SEQUENCE_LEN = 10
MIN_WORD_FREQUENCY = 10
STEP = 1

#from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import ColorScheme as cs
cmaps = cs.CCcolormap()
bfi = cs.baseFigInfo()

#%% info
loadloc = './Data/'
    
filename = loadloc + '/WordLevel_TrainAndTestData'          
with open(filename, 'rb') as f:
    TrainAndTest = pickle.load(f)

# save this, so you can also run stuff just ont he validation set:
filename = loadloc + '/TrainAndTestSplits'            
with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
    TT = pickle.load(f)  
TT_sentences = TT[0]
TT_next_words = TT[1]
TT_index = TT[4][:TT[5]]
TT_sentences_test = TT[2]
TT_next_words_test= TT[3]
TT_index_test = TT[4][TT[5]:]
    
sentences = TrainAndTest[0]
next_words = TrainAndTest[1]
words = TrainAndTest[4]

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

#% get panda info (with ISI etc)
filename = loadloc + '/WordLevel_TrainAndTestData_PDinfo_light'            
with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
    PandaInfo = pickle.load(f)
    
#%% go through all and extract the duration 
WordInfo = pd.DataFrame({'word':str(),
                       'ID': str(),
                       'N': int(),
                       'mean': float(),
                       'std': float(),
                       'median': float(),
                       'IQR': float()}, index = [0])

for wrcnt, word in enumerate(words):
    Durs = []
    for filecnt, subPan in enumerate(PandaInfo[2]):
         Durs = Durs + list(subPan['End'][subPan['Word']==word] - subPan['Begin'][subPan['Word']==word])
    Dseries = [word, words[word][1], len(Durs), np.mean(Durs), np.std(Durs), np.median(Durs), scipy.stats.iqr(Durs)]
    WordInfo.loc[wrcnt] = Dseries

#% save the final wordlist panda info
#filename = loadloc + '/CorpusInfo'            
#with open(filename, 'wb') as f: 
#    pickle.dump([WordInfo],f)      
    
#%% go to celex to get some additional information
filename = loadloc + '/CorpusInfo'            
with open(filename, 'rb') as f: 
    WordInfo = pickle.load(f)[0]

# dump all words as text file so celex can extract information about syllable features
#fileO = open(loadloc +'/worddump.txt', 'w+')
#for word in words:
#    fileO.write(word + '\n')
#fileO.close()

# read the Celex Data
filename = loadloc +'/celexwords.txt'
Cel = pd.read_csv(filename, delimiter = '\\', engine = 'python')

CelinWI = pd.DataFrame({'Celexword':str(),
                           'WordSyl': str(),
                           'Nsyl': int(),
                           'Nsylphon': int(),
                           'PhonSylCLX': str(),
                           'PhonCLX': str(),
                           'Freq': float(),
                           'FreqLog': float(),
                           'length': int()}, index = [0])
# go through the wordInfo and add celex info (if excisting):
for it in range(len(WordInfo)):
    BI = Cel['Word'] == WordInfo['word'][it]
    if BI.any():
        CelinWI.loc[it] = Cel[BI].values.tolist()[0] + [len(WordInfo['word'][it])]
    else:
        CelinWI.loc[it] = ['', '', 0,0,'','', 0.0, 0.0 ,len(WordInfo['word'][it])]
WordInfo_c = pd.concat([WordInfo, CelinWI], axis=1, sort = False)

#% save the final wordlist panda info
#filename = loadloc + '/CorpusInfoAndCelex'            
#with open(filename, 'wb') as f:  
#    pickle.dump([WordInfo_c],f)   

###################################     
#%% plot some descriptives:
filename = loadloc + '/CorpusInfoAndCelex'            
with open(filename, 'rb') as f:  
    WordInfo_c = pickle.load(f)[0] 

#%% plot some examples of words from the corpus
subwords = ['van', 'zijn', 'snel','stem','hebben', 'eten', 'volgen', 'dragen']
freqs = np.zeros(len(subwords))
AD = list()
for wrcnt, word in enumerate(subwords):
    BI = WordInfo_c['word'] == word
    freqs[wrcnt] = WordInfo_c[BI].Freq.values[0]
    Durs = []
    for filecnt, subPan in enumerate(PandaInfo[2]):
         Durs = Durs + list(subPan['End'][subPan['Word']==word] - subPan['Begin'][subPan['Word']==word])
    AD.append(Durs)

#%% Figure 2A
axs = list()
fig = plt.figure(constrained_layout=True, figsize = (bfi.figsize.Col2,1.25)) # iin inches
gs = fig.add_gridspec(1, 8)
for wrcnt, word in enumerate(subwords):    
    axs.append(fig.add_subplot(gs[0,wrcnt]))   
    sns.distplot(AD[wrcnt],ax=axs[-1]).set_title('{} ({:.1f})'.format(word, freqs[wrcnt]))  
    axs[-1].set_xlim([0,1]); axs[-1].set_xticks([0,0.5,1])
    axs[-1].set_ylim([0, 10])
    plt.text(0.95, 0.9, '{:.3f}'.format(np.mean(AD[wrcnt]))[1:], transform=axs[-1].transAxes, \
             horizontalalignment='right',verticalalignment='top')
    if wrcnt == len(subwords)-1:
        axs[-1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        axs[-1].set_xlabel('duration (sec)')
    else:
        axs[-1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    axs[-1].tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
plt.show()
#fig.savefig(figloc + ID + '_ExampleTimeDistWords.pdf', format='pdf')
#fig.savefig(figloc + ID + '_ExampleTimeDistWords.jpg', format='jpg')

#%% plot overall mean + std (independent of word length)
axs = list()
fig = plt.figure(constrained_layout=True, figsize = (bfi.figsize.Col1,2))
gs = fig.add_gridspec(1, 2)
axs.append(fig.add_subplot(gs[0,0]))
BI = np.isnan(WordInfo_c['mean']) ==False
sns.distplot(WordInfo_c[BI]['mean'], ax=axs[-1]).set_title('Mean ({:.3f})'.format(np.mean(WordInfo_c[BI]['mean'])))
axs.append(fig.add_subplot(gs[0,1]))
sns.distplot(WordInfo_c[BI]['std'], ax=axs[-1]).set_title('Std ({:.3f})'.format(np.mean(WordInfo_c[BI]['std'])))
 
#fig.savefig(figloc + ID + '_MeanStd.pdf', format='pdf')
#fig.savefig(figloc + ID + '_MeanStd.jpg', format='jpg')   

#%% word length:
unwl = [2,3,4,5,6,7,8,9,10,11,12,13,14,15] # word lengths to plot
axs = list()
fig = plt.figure(constrained_layout=True, figsize = (bfi.figsize.Col2,7))
Ncol = 4
gs = fig.add_gridspec(5, Ncol)
pl_row = 0
pl_col = 0
Omeandur= np.zeros(len(unwl))
Ostddur=np.zeros(len(unwl))
for it,wl in enumerate(unwl):
    axs.append(fig.add_subplot(gs[pl_row,pl_col]))
    pl_col = pl_col+1
    if pl_col == Ncol:
        pl_col = 0
        pl_row = pl_row+1
        
    BI = WordInfo_c['length']==wl
    meanDur = WordInfo_c[BI]['mean'].values.tolist()
    meanDur = [x for x in meanDur if str(x)!='nan']
    Omeandur[it] = np.mean(meanDur)
    meanStd = WordInfo_c[BI]['std'].values.tolist()
    meanStd = [x for x in meanStd if str(x)!='nan']    
    Ostddur[it] = np.mean(meanStd)
    sns.distplot(meanDur,ax=axs[-1]).set_title('{}'.format(str(wl)))
    plt.text(0.95, 0.9, '{:.3f}'.format(Omeandur[it])[1:], transform=axs[-1].transAxes, \
             horizontalalignment='right',verticalalignment='top')
    axs[-1].set_xlim([0,1.5])
    if it == len(unwl)-1:
        axs[-1].set_xlabel('duration (sec)')
plt.show() 
#fig.savefig(figloc + ID + '_WordLength.pdf', format='pdf')
#fig.savefig(figloc + ID + '_WordLength.jpg', format='jpg')

#%% amount of syllables:
unsl = [1,2,3,4,5,6] # word lengths to plot
axs = list()
fig = plt.figure(constrained_layout=True, figsize = (bfi.figsize.Col2,7))
gs = fig.add_gridspec(4, Ncol)
pl_row = 0
pl_col = 0
OmeandurS = np.zeros(len(unsl))
OstddurS = np.zeros(len(unsl))
for it,sl in enumerate(unsl):
    axs.append(fig.add_subplot(gs[pl_row,pl_col]))
    pl_col = pl_col+1
    if pl_col == Ncol:
        pl_col = 0
        pl_row = pl_row+1
        
    BI = WordInfo_c['Nsyl']==sl
    meanDurS = WordInfo_c[BI]['mean'].values.tolist()
    meanDurS = [x for x in meanDurS if str(x)!='nan']
    OmeandurS[it] = np.mean(meanDurS)
    meanStdS = WordInfo_c[BI]['std'].values.tolist()
    meanStdS = [x for x in meanStdS if str(x)!='nan']    
    OstddurS[it] = np.mean(meanStd)
    sns.distplot(meanDurS,ax=axs[-1]).set_title('{}'.format(str(sl)))
    plt.text(0.95, 0.9, '{:.3f}'.format(OmeandurS[it])[1:], transform=axs[-1].transAxes, \
             horizontalalignment='right',verticalalignment='top')
    axs[-1].set_xlim([0,1.5])
    if it == len(unsl)-1:
        axs[-1].set_xlabel('duration (sec)')
plt.show() 
#fig.savefig(figloc + ID + '_SyllLength.pdf', format='pdf')   
#fig.savefig(figloc + ID + '_SyllLength.jpg', format='jpg')     

#%% Figure 2D
axs = list()
fig = plt.figure(constrained_layout=True, figsize = (bfi.figsize.Col2,2)) 
gs = fig.add_gridspec(1, 4)
listval = [unwl, unsl]
names = ['letters','syllables']
colnames = ['length','Nsyl']
AmeanDur = [Omeandur]+[OmeandurS]
ls_sm = list()
cntpl = 0

def doplot(cntpl, length, meanDur, val, Adur, rh, p, tit):
    axs.append(fig.add_subplot(gs[0,cntpl]))
    sns.violinplot(x=length,y=meanDur, ax=axs[-1], inner = None, hue = length>0); axs[-1].scatter(val,Adur,c=cs.cols[1],s=10)
    if it == 0:
        axs[-1].set_xticks(np.arange(1,15,2));
        axs[-1].set_xticklabels(np.arange(2,15,2))
    axs[-1].set_xlabel('N ' + names[it]); axs[-1].set_ylabel(tit); 
    axs[-1].set_title('\u03C1={:.3f}, p={:.3f}'.format(rh,p))
    axs[-1].get_legend().set_visible(False)

for it in range(2):
    # word length
    BI = np.isin(np.array(WordInfo_c[colnames[it]]), listval[it])
    meanDur = np.array(WordInfo_c[BI]['mean'].values.tolist())
    meanStd = np.array(WordInfo_c[BI]['std'].values.tolist())
    itNN = [it for it,x in enumerate(meanDur) if str(x)!='nan']
    meanDur = meanDur[itNN]
    meanStd = meanStd[itNN]
    length = np.array(WordInfo_c[BI][colnames[it]].values.tolist())
    length = length[itNN]    
    rhstd,p = scipy.stats.pearsonr(length,meanStd)
    rh,p = scipy.stats.pearsonr(length,meanDur)
    doplot(cntpl, length, meanDur, np.array(listval[it])-listval[it][0], AmeanDur[it], rh, p, 'duration')
    
    axs.append(fig.add_subplot(gs[0,cntpl+1]))
    rate = 1/(meanDur/length)
    rh,p = scipy.stats.pearsonr(length,rate)
    doplot(cntpl+1, length, rate, np.array(listval[it])-listval[it][0], 1/(AmeanDur[it]/np.array(listval[it])), rh, p, 'rate')
    
    subdata = WordInfo_c[BI]
    ls_sm.append(ols('mean ~ ' + colnames[it], data = subdata).fit())
    cntpl = cntpl + 2
plt.show()
#fig.savefig(figloc + ID + '_RateEff.pdf', format='pdf')  
#fig.savefig(figloc + ID + '_RateEff.jpg', format='jpg')  

ls_sm[0].summary()
ls_sm[1].summary()

#%% same, but for word frequency
BI = np.array(WordInfo_c['Freq']) > 0
meanDurWF = np.array(WordInfo_c[BI]['mean'].values.tolist())
meanStdWF = np.array(WordInfo_c[BI]['std'].values.tolist())
itNN = [it for it,x in enumerate(meanDurWF) if str(x)!='nan']
meanDurWF = meanDurWF[itNN]
meanStdWF = meanStdWF[itNN]
lengthWF = np.array(WordInfo_c[BI]['Freq'].values.tolist())
lengthWF = lengthWF[itNN]
rh,p = scipy.stats.pearsonr(lengthWF,meanDurWF)
rhstd,p = scipy.stats.pearsonr(lengthWF,meanStdWF)
 
#%% Figure 2B
g = sns.jointplot(x= lengthWF,y = meanDurWF, kind = 'hex', size = bfi.figsize.Col2/4)
sns.regplot(lengthWF,meanDurWF, ax=g.ax_joint,scatter=False)
g.ax_joint.set_xlabel('word frequency ($x10^{6}$)');g.ax_joint.set_ylabel('duration');g.ax_marg_x.set_title('mean')
#g.savefig(figloc + ID + '_WordFreqMean.pdf', format='pdf')  
#g.savefig(figloc + ID + '_WordFreqMean.jpg', format='jpg')  

g = sns.jointplot(x= lengthWF,y = meanStdWF, kind = 'hex', size = bfi.figsize.Col2/4)
sns.regplot(lengthWF,meanStdWF, ax=g.ax_joint,scatter=False)
g.ax_joint.set_xlabel('word frequency ($x10^{6}$)');g.ax_joint.set_ylabel('duration');g.ax_marg_x.set_title('standard deviation')
#g.savefig(figloc + ID + '_WordFreqStd.pdf', format='pdf')  
#g.savefig(figloc + ID + '_WordFreqStd.jpg', format='jpg')  

#%% Figure 2C
SL = 1
sylam = ['mono-syllabic words','bi-syllabic words']
BI = np.logical_and(np.array(WordInfo_c['Nsyl']) == SL, np.array(WordInfo_c['Freq']) > 0)
meanDurWFS = np.array(WordInfo_c[BI]['mean'].values.tolist())
meanStdWFS = np.array(WordInfo_c[BI]['std'].values.tolist())
itNN = [it for it,x in enumerate(meanDurWFS) if str(x)!='nan']
meanDurWFS = meanDurWFS[itNN]
meanStdWFS = meanStdWFS[itNN]
lengthWFS = np.array(WordInfo_c[BI]['Freq'].values.tolist())
lengthWFS = lengthWFS[itNN]

g = sns.jointplot(x= lengthWFS,y = meanDurWFS, kind = 'hex', size = bfi.figsize.Col2/4)
sns.regplot(lengthWFS,meanDurWFS, ax=g.ax_joint,scatter=False)
g.ax_joint.set_xlabel('word frequency ($x10^{6}$)');g.ax_joint.set_ylabel('duration');g.ax_marg_x.set_title(sylam[SL-1])
#g.savefig(figloc + ID + '_WordFreqMean_SL' + str(SL) + '.pdf', format='pdf')  
#g.savefig(figloc + ID + '_WordFreqMean_SL' + str(SL) +'.jpg', format='jpg')  
g = sns.jointplot(x= lengthWFS,y = meanStdWFS, kind = 'hex', size = bfi.figsize.Col2/4)
sns.regplot(lengthWFS,meanStdWFS, ax=g.ax_joint,scatter=False)
g.ax_joint.set_xlabel('word frequency ($x10^{6}$)');g.ax_joint.set_ylabel('duration');g.ax_marg_x.set_title(sylam[SL-1])
#g.savefig(figloc + ID + '_WordFreqStd_SL' + str(SL) + '.pdf', format='pdf')  
#g.savefig(figloc + ID + '_WordFreqStd_SL' + str(SL) + '.jpg', format='jpg') 

 

 

