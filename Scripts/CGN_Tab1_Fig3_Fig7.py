#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:01:09 2020

@author: sanoev

The ordinary least square and related figures.
This scripts extracts the best RNN model and evaluates it's performance and whether it can predict 
"""
import os
import pickle
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import math
from patsy import dmatrices
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
#from keras.utils.vis_utils import plot_model
import ColorScheme as cs
import RNN_subFun as seq2seq
cmaps = cs.CCcolormap()
bfi = cs.baseFigInfo()   

SEQUENCE_LEN = 10
MIN_WORD_FREQUENCY = 10
STEP = 1

#%% Load the RNN model and it's associated files
base = ''
loadloc = './Data/'  
    
filename = loadloc + '/WordLevel_TrainAndTestData'          
with open(filename, 'rb') as f:
    TrainAndTest = pickle.load(f)
filename = loadloc + '/WordLevel_EmbWeights' 
with open(filename, 'rb') as f:
    LayerInf = pickle.load(f)
embedding_matrix = LayerInf[0]

# save this, so you can also run stuff just ont the validation set:
filename = loadloc + '/TrainAndTestSplits'            
with open(filename, 'rb') as f:  
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

# model from 02_09
print('Build model...')
model = Sequential()
model.add(Embedding(input_dim=len(words),output_dim=len(embedding_matrix[0])))
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.add(LSTM(300, return_sequences = False, recurrent_dropout = 0.2, dropout = 0.2, activation = 'tanh'))
model.add(Dense(len(words), activation = 'softmax'))
opt = keras.optimizers.Adam(learning_rate = 0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])
model.summary()

# visualize model:
#filename = figloc + '_modelsummary.pdf'
#plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

#% get panda info (with ISI etc)
filename = loadloc + '/WordLevel_TrainAndTestData_PDinfo_light'            
with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
    PandaInfo = pickle.load(f)
    
#%% read the output evaluation files of the model
dirmodels = loadloc + '/checkpoints/'         
all_files =  os.listdir(dirmodels) 
all_files = [x for x in all_files if x.find('LSTM') > -1]
all_files.sort()

acc = []; val_acc = []; loss = []; val_loss = []; per = []; val_per = [];
for file in all_files:
    # read the accuracy from the file name..
    inxAcc = file.find('-acc')
    acc.append(float(file[inxAcc+4:inxAcc+10]))
    inxAcc = file.find('val_acc')
    val_acc.append(float(file[inxAcc+7:inxAcc+13]))
    inxAcc = file.find('-loss')
    loss.append(float(file[inxAcc+5:inxAcc+11]))
    inxAcc = file.find('val_loss')
    val_loss.append(float(file[inxAcc+8:inxAcc+14]))

# get perplexity
def linenext(Tfile, inx):
    linev = Tfile.readline()
    linev = linev.split()
    return float(linev[inx])
    
alllines = []
perfile = open(loadloc + "meanProb.txt","r+") 
prop = []; val_prop = []; per = []; val_per = []; bi = []; val_bi = [];
for line in perfile:
    propt = line.find('Epoch')
    if propt >= 0:        
        prop.append(linenext(perfile,1))
        val_prop.append(linenext(perfile,1))
        per.append(linenext(perfile,1))
        val_per.append(linenext(perfile,1))
        bi.append(linenext(perfile,1))
        val_bi.append(linenext(perfile,1))
    alllines.append(line)

# get model at minimum probability
mininx = np.argmax(val_prop)
model.load_weights(dirmodels + all_files[-1])
import copy
model2 = copy.deepcopy(model)
model2.load_weights(dirmodels + all_files[0])

#%% get the prediction from the model and extract many word parameters
# takes long. Don't rerun (now only done for test)!
import nltk
filename = loadloc + '/CorpusInfoAndCelex'            
with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
    WordInfo_c = pickle.load(f)[0] 

NLTKmodel = seq2seq.cal_perplexity_model(TT_sentences, TT_next_words)
  
for cntSenType in range(1,2):
    if cntSenType == 0:
        filename = loadloc + '/PredCorr' + '_Train'
        Csentences = TT_sentences
        Cnext_words = TT_next_words
        PandaIndex = TT_index
    else:
        filename = loadloc + '/PredCorr' + '_Test'
        Csentences = TT_sentences_test
        Cnext_words = TT_next_words_test
        PandaIndex = TT_index_test
           
    # the bigram
    tokenized_text_bi = [[sen[-1]] + [Cnext_words[it]] for it,sen in enumerate(Csentences)] # the whole sentence        
    test_data_bigram = [nltk.bigrams(t,  pad_right=False, pad_left=False) for t in tokenized_text_bi]
        
    x_pred = np.zeros((len(Csentences), SEQUENCE_LEN)) # preallocate

    PredInfo = pd.DataFrame({'pred':float(),
                           'pred_e0':float(),
                           'SOA': float(),
                           'ISI': float(),
                           'length': int(),
                           'word': str(),
                           'meanDur': float(),
                           'stdDur':float(),
                           'medianDur':float(),
                           'IQRDur':float(),
                           'NSyl': float(),
                           'Freq': float(),
                           'FreqLog': float(),    
                           'N1word': int(),
                           'N1Dur': float(),
                           'N1length': float(),                           
                           'N1meanDur': float(),
                           'N1stdDur':float(),
                           'N1medianDur':float(),
                           'N1IQRDur':float(),
                           'N1NSyl': float(),
                           'N1Freq': float(),
                           'N1FreqLog': float(),    
                           'bigram': float(),
                           'wordrate':float(),
                           'syllablerate':float(),
                           'letterrate':float()}, index = range(0,len(Csentences)))     
    
    nomin = False
    for it in range(len(Csentences)):        
        for t, word in enumerate(Csentences[it]):
            x_pred[it, t] = word_indices[word]                          
    preds = model.predict(x_pred, verbose=0)
    preds_e0 = model2.predict(x_pred, verbose=0)
    
    for it in range(len(Csentences)):
        per = NLTKmodel.perplexity(test_data_bigram[it])
        y_pred = word_indices[Cnext_words[it]]
        predsval = preds[it,y_pred]
        predsval_e0 = preds_e0[it,y_pred]
        PinxF = PandaInfo[0][PandaIndex[it]][1]-1 #file index
        PinxI = PandaInfo[0][PandaIndex[it]][2:] # word index
        PinxI_lastWord = PandaInfo[0][PandaIndex[it]][-1] # last word index
        PinxI_nextWord = PandaInfo[1][PandaIndex[it]][-1] # next word index
        PinxIAll = PinxI + [PinxI_nextWord]
        
        wordlength = np.zeros([10,1])
        nsyl = 0
        nlet = 0
        for t, word in enumerate(Csentences[it]):                
            wordlength[t] = len(word)
            BI = WordInfo_c['word'] == word
            if BI.any():
                Info = WordInfo_c[BI]
                if Info['Nsyl'].values.tolist()[0] > 0:
                    nsyl = nsyl+Info['Nsyl'].values.tolist()[0]
                else:
                    nsyl = nsyl+2
                nlet = nlet + Info['length'].values.tolist()[0]
            else:
                nlet = nlet + 5
                nsyl = nsyl + 2
                
        start = np.array(PandaInfo[2][PinxF].iloc[PinxIAll,1])  # begin of the word
        SOA = start[1:]-start[0:-1]
        stimDur = PandaInfo[2][PinxF].iloc[PinxIAll,2]-start  
        meanrate = 1/np.mean(SOA)
        
        fullsenDur = start[-1]-start[0]
        PwordRate = 1/(fullsenDur/10)
        Psylrate = 1/(fullsenDur/nsyl)
        Pletterrate = 1/(fullsenDur/nlet)
        
        N1word = PandaInfo[2][PinxF].iloc[PinxI_lastWord,0]
        start = PandaInfo[2][PinxF].iloc[PinxI_lastWord,1] # begin of the word
        end = PandaInfo[2][PinxF].iloc[PinxI_lastWord,2] # end of the word
        N1Dur = end-start
        SOAs= PandaInfo[2][PinxF].iloc[PinxI_nextWord,1]-start
        ISIs=PandaInfo[2][PinxF].iloc[PinxI_nextWord,1]-end
        AL=len(N1word)        
       
        BI = WordInfo_c['word'] == Cnext_words[it]
        BIN1 = WordInfo_c['word'] == N1word
        if BI.any():
            L = WordInfo_c[BI].values.tolist()[0]
        else:
            L = list(np.zeros(16))
        if BIN1.any():
            LN1 = WordInfo_c[BIN1].values.tolist()[0]
        else:
            LN1 = list(np.zeros(16))
                
        PredInfo.loc[it] = [predsval,predsval_e0, SOAs,ISIs,AL] + [L[0]] + L[3:7] + [L[10]] + L[13:15] + \
            [LN1[0], N1Dur, LN1[15]] + LN1[3:7] + [LN1[10]] + LN1[13:15] + [per,PwordRate,Psylrate,Pletterrate] 
            
        if SOAs<0:
            nomin = True
            print('error')
        if nomin:
            break
        
    #% save the final wordlist panda info
    #with open(filename, 'wb') as f: 
    #    pickle.dump(PredInfo,f)  

#%%
filename = loadloc + '/PredCorr' + '_Test'          
with open(filename, 'rb') as f:  
    PredInfo = pickle.load(f)     
    
#%% Table 1
PredInfo = PredInfo.replace([np.inf,-np.inf], np.nan)
PredInfo['bigramLog'] = np.log(np.array(PredInfo['bigram']))
PredInfo['predLog'] = np.array(PredInfo['pred'])**(1/6)
PredInfo['N1meanDur'] = PredInfo['N1meanDur'].replace((0.00), np.nan)
PredInfo['N1meanDurLog'] = np.log(np.array(PredInfo['N1meanDur']))
PredInfo['SOAlog'] = np.log(np.array(PredInfo['SOA']))

BI = np.isnan(np.array(PredInfo['bigram'])) == False
sub_pred = PredInfo[BI]
BI = np.array(sub_pred['N1Freq']>0)
sub_predinfo = sub_pred[BI].select_dtypes(include=[np.number])
sub_predinfo2 = sub_pred[BI].select_dtypes(include=[np.number]).apply(scipy.stats.zscore) # for standardized betas..

columnofi = ['predLog','bigramLog','N1Freq','N1meanDurLog','syllablerate']
features = "+".join(columnofi)
y, X = dmatrices('SOA ~' + features, sub_predinfo, return_type='dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

CC = sub_predinfo.corr()
lm = ols('SOAlog ~ ' + features, data = sub_predinfo).fit()

#need for the plots later:
sub_predinfo2['SOAlog'] = sub_predinfo['SOAlog']
sub_predinfo2['predLog'] = sub_predinfo['predLog']

# plot summary
lm.summary()
   
#%% Fig 3B
lm = ols('SOAlog ~ ' + features, data = sub_predinfo2).fit()
X = np.linspace(0,1,100)**(1/6)
X2 = np.zeros(len(X))
d = dict(predLog=X,bigramLog=X2,N1Freq=X2,N1meanDurLog=X2,syllablerate=X2)
PV = lm.predict(exog =d)

ax = list()
fig = plt.figure(constrained_layout=True, figsize = (bfi.figsize.Col15*(3/5),bfi.figsize.Col15*(3/5)))
gs = fig.add_gridspec(2, 2)
ax.append(fig.add_subplot(gs[1,0]))
ax[-1].plot(X, PV)
ax[-1].set_xlabel('Pred')
ax[-1].set_ylabel('onset delays')
x = np.array([0, 0.25, 0.5, 0.75, 1])
ax[-1].set_xticks(x) 
ax[-1].set_xticklabels(np.round(x**6,4), rotation=45)
y = np.array([0,-1,-2,-3])
ax[-1].set_ylim([-3.5,0.5])
ax[-1].set_yticks(y)
ax[-1].set_yticklabels(np.round(np.exp(y),3))
ax.append(fig.add_subplot(gs[0,0]))
axs = sns.distplot(sub_predinfo2['predLog'])
ax.append(fig.add_subplot(gs[1,1]))
sns.distplot(sub_predinfo2['SOAlog'],vertical=True)
ax[-1].set_yticks(y);
ax[-1].set_ylim([-3.5,0.5])
plt.show()

#fig.savefig(figloc + ID + '_fitline.pdf', format='pdf')  
#fig.savefig(figloc + ID + '_fitline.jpg', format='jpg')  

#%% Fig 3C
std = 0.5
#% get values within this std for the other features
BI = np.abs(sub_predinfo2['N1meanDurLog']) < std
sub = sub_predinfo2[BI]
BI2 = np.abs(sub['bigramLog']) < std
sub = sub[BI2]
BI3 = np.abs(sub['syllablerate']) < std
sub = sub[BI3]

pred = sub['predLog']
SOA = (sub['SOAlog'])

g = sns.jointplot(x= pred,y = SOA, kind = 'hex', size = bfi.figsize.Col15*(2/5))
sns.regplot(pred,SOA,ax=g.ax_joint,scatter=False)
y = np.array([0,-1,-2,-3])
g.ax_joint.set_xticks(x)
g.ax_joint.set_xticklabels(np.round(x**6,4), rotation=45)
g.ax_joint.set_yticks(y)
g.ax_joint.set_yticklabels(np.round(np.exp(y),3))
g.ax_joint.set_ylim([-3.5,0.5])

np.corrcoef(pred,SOA)
#g.savefig(figloc + ID + '_pltwithinstd.pdf', format='pdf')  
#g.savefig(figloc + ID + '_pltwithinstd.jpg', format='jpg')  

#%% morph the prediction based on syllable rate + sinusoid
# prediction is that higher predictions lead to lower SOAs
# we can model this as: pred = -Asin(T*2*pi*w) (negative as higher pre)
# as it effectively is a normalization curve, do R-square splitting (takes long since this requires permutations)
Nperm = 1000
msk = np.zeros([Nperm,len(sub_predinfo)], bool)
for perm in range(Nperm):
    msk[perm,:] = np.random.rand(len(sub_predinfo)) < 0.9;

lm_base = ols('SOA ~ pred  + N1meanDur + syllablerate', data=sub_predinfo).fit()
k2_base,p = scipy.stats.normaltest(sub_predinfo['predLog'])

pred = np.array(sub_predinfo['pred'])
F = np.array(sub_predinfo['syllablerate'])

APbase = np.zeros(Nperm)
sub_predinfo['newpred_syl'] = 0
for perm in range(Nperm):
    train = sub_predinfo[msk[perm,:]]
    test = sub_predinfo[~msk[perm,:]]
    lm_split = ols('SOA ~ newpred_syl + N1meanDur + syllablerate', data=train).fit()
    predV = lm_split.predict(test)
    APbase[perm] = np.corrcoef(test['SOA'], predV)[0,1]**2
saveRQ_base_split = np.mean(APbase)
saveRQ_base_splitM = np.median(APbase)

# preallocations
saveAIC = np.zeros([20,20])
saveRQ = np.zeros([20,20])
saveRQ_con = np.zeros([20,20])
saveRQ_pval = np.zeros([20,20])
saveRQ_tval = np.zeros([20,20])
saveRQ_split = np.zeros([20,20])
saveRQ_splitM = np.zeros([20,20])
saveRQ_betaV = np.zeros([20,20])
saveRQ_mse = np.zeros([20,20])
k2 = np.zeros([20,20])
NA = np.linspace(-4,-1,20)
Noff = np.linspace(-math.pi, math.pi,20)

#%
for it,A in enumerate(NA):
    for itOff, off in enumerate(Noff):
        nums = (np.arcsin(pred/A)-off)/(2.0*math.pi*F)
        SOAt = np.arcsin(np.array(sub_predinfo['SOA']))
        sub_predinfo['newpred_syl'] = nums
        sub_predinfo['newSOA'] = SOAt
        lm_newpred = ols('SOA ~ newpred_syl + N1meanDur + syllablerate', data=sub_predinfo).fit()
        lm_newpred_cont = ols('SOAt ~ newpred_syl + N1meanDur + syllablerate', data=sub_predinfo).fit()
        # split and extract r-square        
        AP = np.zeros(Nperm)
        mse = np.zeros(Nperm)
        coef = np.zeros(Nperm)
        for perm in range(Nperm):
            train = sub_predinfo[msk[perm,:]]
            test = sub_predinfo[~msk[perm,:]]
            lm_split = ols('SOA ~ newpred_syl + N1meanDur + syllablerate', data=train).fit()
            coef[perm] = lm_split.params[1] # non-normalized coefficient
            predV = lm_split.predict(test)
            AP[perm] = np.corrcoef(test['SOA'], predV)[0,1]**2
            mse[perm] = np.mean((test['SOA']-predV)**2)
        saveRQ_tval[it,itOff], saveRQ_pval[it,itOff] = scipy.stats.ttest_rel(AP, APbase)
        # all values to be saved
        saveRQ_split[it,itOff] = np.mean(AP)   
        saveRQ_splitM[it,itOff] = np.median(AP)
        saveAIC[it,itOff] = lm_newpred.aic
        saveRQ[it,itOff] = lm_newpred.rsquared
        saveRQ_con[it,itOff] = lm_newpred_cont.rsquared
        saveRQ_betaV[it,itOff] = np.mean(coef)
        saveRQ_mse[it,itOff]=np.mean(mse)
        k2[it,itOff], p = scipy.stats.normaltest(nums)

#% save the final wordlist panda info
#filename = loadloc + '/arcsintransform'
#with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([saveRQ, saveRQ_con, saveRQ_split, saveRQ_splitM, saveRQ_tval, saveRQ_pval, saveRQ_base_split, saveRQ_base_splitM, saveAIC, k2, lm_base, k2_base,saveRQ_betaV],f)  

#%% Figure 7    
pred = np.array(sub_predinfo['pred'])
F = np.array(sub_predinfo['syllablerate'])
filename = loadloc + '/arcsintransform'
with open(filename, 'rb') as f:
    AF = pickle.load(f)  
saveRQ_split = AF[2]
plot= saveRQ_split
   
fig = plt.figure(constrained_layout=True,  figsize = (bfi.figsize.Col15,3))
grid = plt.GridSpec(2, 4, figure = fig)
ax1 = plt.subplot(grid[:,0:2])
#fig, ax1 = plt.subplots(figsize=(5,3))
pos = ax1.imshow(plot, aspect= 'auto', interpolation='none', extent = [-math.pi, math.pi,1,4], cmap = cmaps.cmap1)  
plt.xlabel('phase offset (radians)')
plt.ylabel('Amplitute (a.u.)')
cbar = plt.colorbar(pos)      
cbar.set_label('R$^{2}$')

#% show one example
NA = np.linspace(-4,-1,20)
Noff = np.linspace(-math.pi, math.pi,20)
inx = plot[16,:].argmax()
A = NA[16]
off = Noff[inx]
print(A)
print(off/math.pi)
nums = (np.arcsin(pred/A)-off)/(2.0*math.pi*F);
for chit,ch in enumerate(nums):
    if np.isnan(ch):
        nums[chit] = (np.arcsin(1)-off)/(2.0*math.pi*F[chit]);
        
nums = nums*1000
ax2 = plt.subplot(grid[0,2:])
v = sns.distplot(pred, ax = ax2, kde=False)
v.set_xlabel('prediction')
v.set_ylabel('N')
ax3 = plt.subplot(grid[1,2:])
v = sns.distplot(nums, ax = ax3, kde=False)
v.set_xlabel('time shift (ms)')
v.set_ylabel('N')
plt.show()

#fig.savefig(figloc + ID + '_ArcSinMorph2_' + PL + '.pdf', format='pdf')  
#fig.savefig(figloc + ID + '_ArcSinMorph.jpg', format='jpg') 
 