# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:38:29 2020

@author: sanoev

STiMCON: Speech Timing in a Model Constrained Oscillatory Network

This script contains the code for creating sensory input

"""

import numpy as np
import scipy.signal as sp

class modelSen(object):
    def __init__(self, stimpara,parameters):
        self.Nnodes = parameters['Nnodes']
        self.wordDur = stimpara['word_duration']
        self.onsetdelay = stimpara['onsetdelay']
        self.Nnames = parameters['LMnames']
    
    def create_stim(self, seninput):
        Ntime = int(self.onsetdelay*2+seninput['tot_length'])     
        sensory_input = np.zeros([self.Nnodes,Ntime])
       
        for it,stim in enumerate(seninput['stim_ord']):
            stt = int(seninput['stim_time'][it])+self.onsetdelay
            sensory_input[stim,stt:stt+self.wordDur] = np.linspace(0,2,self.wordDur)
        return sensory_input
    
    def create_stim_vartimethres(self, seninput):
        Ntime = int(self.onsetdelay*2+seninput['tot_length'])     
        sensory_input = np.zeros([self.Nnodes,Ntime])
       
        if len(seninput['stim_ord'])>0:
            for it,stim in enumerate(seninput['stim_ord']):
                stt = int(seninput['stim_time'][it])+self.onsetdelay
                sensory_input[stim,stt:stt+self.wordDur] = np.linspace(0,seninput['intensity'][it],self.wordDur)
        else:
            for stimC in range(len(seninput['intensity'][0])):
                for it in range(self.Nnodes):
                    stt = int(seninput['stim_time'][stimC]+self.onsetdelay)
                    sensory_input[it,stt:stt+self.wordDur] = np.linspace(0,seninput['intensity'][it,stimC],self.wordDur)               
        return sensory_input
    
    def create_stim_vartimethredur(self, seninput):
        Ntime = int(self.onsetdelay*2+seninput['tot_length'])     
        sensory_input = np.zeros([self.Nnodes,Ntime])
       
        if len(seninput['stim_ord'])>0:
            for it,stim in enumerate(seninput['stim_ord']):
                stt = int(seninput['stim_time'][it])+self.onsetdelay
                curwordDur = seninput['stim_dur'][it]                
                #sensory_input[stim,stt:stt+curwordDur] = np.arange(curwordDur)*seninput['increase']
                sensory_input[stim,stt:stt+curwordDur] = np.linspace(0,seninput['intensity'][it], curwordDur)
                #sensory_input[stim,stt:stt+self.wordDur] = np.linspace(0,seninput['intensity'][it],self.wordDur
        else:
            for stimC in range(len(seninput['intensity'][0])):
                for it in range(self.Nnodes):
                    stt = int(seninput['stim_time'][stimC]+self.onsetdelay)
                    sensory_input[it,stt:stt+self.wordDur] = np.linspace(0,seninput['intensity'][it,stimC],self.wordDur)               
        return sensory_input    
    
    def create_stim_vartimethresG(self, seninput, GausW = 3):
        Ntime = int(self.onsetdelay*2+seninput['tot_length'])     
        sensory_input = np.zeros([self.Nnodes,Ntime])
       
        if len(seninput['stim_ord'])>0:
            for it,stim in enumerate(seninput['stim_ord']):
                stt = int(seninput['stim_time'][it])+self.onsetdelay
                sig = sp.windows.gaussian(self.wordDur*2, self.wordDur/GausW)
                sig = sig*seninput['intensity'][it]
                sensory_input[stim,stt:stt+self.wordDur*2] = sig
        else:
            for stimC in range(len(seninput['intensity'][0])):
                for it in range(self.Nnodes):
                    stt = int(seninput['stim_time'][stimC]+self.onsetdelay)
                    sig = sp.windows.gaussian(self.wordDur*2, self.wordDur/GausW)
                    sig = sig*seninput['intensity'][it,stimC]
                    sensory_input[it,stt:stt+self.wordDur] = sig            
        return sensory_input
    
    
    
    
    
    