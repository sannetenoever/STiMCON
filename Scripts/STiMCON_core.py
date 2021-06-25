# -*- coding: utf-8 -*-
"""
Created on Fri May  1 09:14:09 2020
@author: sanoev

STiMCON: Speech Timing in a Model Constrained Oscillatory Network

This script contains the core code for a single layer of the STiMCON

"""

import numpy as np
import math as math

#%% define class for single layer activation parameters
class modelPara(object):
    def __init__(self, parameters):
        self.Nnodes = parameters['Nnodes']
        self.oscFreq = parameters['OsFreq']
        self.oscAmp = parameters['OsAmp']
        self.oscOffset = parameters['OsOffset']
        self.latAct_thres = parameters['activation_threshold']        
        self.feedbackInfluence = parameters['feedbackinf']
        self.feedbackDecay = parameters['feedbackdecay']
        self.feedbackDelay = parameters['feedbackdelay']
        self.feedbackMat = parameters['feedbackmat']
        self.latLatinhib = parameters['latinhibstrength']
        self.latSelexc = parameters['selfexitation']
        self.latInhib = parameters['Inhib']        
        self.fs = parameters['fs']        
        
    def runsingle(self, sensory_input):
        # define the layers
        semantic_node = np.zeros(self.Nnodes)
        feedback_conn = np.zeros(self.Nnodes)
        feedforward_conn = np.zeros(self.Nnodes)
        inhibT = np.zeros(self.Nnodes)+200*self.fs
        
        # main loop 
        Ntime = np.ma.size(sensory_input,1)
        act = np.zeros([self.Nnodes,int(Ntime)]) # activation
        sptim = np.zeros([self.Nnodes,int(Ntime)]) # spike times (1: through feedback, 2: through feedforward + feedback)
        fbm = np.zeros([self.Nnodes,int(Ntime)]) # feedback input (fbm2 is feedback input including the delay for plotting)
        
        for T in range(0,len(sensory_input[0])):    
            for nodecnt in range(0,self.Nnodes):
                feedforward_conn[nodecnt] = feedforward_conn[nodecnt]-self.feedbackDecay
                if feedforward_conn[nodecnt]<0:
                    feedforward_conn[nodecnt] = 0
                inhibT[nodecnt] = inhibT[nodecnt]+1
                semantic_node[nodecnt] = semantic_node[nodecnt]*self.latSelexc + self.oscFun(T) + sensory_input[nodecnt,T] + \
                    fbm[nodecnt,T-self.feedbackDelay] + self.inhibFun(inhibT[nodecnt])
            semantic_node = self.latin(semantic_node)
            for nodecnt in range(0,self.Nnodes):
                if semantic_node[nodecnt] > self.latAct_thres:
                    if act[nodecnt][T-1] < self.latAct_thres:
                        inhibT[nodecnt] = 0
                    sptim[nodecnt,T] = 1              
                    if sensory_input[nodecnt,T] > 0:
                        feedforward_conn = np.zeros(self.Nnodes);
                        feedforward_conn[nodecnt] = 1
                        sptim[nodecnt,T] = 2
                act[nodecnt][T] = semantic_node[nodecnt]
            feedback_conn = self.feedback(feedforward_conn)
            fbm[:,T] = feedback_conn
        fbm2 = np.concatenate((np.zeros([self.Nnodes, self.feedbackDelay]), fbm[:,0:-self.feedbackDelay]), 1)
        
        modelOut = {'activation': act,
                    'spiketimes': sptim,
                    'feedback': fbm,
                    'feedbackdel': fbm2};        
        return modelOut
    
    def oscFun(self,T):
        return (math.cos(2.0*math.pi*float(T/self.fs)*self.oscFreq+self.oscOffset)*self.oscAmp)
    
    def inhibFun(self,Ti):
        inhib = self.latInhib
        if Ti < 0.02*self.fs:
            inhib = inhib*-3 # suprathreshold activation
        elif Ti < 0.05*self.fs:
            inhib = inhib*3
        elif Ti < 0.1*self.fs:
            inhib = inhib*3  
        return inhib

    def latin(self, semantic_node):
        latmat = np.zeros([self.Nnodes,self.Nnodes])+self.latLatinhib
        np.fill_diagonal(latmat, 0)
        semantic_node.dot(latmat)
        semantic_node = semantic_node + semantic_node.dot(latmat)
        return semantic_node

    def feedback(self,feedforward_conn):
        feedbackmat = self.feedbackMat*self.feedbackInfluence
        feedback_conn = feedforward_conn.dot(feedbackmat)
        return feedback_conn
        
        
        