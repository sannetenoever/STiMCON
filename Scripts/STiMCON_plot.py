# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:35:12 2020

@author: sanoev

STiMCON_plots

This script contains the plots which can be created form the STiMCON_core output

"""

import matplotlib.pyplot as plt
import numpy as np
import ColorScheme as cs 
cmaps = cs.CCcolormap()
bfi = cs.baseFigInfo()

#%%
class PlotObj(object):
    def __init__(self, modelOut,parameters):
         self.act = modelOut['activation']
         self.sptim = modelOut['spiketimes']
         self.fbm = modelOut['feedback']
         self.fbm2 = modelOut['feedbackdel']
         self.Nnames = parameters['LMnames'] 
         self.T = np.ma.size(self.act,1)
    
    def timecourse(self, sensory_input, nodestoplot = 'all', plotsum = False, fsize = [5,5]):
        if nodestoplot == 'all':
            nodestoplot = np.arange(0,len(sensory_input))
        Names = self.Nnames[nodestoplot]
        Ts = [0,self.T]
        act = self.act[Ts[0]:Ts[1]][nodestoplot]
        if plotsum == True:
            #x = sum(self.act[Ts[0]:Ts[1]])
            act = np.concatenate((act, np.expand_dims(sum(self.act[Ts[0]:Ts[1]]),0)), axis=0)
            Names = np.append(Names, ['sum'])
        TT = np.arange(Ts[0],Ts[1])
        fig,axs = plt.subplots(3, figsize = fsize)
        axs[0].plot(TT,sensory_input[nodestoplot,Ts[0]:Ts[1]].transpose())
        axs[0].set_title('sensory input')
        axs[0].legend(self.Nnames)
        axs[0].set_ylabel('input strength')
        axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
        axs[1].plot(TT,self.fbm2[nodestoplot,Ts[0]:Ts[1]].transpose())
        axs[1].set_title('feedback input')        
        axs[1].set_ylabel('feedback strength')
        axs[1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
        axs[2].plot(TT,act.transpose())
        axs[2].set_title('neuronal output')        
        axs[2].set_ylabel('activation strength')
        axs[2].set_xlabel('time (ms)')
        plt.tight_layout()
        plt.show()
        return fig
        
    def axfig(self, sensory_input, nodestoplot = 'all', plotsum = False, sumonly = False, fsize = [5,7]):
        if nodestoplot == 'all':
            nodestoplot = np.arange(0,len(sensory_input))
        Names = self.Nnames[nodestoplot]
        Ts = [0,self.T]
        fig,axs = plt.subplots(4, figsize = fsize)
        for plcnt in range(4):
            toplot = []
            if plcnt == 0:
                toplot = sensory_input[nodestoplot,Ts[0]:Ts[1]];
                tit = 'sensory input'
                cname = 'strength'
            elif plcnt == 1:
                toplot = self.sptim[nodestoplot]
                tit = 'supra threshold activation'
                cname = 'activation'
            elif plcnt == 2:
                toplot = self.fbm2[nodestoplot,Ts[0]:Ts[1]]
                tit = 'feedback input'
                cname = 'strength'
            elif plcnt == 3:
                toplot = self.act[nodestoplot]
                tit = 'overall activation'
                cname = 'strength'
            if plotsum == True:
                toplot = np.concatenate((toplot, np.expand_dims(sum(toplot),0)), axis=0)
                Names = np.append(Names, ['sum'])
            if toplot.min() < 0:              
                pos = axs[plcnt].imshow(toplot, aspect='auto', origin = 'lower', cmap = cmaps.cmap2)                
                pos.set_clim(-1*abs(toplot).max(),abs(toplot).max())
            else:                       
                pos = axs[plcnt].imshow(toplot, aspect='auto', origin = 'lower', cmap = cmaps.cmap1)
            x = np.linspace(Ts[0],Ts[1],5).astype(int)
            axs[plcnt].set_xticks(np.linspace(Ts[0],Ts[1],len(x)))
            axs[plcnt].set_xticklabels(x)   
            axs[plcnt].set_yticks(np.arange(0,len(toplot[:,0])))
            axs[plcnt].set_yticklabels(Names)
            if plcnt == 3:
                axs[plcnt].set_xlabel('time (ms)')
            else:
                axs[plcnt].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=True,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
            cbar = fig.colorbar(pos, ax=axs[plcnt])
            cbar.set_label(cname)
            axs[plcnt].set_title(tit, fontsize = bfi.fontsize.axes)
        plt.tight_layout()
        plt.show()
        return fig

        
