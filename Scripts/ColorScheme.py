#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:38:45 2020

@author: sanoev
"""

#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.font_manager
from matplotlib import cycler
matplotlib.font_manager.findSystemFonts()
'/usr/share/fonts/truetype/LiberationSans-Regular.ttf'
sty = 'default'
mpl.style.use(sty)
cols = ['#5EA1D3','#E24F4B','#FFC271','#98DD87','#B598ED']
mpl.rcParams['axes.prop_cycle'] = cycler('color',cols)
plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['font.size'] = 8
plt.rcParams['pdf.fonttype'] = 42 # for the eps files
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 1.5

class CCcolormap(object):
     def __init__(self):
         self.cmap1 = clr.LinearSegmentedColormap.from_list('custom bar',['#f6f6f6','#c1231f'], N = 256)  
         self.cmap1 = 'OrRd'                                                                 
         self.cmap2 = clr.LinearSegmentedColormap.from_list('custom bar',['#317cb3','#f6f6f6','#c1231f'], N = 256) 
         self.cmap3 = clr.LinearSegmentedColormap.from_list('custom bar',['#ffac3e','#c1231f'], N = 256)
                                                                     
class baseFigInfo(object):
    def __init__(self):
        self.figsize = self.Figsize()
        self.fontsize = self.Fontsize()        
    class Figsize(object):
        def __init__(self):
            self.Col1 = 3.54
            self.Col15 = 5.51 
            self.Col2 = 7.48
    class Fontsize(object):
        def __init__(self):            
            self.title = 10
            self.axes = 8
        
colmap = CCcolormap()
bfi = baseFigInfo()      
#plt.rcParams['image.cmap'] = 
params = {'legend.fontsize': bfi.fontsize.axes,
          'figure.figsize': (bfi.figsize.Col2, bfi.figsize.Col2),
         'axes.labelsize': bfi.fontsize.axes,
         'axes.titlesize': bfi.fontsize.title,
         'xtick.labelsize':bfi.fontsize.axes,
         'ytick.labelsize':bfi.fontsize.axes}
plt.rcParams.update(params)

