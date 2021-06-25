# STiMCON 
This repository describes the script from the paper <i>Ten Oever & Martin (2021), Oscillatory tTracking of pseudo-rhythmic speech is constrained modulated by linguistic predictions in an oscillating computational model</i>

## Explanation
The repository consists of scripts belonging to the Corpus Gesproken Nederlands (CGN) and simulations and fitting with STiMCON

### CGN related files:
<b>CGN_Fig2.py:<b/>
This script extracts and plot the basic tmeporal variation in the syllables and words of the CGN related to Figure 2 of the main manuscript.

<b>CGN_Tab1_Fig3_Fig7.py:<b/>
The ordinary least square and related figures.

<b>RNN_Model.py:<b/>
The RNN model

<b>RNN_subFun.py:<b/>
Subfunction to use the RNN_Model

### STiMCON related files:
<b>STiMCON_Fig4_Fig5_Fig8A.py<b/>
Shows the basic behavior of STiMCON (Figure 4), the thresholds/timing of activation (Figure 5) and ambiguous daga overall simulations (Figure 8A)

<b>STiMCON_Fig6.py<b/>
Shows how acoustic time and stimulus time is not the same in STiMCON (Figure 6)

<b>STiMCON_Fig8C.py<b/>
Fitting of the da/ga data using the first active node as output (Figure 8C)

<b>STiMCON_Fig8D.py<b/>
Fitting of the da/ga data using the relative node activation as output (Figure 8D)

<b>STiMCON_core.py<b/>
Core script for the STiMCON model which has all the low-level things

<b>STiMCON_plot.py<b/>
Plotting output of the STiMCON

<b>STiMCON_sen.py<b/>
Creating sensory input going into the STiMCON