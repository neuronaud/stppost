
import STPfcts
import numpy as np
from numba import jit
from math import factorial, log
from sklearn.neighbors import KDTree
from scipy.signal import periodogram, welch
from pylab import *
import seaborn as sns
from scipy.stats import uniform
from scipy.ndimage import zoom
import importlib
import time
importlib.reload(STPfcts)


def getDelta(alphapre,alphapost,alphapost_preonly,num_recipients,num_spikes,sender_period, t_sim ,STF,baseline,isivec=[],sender_times=[]):
  V_recipients,allkernels, V_sender,time,temp, isivec, sender_times = STPfcts.OnePreNpostSTP(num_recipients=num_recipients,num_spikes=num_spikes,alphapre=alphapre,alphapost=alphapost,sender_period=sender_period, t_sim = t_sim,Poisson=True, STF=STF,baseline=baseline,isivec=isivec, sender_times=sender_times)

  catV = np.reshape(V_recipients,(int(t_sim/dt),-1))
  npops = catV.shape[1]
  sepop_save = np.zeros(npops)
  for j in range(npops):
    sepop_save[j] = STPfcts.spectral_entropy(catV[:,j], t_sim/dt, method='fft', nperseg=None, normalize=False,axis=-1)

  V_recipients,allkernels, V_sender,time,temp, isivec, sender_times = STPfcts.OnePreNpostSTP(num_recipients=num_recipients,num_spikes=num_spikes,alphapre=alphapre,alphapost=alphapost_preonly,sender_period=sender_period, t_sim = t_sim,Poisson=True,STF=STF,baseline=baseline,isivec=isivec, sender_times=sender_times)

  catV = np.reshape(V_recipients,(int(t_sim/dt),-1))
  npops = catV.shape[1]
  sepop = np.zeros(npops)
  for j in range(npops):
    sepop[j] = STPfcts.spectral_entropy(catV[:,j], t_sim/dt, method='fft', nperseg=None, normalize=False,axis=-1)


  PrePost_Minus_PreOnly = np.mean(sepop_save)-np.mean(sepop)
  return PrePost_Minus_PreOnly, sepop_save, sepop, isivec, sender_times


def plotviolins(setotvec_prepost,setotvec_preonly,preflat):
  import pandas as pd
  data = {
    '': ['pre+post'] * len(setotvec_prepost) + ['pre only'] * len(setotvec_preonly) +  ['homo. pre'] * len(preflat),
    'entropy': np.concatenate([
        setotvec_prepost,  # Category A
        setotvec_preonly, # Category B
        preflat,
    ])
  } 
  df = pd.DataFrame(data)
  fig3=figure(figsize=(1.5,3))
  ax3a=subplot(1,1,1)
  sns.violinplot(y='', x='entropy', data=df,density_norm='width')
  sns.despine()
  return fig3



sender_periods = np.arange(2, 31, 4)
alphapres = np.arange(0.1, 1, 0.1)
results_mid = np.zeros((len(sender_periods), len(alphapres)))
results_weak = np.zeros((len(sender_periods), len(alphapres)))
results_strong = np.zeros((len(sender_periods), len(alphapres)))



STF = True
if STF:
  alphapost_preonly = 1
  alphapost_weak = .8  
  alphapost_mid=0.7 
  alphapost_strong=-.2 #-.2 allows for some swithcin to STD due to post
  baseline = -1 # 0 (weak) or 1 (strong) for STD, 0 (weak) or -1 (strong) for STF
else:
  alphapost_weak = .2  # week: 0.2, mid, 0.5, strong 1
  alphapost_mid=0.5
  alphapost_strong=1
  alphapost_preonly = .1
  baseline = 1 # 0 (weak) or 1 (strong) for STD, 0 (weak) or -1 (strong) for STF


#fast, imprecise
num_recipients = 100
num_spikes = 20
t_sim = 1000
Nrep = 2
dt = .1

#mid choice
num_recipients = 50
num_spikes = 100
t_sim = 15000
Nrep=5
dt = .1

#slow, precise
num_recipients = 50
num_spikes = 100
t_sim = 15000
Nrep=50
dt = .1


# VIOLIN PLOTS
data = zeros((3, Nrep))
for k in range(Nrep): 
  if STF:
    ap=.9
    sp=2
  else:
    ap=.9
    sp=3
  PrePost_Minus_PreOnly, setotvec_prepost, setotvec_preonly,isivec, sender_times= getDelta(ap, alphapost_strong, alphapost_preonly, num_recipients, num_spikes,sp, t_sim , STF,baseline)
  ap=.1
  PrePost_Minus_PreOnly, temp, setotvec_preflat,isivec, sender_times= getDelta(ap, alphapost_strong, alphapost_preonly, num_recipients, num_spikes,sp, t_sim , STF,baseline,isivec=isivec, sender_times=sender_times)
  data[:,k] = array([
      np.nanmean(setotvec_preflat),
      np.nanmean(setotvec_preonly),
      np.nanmean(setotvec_prepost),  # Category A
  ])

fig3=figure(figsize=(1.5,3))
ax3a=subplot(1,1,1)
parts = ax3a.violinplot(transpose(data),)
for pc in parts['bodies']:
    pc.set_edgecolor('black')
    pc.set_edgecolors('black')
    pc.set_alpha(1)
    pc.set_facecolors('#D43F3A')
parts['cmaxes'].set_edgecolor('black')
parts['cmins'].set_edgecolor('black')
parts['cbars'].set_edgecolor('black')
ax3a.set_xticks([1, 2, 3], ['', '', ''])
#ax3a.set_yticks([7,8,9])
ax3a.set_yticks([4,5,6,7,8,9,10,11,12])
#ax3a.set_ylim((4,5.4))
ax3a.plot(array([1,2,3]),array([mean(data[0,:]),mean(data[1,:]),mean(data[2,:])]),color='k',marker='s',ms=5,ls='')
sns.despine()
np.savetxt('violin_plot_data.csv', data, delimiter=',', fmt='%.4e')
fig3.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/seviolin.pdf',dpi=150)
plt.show()
# END VIOLIN PLOTS



# GET HEATMAP DATA
for i, sp in enumerate(sender_periods):
    for j, ap in enumerate(alphapres):

        temp=[]
        for k in range(Nrep):
          PrePost_Minus_PreOnly, setotvec_prepost, setotvec_preonly, isivec, sender_times= getDelta(ap, alphapost_mid, alphapost_preonly, num_recipients, num_spikes,sp, t_sim , STF,baseline)
          temp.append(PrePost_Minus_PreOnly)
          print('i'+ str(i)+' j '+str(j) + ' rep '+str(k))

        results_mid[i, j] = np.nanmean(temp)

        temp=[]
        for k in range(Nrep):
          PrePost_Minus_PreOnly, setotvec_prepost, setotvec_preonly, isivec, sender_times = getDelta(ap, alphapost_weak, alphapost_preonly, num_recipients, num_spikes,sp, t_sim , STF,baseline,isivec, sender_times)
          temp.append(PrePost_Minus_PreOnly)
          print('i'+ str(i)+' j '+str(j) + ' rep '+str(k))

        results_weak[i, j] = np.nanmean(temp)


        #start_time = time.perf_counter()
        #temp=[]
        for k in range(Nrep):
          PrePost_Minus_PreOnly, setotvec_prepost, setotvec_preonly, isivec, sender_times= getDelta(ap, alphapost_strong, alphapost_preonly, num_recipients, num_spikes,sp, t_sim , STF,baseline,isivec, sender_times)
          temp.append(PrePost_Minus_PreOnly)
          print('i'+ str(i)+' j '+str(j) + ' rep '+str(k))

        #end_time = time.perf_counter()
        #print(end_time - start_time)
        #print(nanstd(temp))
        #print(nanstd(temp)/sqrt(len(~isnan(temp))))
        
        results_strong[i, j] = np.nanmean(temp)







zoomfactor = 3


interp_data_strong = zoom(results_strong, zoomfactor, order=3)
interp_data_mid = zoom(results_mid, zoomfactor, order=3)
interp_data_weak = zoom(results_weak, zoomfactor, order=3)


for k in range(3):
  if k==0: 
    interp_data = interp_data_weak*1.0
  elif k==1: 
    interp_data = interp_data_mid*1.0
  elif k==2: 
    interp_data = interp_data_strong*1.0

  close('all')
  fig1 = figure(figsize=(3, 2.5))
  if STF:
    ax1 = sns.heatmap(interp_data, vmin = -1, vmax=1,annot=False, cmap='PRGn')
  else:
    ax1 = sns.heatmap(interp_data, vmin = -2, vmax=2,annot=False, cmap='PRGn')
  ax1.set_xlabel('alphapre')
  ax1.set_ylabel('sender_period')
  ax1.set_yticks([.5,interp_data.shape[0]-.5])
  ax1.set_xticks([.5,interp_data.shape[1]-.5])
  ax1.set_xticklabels(['0','1'])


  if STF==False:
    if baseline==0:
      if k==0: 
        np.save('heatmapdata_b0.npz',interp_data_strong ,interp_data_mid, interp_data_weak)
        fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/Heatmap_STD_b0_low.pdf',dpi=150)
      elif k==1: 
        fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/Heatmap_STD_b0_mid.pdf',dpi=150)
      elif k==2: 
        fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/Heatmap_STD_b0_high.pdf',dpi=150)
    elif baseline ==1:
      if k==0: 
        np.save('heatmapdata_b1.npz',interp_data_strong ,interp_data_mid, interp_data_weak)
        fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/Heatmap_STD_b1_low.pdf',dpi=150)
      elif k==1: 
        fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/Heatmap_STD_b1_mid.pdf',dpi=150)
      elif k==2: 
        fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/Heatmap_STD_b1_high.pdf',dpi=150)
  else:
    if baseline==0:
      if k==0:
        np.save('heatmapdata_b0_STF.npz',interp_data_strong ,interp_data_mid, interp_data_weak) 
        fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/Heatmap_STF_b0_low.pdf',dpi=150)
      elif k==1: 
        fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/Heatmap_STF_b0_mid.pdf',dpi=150)
      elif k==2: 
        fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/Heatmap_STF_b0_high.pdf',dpi=150)
    elif baseline ==-1:
      if k==0: 
        np.save('heatmapdata_bm1_STF.npz',interp_data_strong ,interp_data_mid, interp_data_weak) 
        fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/Heatmap_STF_bmin1_low.pdf',dpi=150)
      elif k==1: 
        fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/Heatmap_STF_bmin1_mid.pdf',dpi=150)
      elif k==2: 
        fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/Heatmap_STF_bmin1_high.pdf',dpi=150)
  


