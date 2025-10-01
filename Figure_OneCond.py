
def plotting_OneCond(time,V_recipients,num_recipients,allkernels,StoreNet,tsim,dt):
  from seaborn import violinplot
  # Plotting the results
  fig1= figure(figsize=(5, 1.5))
  ax1 = subplot(1,1,1)
  for j in range(num_recipients):
      ax1.plot(time, V_recipients[j, :])
      ax1.set_ylabel('Pot. (mV)')
      if j == num_recipients - 1:
          ax1.set_xlabel('Time (ms)')
      sns.despine()
  ax1.set_ylim((-67,-55))
  ax1.set_yticks([-65,-55])


  fig2=figure(figsize=(10, 4))
  ax2a=subplot(1,2,1)
  for j in range(num_recipients):
    ax2a.plot(np.arange(len((allkernels[j])))*dt, allkernels[j])
    ax2a.set_xlabel('Time from spike (ms)')
    ax2a.set_ylabel('Plasticity potential')

    sns.despine()



  setot = 0
  se2tot = 0
  for j in range(num_recipients):

    se = spectral_entropy(V_recipients[j, :], tsim/dt, method='fft', nperseg=None, normalize=False,axis=-1)
    setot += se
    se2tot += se**2
    #print(se)
    x = uniform.rvs(loc=.9,scale=.2,size=1)
    ax2b=subplot(1,5,4)
    ax2b.plot(x,se,'.')
    ax2c=subplot(1,5,5)
    ax2c.plot(StoreNet[j],se,'.')
    ax2c.set_xlabel('STP stregnth')

  sns.despine()
  avgse = setot/num_recipients
  varse = se2tot/num_recipients-avgse**2
  sem = np.sqrt(varse)/np.sqrt(float(num_recipients))
  print(setot,sem)
  ax2b.bar(1,avgse,width=.8,edgecolor='k',facecolor='w',yerr=sem)
  catV = np.reshape(V_recipients,(2000*10,-1))
  setot = spectral_entropy(catV[:,0], 1000/dt, method='fft', nperseg=None, normalize=False,axis=-1)
  ax2b.bar(2,setot,width=.8,edgecolor='b',facecolor='w')

  ax2b.set_xlim((0,3))
  #plt.ylim((0,5))
  ax2b.set_xticks([1,2],['single','popul.'],rotation=40)
  ax2b.set_ylabel('Spectral Entropy (bits)')

  ax2c.plot(setot)

  npops = catV.shape[1]
  sepop = np.zeros(npops)
  for j in range(npops):
    sepop[j] = spectral_entropy(catV[:,j], 1000/dt, method='fft', nperseg=None, normalize=False,axis=-1)
  sepop_save = sepop

  return sepop_save, fig1, fig2





STF = True
if STF:
  alphapost_preonly = 1
  alphapost_weak = .8  
  alphapost_mid=0.7 
  alphapost_strong=-.2 #-.2 allows for some swithcin to STD due to post
  baseline = -1 # 0 (weak) or 1 (strong) for STD, 0 (weak) or -1 (strong) for STF
  alphapre=.9 
else:
  baseline = 1 # 0 (weak) or 1 (strong) for STD, 0 (weak) or -1 (strong) for STF
  alphapost_weak = .2  # week: 0.2, mid, 0.5, strong 1
  alphapost_mid=0.5
  alphapost_strong=1
  alphapost_preonly = .1
  alphapre=.5 



sender_period = 3

num_spikes = 20
num_recipients = 10  # Number of recipient neurons
t_sim=1000
dt=.1

V_recipients,allkernels, V_sender,time, StoreNet, isivec, sender_times = OnePreNpostSTP(num_recipients=num_recipients,num_spikes=num_spikes,alphapre=alphapre,alphapost=alphapost_strong,sender_period=sender_period, t_sim = 1000, Poisson=True,STF=STF,baseline=baseline)
sepop_save, fig1, fig2=plotting_OneCond(time,V_recipients,num_recipients,allkernels,StoreNet,t_sim,dt)

if STF:
  ax = fig1.gca()
  ax.set_ylim((-67,-10))
  fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/trace_prepost_STF.pdf',dpi=150)
  fig2.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/workplots_prepost_STF.pdf',dpi=150)
else:
  fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/trace_prepost.pdf',dpi=150)
  fig2.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/workplots_prepost.pdf',dpi=150)


V_recipients,allkernels, V_sender,time, StoreNet, isivec, sender_times = OnePreNpostSTP(num_recipients=num_recipients,num_spikes=num_spikes,alphapre=alphapre,alphapost=alphapost_preonly,sender_period=sender_period, t_sim = 1000, Poisson=True,STF=STF,baseline=baseline,isivec=isivec, sender_times=sender_times)
sepop_save, fig1, fig2=plotting_OneCond(time,V_recipients,num_recipients,allkernels,StoreNet,t_sim,dt)

if STF:
  ax = fig1.gca()
  ax.set_ylim((-67,-10))
  fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/trace_preonly_STF.pdf',dpi=150)
  fig2.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/workplots_preonly_STF.pdf',dpi=150)
else:
  fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/trace_preonly.pdf',dpi=150)
  fig2.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/workplots_preonly.pdf',dpi=150)


V_recipients,allkernels, V_sender,time, StoreNet, isivec, sender_times = OnePreNpostSTP(num_recipients=num_recipients,num_spikes=num_spikes,alphapre=0.1,alphapost=alphapost_preonly,sender_period=sender_period, t_sim = 1000, Poisson=True,STF=STF,baseline=baseline,isivec=isivec, sender_times=sender_times)
sepop_save, fig1, fig2=plotting_OneCond(time,V_recipients,num_recipients,allkernels,StoreNet,t_sim,dt)

if STF:
  ax = fig1.gca()
  ax.set_ylim((-67,-10))
  fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/trace_preonly_weak_STF.pdf',dpi=150)
  fig2.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/workplots_preonly_weak_STF.pdf',dpi=150)
else:
  fig1.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/trace_preonly_weak.pdf',dpi=150)
  fig2.savefig('/Users/rnaud/Library/CloudStorage/OneDrive-UniversityofOttawa/3CurrentResearch/STPPost/FiguresRaw/workplots_preonly_weak.pdf',dpi=150)










#### segment with pre only

V_recipients,allkernels, V_sender,time, StoreNet = OnePreNpostSTP(num_recipients=num_recipients,num_spikes=num_spikes,alphapre=alphapre,alphapost=alphapost_preonly,sender_period=sender_period, t_sim = 1000,Poisson=True,STF=STF,baseline=baseline)

# Plotting the results
plt.figure(figsize=(10, 2))
for j in range(num_recipients):
    plt.plot(time, V_recipients[j, :])
    plt.title(f'Recipient Neurons Potential')
    plt.ylabel('Pot. (mV)')
    if j == num_recipients - 1:
        plt.xlabel('Time (ms)')
    sns.despine()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
for j in range(num_recipients):
  plt.plot(np.arange(len((allkernels[j])))*dt, allkernels[j])
  plt.xlabel('Time from spike (ms)')
  plt.ylabel('Plasticity potential')

  sns.despine()


plt.subplot(1,5,4)
setot = 0
se2tot = 0
for j in range(num_recipients):
  se = spectral_entropy(V_recipients[j, :], 1000/dt, method='fft', nperseg=None, normalize=False,axis=-1)
  setot += se
  se2tot += se**2
  #print(se)
  x = uniform.rvs(loc=.9,scale=.2,size=1)
  plt.subplot(1,5,4)
  plt.plot(x,se,'.')
  plt.subplot(1,5,5)
  plt.plot(StoreNet[j],se,'.')
  plt.xlabel('STP stregnth')

plt.subplot(1,5,4)

sns.despine()
avgse = setot/num_recipients
varse = se2tot/num_recipients-avgse**2
sem = np.sqrt(varse)/np.sqrt(float(num_recipients))
print(setot,sem)
plt.bar(1,avgse,width=.8,edgecolor='k',facecolor='w',yerr=sem)
catV = np.reshape(V_recipients,(2000*10,-1))
setot = spectral_entropy(catV[:,0], 1000/dt, method='fft', nperseg=None, normalize=False,axis=-1)
plt.bar(2,setot,width=.8,edgecolor='b',facecolor='w')
plt.xlim((0,3))
#plt.ylim((0,5))
plt.title('pre=post')
plt.xticks([1,2],['single','popul.'],rotation=40)
plt.ylabel('Spectral Entropy (bits)')









