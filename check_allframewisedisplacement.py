import glob
import pandas as pd
import numpy as np
confounds = glob.glob('/labs/burwellstudy/data/fmri/fmriprep-es2/fmriprep/*/*/*/*confounds.tsv')
for ii in range(0,len(confounds)):
  curconf = pd.read_csv(confounds[ii],sep='\t')
  if ii==0:
     allfd = curconf['FramewiseDisplacement'][1:-1]
  else:
     allfd = pd.concat((allfd,curconf['FramewiseDisplacement'][1:-1]))

  tmp = curconf['stdDVARS'][1:-1]                     #standardized DVARS
  #tmp = tmp - np.median(tmp)                         #median deviation
  #tmp = tmp / (np.median(np.abs(tmp)) * 1.4826)      #norm. abs. deviation
  if ii==0:
     alldv = tmp
  else:
     alldv = pd.concat((alldv,tmp))

import matplotlib.pyplot as plt
plt.hist(allfd,2000)
plt.xlim(0,1)
plt.ylabel('FramewiseDisplacement')
plt.show()

plt.hist(alldv,2000)
plt.xlim(0,3)
plt.ylabel('DVARS')
plt.show()

print('50th pctile (FD): ' + str(np.round(np.percentile(allfd,50),4)))
print('67th pctile (FD): ' + str(np.round(np.percentile(allfd,67),4)))
print('75th pctile (FD): ' + str(np.round(np.percentile(allfd,75),4)))
print('80th pctile (FD): ' + str(np.round(np.percentile(allfd,80),4)))
print('90th pctile (FD): ' + str(np.round(np.percentile(allfd,90),4)))

print('50th pctile (DVAR): ' + str(np.round(np.percentile(alldv,50),4)))
print('67th pctile (DVAR): ' + str(np.round(np.percentile(alldv,67),4)))
print('75th pctile (DVAR): ' + str(np.round(np.percentile(alldv,75),4)))
print('80th pctile (DVAR): ' + str(np.round(np.percentile(alldv,80),4)))
print('90th pctile (DVAR): ' + str(np.round(np.percentile(alldv,90),4)))


