import os
import glob
import numpy as np
import pandas as pd
import scipy as sp
from nilearn import plotting #Needs at least this version: pip install nilearn==0.5.0a0 (cf. https://nilearn.github.io/whats_new.html)

denoisedir= '/labs/burwellstudy/data/fmri/fmriprep-es2/fmriprep/denoised'
pipelines = [
'00P',
'01P',
'02P',
'03P',
'06P',
'09P',
'24P',
'36P',
'03P+SpkRegFD-p2261',
'03P+SpkRegFD-p2496',
'09P+SpkRegFD-p2261',
'09P+SpkRegFD-p2496',
'36P+SpkRegFD-p2261',
'36P+SpkRegFD-p2496',
'AROMANonAgg',
'AROMANonAgg+01P',
'AROMANonAgg+02P',
'AROMANonAgg+03P']
atlas     = '/labs/burwellstudy/data/rois/Gordon2016+HarvOxSubCort.nii' #349 nodes
atlasis4d = False

centsv    = '00P'
pthr      = .001
trpctbad  =   60
badsubids = '/labs/burwellstudy/projects/twinness/scripts/esmf_poorquality.txt'

subsummary = pd.read_csv(denoisedir + '/' + centsv + '.tsv',sep='\t')
subs2drop  = list(np.unique(np.concatenate((np.loadtxt(badsubids),
                  np.array(subsummary.ID[np.where(((subsummary.TRabovethr / subsummary.TR) * 100)>trpctbad)[0]])),axis=0)).astype('int').astype('str')) #kloodge
subjects   = [i for i in os.listdir(denoisedir) if 'sub-' in i and i.split('sub-')[1] not in subs2drop]
subid      = np.zeros((len(subjects),1))
graphs     = np.zeros((len(subjects),
                     np.loadtxt(glob.glob(denoisedir + '/' + subjects[0] + '/*' + 'Proc-' + pipelines[0] + '_' + 'ROI-' + os.path.basename(atlas).split('.')[0] + '*.tsv')[0]).shape[1],
                     np.loadtxt(glob.glob(denoisedir + '/' + subjects[0] + '/*' + 'Proc-' + pipelines[0] + '_' + 'ROI-' + os.path.basename(atlas).split('.')[0] + '*.tsv')[0]).shape[1]))
qcfc_fdr   = np.zeros((len(pipelines),graphs.shape[1],graphs.shape[2]))
qcfc_fdp   = np.zeros((len(pipelines),graphs.shape[1],graphs.shape[2]))
qcfc_fdm   = np.zeros(len(pipelines))
qcfc_fdq   = np.zeros(len(pipelines))
qcfc_ddr   = np.zeros(len(pipelines))
qcfc_ddp   = np.zeros(len(pipelines))
tdof_lsm   = np.zeros(len(pipelines))
tdof_lsv   = np.zeros(len(pipelines))
for ii in range(0,len(pipelines)):
    print('Currently on: '+pipelines[ii])
    subsummary = pd.read_csv(denoisedir + '/' + pipelines[ii] + '.tsv',sep='\t')

    for jj in range(0, len(subjects)):
        subid[jj] = int(subjects[jj].split('sub-')[1])      #Cache subject's identifier
        roits = np.loadtxt(glob.glob(denoisedir+'/'+subjects[jj]+'/*'+'Proc-'+pipelines[ii]+'_'+'ROI-'+os.path.basename(atlas).split('.')[0]+'*.tsv')[0])
        roicm = np.corrcoef(roits.transpose())              #Correlation coeff among ROI time-series
        roicm[roicm==1] = .999                              #The next step can't handle ones, so kloodge
        roicm = np.arctanh(roicm)                           #Fischer z-transformed correlation coeff
        graphs[jj,:,:] = roicm

    lowertri = np.where(np.tril(graphs[jj,:,:], -1)!=0)    

    ##--- QC-FC association with subject mean FramewiseDisplacement (per Parkes et al., 2018; NeuroImage)
    allfd = np.zeros(len(subid))
    for jj in range(0,len(subid)):
        allfd[jj] = subsummary.meanFD[np.where(subsummary.ID==subid[jj][0])[0][0]]
    for jj in range(0, len(lowertri[0])):
        pearsonobj  = sp.stats.pearsonr(allfd,graphs[:,lowertri[0][jj],lowertri[1][jj]])
        qcfc_fdr[ii,lowertri[0][jj],lowertri[1][jj]] = pearsonobj[0]
        qcfc_fdp[ii,lowertri[0][jj],lowertri[1][jj]] = pearsonobj[1] 
    qcfc_fdr[ii,:,:] = qcfc_fdr[ii,:,:] + qcfc_fdr[ii,:,:].transpose()
    qcfc_fdp[ii,:,:] = qcfc_fdp[ii,:,:] + qcfc_fdp[ii,:,:].transpose()
    qcfc_fdm[ii]     = np.nanmedian(qcfc_fdr[ii,:,:])
    qcfc_fdq[ii]     = np.sum(qcfc_fdp[ii,]<pthr) / np.prod(qcfc_fdp[ii,].shape) * 100

    ##--- QC-FC distance-dependence (per Parkes et al., 2018; NeuroImage)
    if atlasis4d:
       atlas_region_coords = plotting.find_probabilistic_atlas_cut_coords(atlas)
    else:
       atlas_region_coords = plotting.find_parcellation_cut_coords(atlas)
    alldists = np.zeros((atlas_region_coords.shape[0],atlas_region_coords.shape[0]))
    for jj in range(0, len(atlas_region_coords)):
        for kk in range(0, len(atlas_region_coords)):
            alldists[jj,kk] = np.linalg.norm(atlas_region_coords[jj,]-atlas_region_coords[kk,])
    qcfc = np.tril(np.mean(graphs,axis=0),-1).ravel()
    dist = np.tril(              alldists,-1).ravel()
    qcfc = np.delete(qcfc, np.where(qcfc==0))
    dist = np.delete(dist, np.where(dist==0))
    spearmanobj = sp.stats.spearmanr(a=dist,b=qcfc,nan_policy='omit')
    qcfc_ddr[ii] = spearmanobj.correlation
    qcfc_ddp[ii] = spearmanobj.pvalue

    ##--- QC-FC association with 
    tdof_lsm[ii] = np.mean(subsummary.PctDFlost) * 100
    tdof_lsv[ii] = np.std(subsummary.PctDFlost)  * 100 






import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.25
 
# set height of bar
ranks = (-qcfc_fdq).argsort()
pnames= [pipelines[i] for i in ranks]
bars1 = qcfc_fdq[ranks]               #[12, 30, 1, 8, 22]
bars2 = np.abs(qcfc_ddr[ranks]) * 100 #[28, 6, 16, 5, 10]
bars3 = tdof_lsm[ranks]               #[29, 3, 24, 25, 17]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='r', width=barWidth, edgecolor='white', label='QCFC-Motion')
plt.bar(r2, bars2, color='g', width=barWidth, edgecolor='white', label='QCFC-DistDep')
plt.bar(r3, bars3, color='b', width=barWidth, edgecolor='white', label='tDOF lost')
 
# Add xticks on the middle of the group bars
plt.xlabel('Pipeline', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], pnames, rotation=45, horizontalalignment="right") 
plt.ylim(0,100)

# Create legend & Show graphic
plt.legend()
plt.tight_layout()
plt.show()



plt.savefig('samplefigure.pdf', bbox_inches='tight')


