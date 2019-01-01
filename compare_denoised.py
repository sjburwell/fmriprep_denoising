import os
import glob
import numpy as np
import pandas as pd
import scipy as sp
from nibabel import load
from nipype.utils import NUMPY_MMAP                 #may only work on certain systems??
from nilearn import plotting #Needs at least this version: pip install nilearn==0.5.0a0 (cf. https://nilearn.github.io/whats_new.html)

#these will be required inputs
bidsdir   = '/labs/mctfr-fmri/bids/es'
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
'03P+SpkReg80thPctile',
'09P+SpkReg80thPctile',
'36P+SpkReg80thPctile',
'00P+aCompCor',
'24P+aCompCor',
'24P+aCompCor+4GSR',
'00P+AROMANonAgg',
'01P+AROMANonAgg',
'02P+AROMANonAgg',
'03P+AROMANonAgg',
'00P+AROMAAgg',
'01P+AROMAAgg',
'02P+AROMAAgg',
'03P+AROMAAgg']

#these will be optional inputs
atlas     = './atlases/Gordon2016+HarvOxSubCort.nii' #should save this to "allsubjdenoised" from fmriprep2denoised_atlas.py 349 nodes
pthr      = .001                                     #p-criterion for quantifying edges (uncorrected)

reconvers =['syngo_MR_B17','syngo_MR_D13D','syngo_MR_E11'] #Trio, ESP, ESQ (respectively)
centsv    = '00P' #or, pipelines[0]
trpctbad  =   60
badidlist = '/labs/burwellstudy/projects/twinness/scripts/esmf_poorquality.txt'

if len(load(atlas, mmap=NUMPY_MMAP).shape)==4:
   atlasis4d = True
else:
   atlasis4d = False

#this section currently not very well-coded or transparent, should redo
allsubjbids     = pd.read_csv(bidsdir+'/participants.tsv','\t')
allsubjdenoised = pd.read_csv(denoisedir + '/' + centsv + '.tsv',sep='\t')    #should just include whole sub-???? string
mapidx          = list()
for ii in range(0, len(allsubjdenoised.participant_id)):
    if np.ravel(np.where(allsubjdenoised.participant_id[ii]==allsubjbids.participant_id))[0]:
       mapidx.append(np.ravel(np.where(allsubjdenoised.participant_id[ii]==allsubjbids.participant_id))[0])
allsubjbids     = allsubjbids.reindex(np.array(mapidx))
allsubjdenoised = allsubjdenoised.join( allsubjbids.iloc[:,1:].reset_index(drop=True) )
subj2drop       = (allsubjdenoised.participant_id[ ((allsubjdenoised.TRabovethr / allsubjdenoised.TR) * 100)>trpctbad ].tolist() + 
                   open(badidlist).read().split() + 
                   allsubjdenoised.participant_id[ np.in1d(allsubjdenoised.SoftwareVersions,reconvers)==False ].tolist())

#resume...
subjects   = [i for i in os.listdir(denoisedir) if 'sub-' in i and i not in subj2drop]
idlist     = np.chararray(len(subjects),itemsize=len(subjects[0]),unicode=True)
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
    allsubjdenoised = pd.read_csv(denoisedir + '/' + pipelines[ii] + '.tsv',sep='\t')

    for jj in range(0, len(subjects)):
        idlist[jj] = subjects[jj]                           #Cache subject's identifier
        roits = np.loadtxt(glob.glob(denoisedir+'/'+subjects[jj]+'/*'+'Proc-'+pipelines[ii]+'_'+'ROI-'+os.path.basename(atlas).split('.')[0]+'*.tsv')[0])
        roicm = np.corrcoef(roits.transpose())              #Correlation coeff among ROI time-series
        roicm[roicm==1] = .999                              #The next step can't handle ones, so kloodge
        roicm = np.arctanh(roicm)                           #Fischer z-transformed correlation coeff
        graphs[jj,:,:] = roicm

    lowertri = np.where(np.tril(graphs[jj,:,:], -1)!=0)    

    ##--- QC-FC association with subject mean FramewiseDisplacement (per Parkes et al., 2018; NeuroImage)
    allfd = np.zeros(len(idlist))
    for jj in range(0,len(idlist)):
        allfd[jj] = allsubjdenoised.meanFD[np.where(allsubjdenoised.participant_id==idlist[jj])[0]]
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

    ##--- QC-FC association with temporal degrees of freedom loss
    tdof_lsm[ii] = np.mean(allsubjdenoised.PctDFlost) * 100
    tdof_lsv[ii] = np.std(allsubjdenoised.PctDFlost)  * 100 






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


