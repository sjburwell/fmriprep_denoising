
import time
import os, glob
import pandas as pd
import numpy as np
from scipy import signal
from scipy.io import savemat
from nibabel import load
from nipype.utils import NUMPY_MMAP #for some reason, works on artemis, but not other machines
from nilearn.input_data import NiftiLabelsMasker
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure

#Data source, filenames, output directories
prepdir  = '/labs/burwellstudy/data/fmri/fmriprep-es2/fmriprep'
cachedir = '/labs/burwellstudy/data/fmri/fmriprep-es2/fmriprep/denoised'
funcdat  = glob.glob(prepdir + '*/*/*/*/*bold_space-MNI152NLin2009cAsym_preproc.nii.gz')
#atlas    = './atlases/Parcels_MNI_222.nii'                       #atlasis4d should be False
#atlas    = './atlases/Power_Neuron_264ROIs_Radius5_Mask.nii'     #atlasis4d should be False
#atlas    = './atlases/Shirer2012.nii'                            #atlasis4d should be False
#atlas    = './atlases/Conn17f_atlas.nii'                         #atlasis4d should be False
atlas    = '/labs/burwellstudy/data/rois/Ray2013-ICA70.nii'      #atlasis4d should be True
#atlas    = './atlases/Gordon2016+HarvOxSubCort.nii'              #atlasis4d should be False
atlasis4d= True  #True (e.g., probabalistic atlas) or False (i.e., one volume, integer masks)
overwrite= False #overwrite contents of "denoised/sub-????" directories

from typing import NamedTuple
class MyStruct(NamedTuple):
    outid:       str
    usearoma:   bool
    nonaggr:    bool
    n_init2drop: int
    noise:      list
    addnoise:   list
    expansion:   int
    spkreg:      int
    fdthr:     float
    dvrthr:    float

#for temporal filtering cosine functions, consider: https://nipype.readthedocs.io/en/latest/interfaces/generated/nipype.algorithms.confounds.html
baseregressors = ["Cosine*","NonSteadyStateOutlier*"]
pipelines = (
MyStruct(outid='00P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=[],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors),
MyStruct(outid='01P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['GlobalSignal'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors),
MyStruct(outid='02P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['WhiteMatter', 'CSF'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors),
MyStruct(outid='03P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors),
MyStruct(outid='06P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors),
MyStruct(outid='24P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ'],expansion=2,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors),
MyStruct(outid='09P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ', 'GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors),
MyStruct(outid='36P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ', 'GlobalSignal', 'WhiteMatter', 'CSF'],expansion=2,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors),
MyStruct(outid='03P+SpkReg75thPctile',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=1,fdthr=0.2266,dvrthr=1.3992,addnoise=baseregressors),
MyStruct(outid='03P+SpkReg80thPctile',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=1,fdthr=0.2501,dvrthr=1.4295,addnoise=baseregressors),
MyStruct(outid='03P+SpkReg90thPctile',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=1,fdthr=0.3263,dvrthr=1.5138,addnoise=baseregressors), 
MyStruct(outid='09P+SpkReg75thPctile',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ','GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=1,fdthr=0.2266,dvrthr=1.3992,addnoise=baseregressors),
MyStruct(outid='09P+SpkReg80thPctile',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ','GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=1,fdthr=0.2501,dvrthr=1.4295,addnoise=baseregressors),
MyStruct(outid='09P+SpkReg90thPctile',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ','GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=1,fdthr=0.3263,dvrthr=1.5138,addnoise=baseregressors),
MyStruct(outid='36P+SpkReg75thPctile',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ','GlobalSignal', 'WhiteMatter', 'CSF'],expansion=2,
         spkreg=1,fdthr=0.2266,dvrthr=1.3992,addnoise=baseregressors),
MyStruct(outid='36P+SpkReg80thPctile',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ','GlobalSignal', 'WhiteMatter', 'CSF'],expansion=2,
         spkreg=1,fdthr=0.2501,dvrthr=1.4295,addnoise=baseregressors),
MyStruct(outid='36P+SpkReg90thPctile',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ','GlobalSignal', 'WhiteMatter', 'CSF'],expansion=2,
         spkreg=1,fdthr=0.3263,dvrthr=1.5138,addnoise=baseregressors), 
MyStruct(outid='aCompCor',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=[],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors+['aCompCor*']),
MyStruct(outid='24P+aCompCor',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ'],expansion=2,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors+['aCompCor*']),
MyStruct(outid='24P+aCompCor+4GSR',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ', 'GlobalSignal'],expansion=2,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors+['aCompCor*']),
MyStruct(outid='00P+AROMANonAgg',usearoma=False,n_init2drop=0,nonaggr=True,
         noise=[],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors),
MyStruct(outid='01P+AROMANonAgg',usearoma=False,n_init2drop=0,nonaggr=True,
         noise=['GlobalSignal'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors),
MyStruct(outid='02P+AROMANonAgg',usearoma=False,n_init2drop=0,nonaggr=True,
         noise=['WhiteMatter', 'CSF'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors),
MyStruct(outid='03P+AROMANonAgg',usearoma=False,n_init2drop=0,nonaggr=True,
         noise=['GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=baseregressors) )

idnum       = np.zeros((len(funcdat),len(pipelines)))
fdthr       = np.zeros((len(funcdat),len(pipelines)))
dvthr       = np.zeros((len(funcdat),len(pipelines)))
ntr         = np.zeros((len(funcdat),len(pipelines)))
ntrabovethr = np.zeros((len(funcdat),len(pipelines)))
pctdflost   = np.zeros((len(funcdat),len(pipelines)))
mfd         = np.zeros((len(funcdat),len(pipelines)))
medfd       = np.zeros((len(funcdat),len(pipelines)))
maxfd       = np.zeros((len(funcdat),len(pipelines)))
if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
for ii in range(0,len(funcdat)): 

    #get stuff for current case
    curfunc = funcdat[ii]
    curdir  = os.path.dirname(curfunc)
    curmask = glob.glob(curdir + '/*bold_space-MNI152NLin2009cAsym_brainmask.nii.gz')[0]
    curconf = glob.glob(curdir + '/*bold_confounds.tsv')[0]
    curaroma= glob.glob(curdir + '/*bold_space-MNI152NLin2009cAsym_variant-smoothAROMAnonaggr_preproc.nii.gz')[0]
    curcache= cachedir + '/' + os.path.basename(curfunc)[0:11]
    dim1,dim2,dim3,timepoints = load(curfunc, mmap=NUMPY_MMAP).shape

    if atlasis4d:
       masker = NiftiMapsMasker(    maps_img=atlas, detrend=True, standardize=True, mask_img=curmask, smoothing_fwhm=6)
    else:
       masker = NiftiLabelsMasker(labels_img=atlas, detrend=True, standardize=True, mask_img=curmask)

    t = time.time()
    print ('Current subject (' + str(ii) + '): ' + curfunc)
 
    if not os.path.isdir(curcache):
       os.mkdir(curcache)

    #select columns of confound tsv to reduce based upon
    confounds  = pd.read_csv(curconf,sep='\t')

    #loop "pipelines" to generate "denoised" data
    for jj in range(0,len(pipelines)):

     outfile = (curcache + '/' + os.path.basename(curfunc)[0:-7] + '_Proc-' + pipelines[jj].outid + '_ROI-' + os.path.basename(atlas)[0:-4] + '_TS.tsv')
     #if not os.path.isfile(outfile) or overwrite:
     n_init2drop  = pipelines[jj].n_init2drop
     do_nonaggr   = pipelines[jj].nonaggr
     do_expansion = pipelines[jj].expansion
     do_spikereg  = pipelines[jj].spkreg
     addnoise     = pipelines[jj].addnoise
     fd_thresh    = pipelines[jj].fdthr
     dvar_thresh  = pipelines[jj].dvrthr
     
     # 'noise' (derivs and expansions can be applied) and 'addnoise' (only 0th-lag and non-expanded allowed) columns
     noise = pipelines[jj].noise
     NoiseReg = np.ones(shape=(timepoints,1))
     if len(noise)>0:
        for kk in range(0,len(noise)):
            NoiseReg = np.concatenate(( NoiseReg, confounds.filter(regex=noise[kk])),axis=1)
     if do_expansion is 1:
        NoiseReg  = np.concatenate(( NoiseReg,np.concatenate(([np.zeros(NoiseReg.shape[1])],np.diff(NoiseReg,axis=0)),axis=0) ),axis=1)
     if do_expansion is 2:
        NoiseReg  = np.concatenate(( NoiseReg,np.concatenate(([np.zeros(NoiseReg.shape[1])],np.diff(NoiseReg,axis=0)),axis=0) ),axis=1)
        NoiseReg  = np.concatenate( (NoiseReg,np.square(NoiseReg)),axis=1)
     if len(addnoise)>0:
        for kk in range(0,len(addnoise)):
            NoiseReg = np.concatenate(( NoiseReg, confounds.filter(regex=addnoise[kk])),axis=1)
     col_mean       = np.nanmean(NoiseReg,axis=0)   #\
     inds           = np.where(np.isnan(NoiseReg))  # replace NaNs w/ column means
     NoiseReg[inds] = np.take(col_mean,inds[1])     #/

     #spike columns - taken from another script, a bit kloogey
     SpikeReg = np.ones([timepoints,1])
     if do_spikereg is 1:
        SpikeReg = (((confounds.stdDVARS > dvar_thresh) | (confounds.FramewiseDisplacement > fd_thresh))==False)*1
     if n_init2drop>0:
        SpikeReg[0:(n_init2drop)] = 0 
     censorcols   = np.where(SpikeReg==0)[0]
     SpikeCols    = np.zeros((NoiseReg.shape[0],len(censorcols)))
     SpikeCols[censorcols,range(0,len(censorcols))] = 1
     if len(np.where(SpikeReg==0)[0])>0:
        NoiseReg  = np.concatenate((NoiseReg,SpikeCols),axis=1)

     #de-mean noise[/spike] matrix, delete columns of constants
     NoiseReg = NoiseReg - np.mean(NoiseReg,axis=0)
     if any (np.mean(NoiseReg,axis=0)==0): 
        NoiseReg = np.delete(NoiseReg,np.where(np.mean(NoiseReg,axis=0)==0)[0][0],1)

     #two routes, the first if the desired approach includes AROMA nonaggressive regression
     nAROMAComps = 0
     if do_nonaggr:
        from nipype.interfaces.fsl.utils import FilterRegressor
        melodicX = np.loadtxt(glob.glob(curdir + '/*bold_MELODICmix.tsv')[0])
        melodicX2= np.concatenate((melodicX,NoiseReg),axis=1)
        noisecols= np.concatenate((np.loadtxt(glob.glob(curdir + '/*bold_AROMAnoiseICs.csv')[0],delimiter=',').astype('int'),
                                  1+np.arange(melodicX.shape[1],melodicX2.shape[1])),axis=0)
        nAROMAComps = len(noisecols)
        np.savetxt(curcache + '/tmpdesign.tsv',melodicX2,delimiter='\t')
        freg = FilterRegressor()
        freg.inputs.design_file    = curcache + '/tmpdesign.tsv'
        freg.inputs.filter_columns = list(np.concatenate((np.loadtxt(glob.glob(curdir + '/*bold_AROMAnoiseICs.csv')[0],delimiter=',').astype('int'), 
                                          1+np.arange(melodicX.shape[1],melodicX2.shape[1])),axis=0))     
        freg.inputs.in_file        = curfunc
        freg.inputs.mask           = curmask
        freg.inputs.out_file       = curcache + '/tmpAROMA.nii.gz'

        if not os.path.isfile(outfile) or overwrite: 
           print ('Non-aggressively regressing ' + str(len(freg.inputs.filter_columns)) + ' parameters from functional file, and extracting ROIs...')
           freg.run()
           #PERHAPS THE ABOVE STEP SHOULD HAVE A LINEAR DETREND TOO!
           
           #extract time-series
           roits = masker.fit_transform(freg.inputs.out_file)
           np.savetxt(outfile, roits, delimiter='\t')
        
           #delete the tmpAROMA.nii.gz
           if os.path.isfile(curcache + '/tmpAROMA.nii.gz'):
              os.remove(curcache + '/tmpAROMA.nii.gz') 
           
           elapsed = time.time() - t
           print ('Elapsed time (s) for ' + pipelines[jj].outid + ': ' + str(np.round(elapsed,1)))

     else:
        if not os.path.isfile(outfile) or overwrite:
           #extract time-series, regress out confounds
           print ('Regressing ' + str(NoiseReg.shape[1]) + ' parameters from ROI time-series...')
           roits = masker.fit_transform(curfunc,confounds=NoiseReg) 
           np.savetxt(outfile, roits, delimiter='\t') 
           
           elapsed = time.time() - t
           print ('Elapsed time (s) for ' + pipelines[jj].outid + ': ' + str(np.round(elapsed,1)))

     #store info into dataframe w/ 
     idnum[ii,jj]       = float(os.path.basename(curfunc)[4:11])
     ntr[ii,jj]         = float(timepoints)
     fdthr[ii,jj]       = pipelines[jj].fdthr
     dvthr[ii,jj]       = pipelines[jj].dvrthr
     ntrabovethr[ii,jj] = float(np.sum(SpikeReg==0)) - n_init2drop
     pctdflost[ii,jj]   = float(NoiseReg.shape[1]+nAROMAComps)/float(NoiseReg.shape[0])
     mfd[ii,jj]         = float(np.mean(confounds['FramewiseDisplacement'][1:-1])) 
     medfd[ii,jj]       = float(np.median(confounds['FramewiseDisplacement'][1:-1]))
     maxfd[ii,jj]       = float(np.max( confounds['FramewiseDisplacement'][1:-1]))

for jj in range(0,len(pipelines)):
   df = pd.DataFrame({'ID':idnum[:,jj], 
                      'TR':ntr[:,jj], 
                      'FDthr':fdthr[:,jj], 
                      'DVARthr':dvthr[:,jj], 
                      'TRabovethr':ntrabovethr[:,jj], 
                      'PctDFlost':pctdflost[:,jj], 
                      'meanFD':mfd[:,jj], 
                      'medFD':medfd[:,jj], 
                      'maxFD':maxfd[:,jj]})
   df.to_csv(path_or_buf= cachedir + '/' + pipelines[jj].outid + '.tsv',sep='\t',index=False)


