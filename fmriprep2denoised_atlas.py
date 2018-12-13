
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
#atlas    = '/labs/burwellstudy/data/rois/Gordon2016/Parcels_MNI_222.nii'
#atlas    = '/labs/burwellstudy/data/rois/Power2011/Power_Neuron_264ROIs_Radius5_Mask.nii'
#atlas    = '/labs/burwellstudy/data/rois/Gordon2016+SubcortRL.nii'
#atlas    = '/labs/burwellstudy/data/rois/Shirer2012.nii'
#atlas    = '/labs/burwellstudy/apps/conn17f/conn/rois/atlas.nii'
#atlas    = '/labs/burwellstudy/data/rois/Ray2013-ICA70.nii'
atlas    = '/labs/burwellstudy/data/rois/Gordon2016+HarvOxSubCort.nii'
atlasis4d= True

overwrite= False

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
cosines = ["Cosine00","Cosine01","Cosine02","Cosine03","Cosine04","Cosine05","Cosine06","Cosine07"]
pipelines = (
MyStruct(outid='00P',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=[],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=cosines),
MyStruct(outid='01P',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=['GlobalSignal'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=cosines),
MyStruct(outid='02P',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=['WhiteMatter', 'CSF'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=cosines),
MyStruct(outid='03P',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=['GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=cosines),
MyStruct(outid='06P',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=cosines),
MyStruct(outid='24P',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ'],expansion=2,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=cosines),
MyStruct(outid='09P',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ', 'GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=cosines),
MyStruct(outid='36P',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ', 'GlobalSignal', 'WhiteMatter', 'CSF'],expansion=2,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=cosines),
MyStruct(outid='36P+SpkRegFD22SD99',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ', 'GlobalSignal', 'WhiteMatter', 'CSF'],expansion=2,
         spkreg=1,fdthr=.22,dvrthr=99,addnoise=cosines),
MyStruct(outid='03P+SpkRegFD-p2261',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=['GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=1,fdthr=.2261,dvrthr=99,addnoise=cosines),
MyStruct(outid='03P+SpkRegFD-p2496',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=['GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=1,fdthr=.2496,dvrthr=99,addnoise=cosines),
MyStruct(outid='09P+SpkRegFD-p2261',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ', 'GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=1,fdthr=.2261,dvrthr=99,addnoise=cosines),
MyStruct(outid='09P+SpkRegFD-p2496',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ', 'GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=1,fdthr=.2496,dvrthr=99,addnoise=cosines),
MyStruct(outid='36P+SpkRegFD-p2261',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ', 'GlobalSignal', 'WhiteMatter', 'CSF'],expansion=2,
         spkreg=1,fdthr=.2261,dvrthr=99,addnoise=cosines),
MyStruct(outid='36P+SpkRegFD-p2496',usearoma=False,n_init2drop=5,nonaggr=False,
         noise=['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ', 'GlobalSignal', 'WhiteMatter', 'CSF'],expansion=2,
         spkreg=1,fdthr=.2496,dvrthr=99,addnoise=cosines),
MyStruct(outid='AROMANonAgg',usearoma=False,n_init2drop=5,nonaggr=True,
         noise=[],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=cosines),
MyStruct(outid='AROMANonAgg+01P',usearoma=False,n_init2drop=5,nonaggr=True,
         noise=['GlobalSignal'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=cosines),
MyStruct(outid='AROMANonAgg+02P',usearoma=False,n_init2drop=5,nonaggr=True,
         noise=['WhiteMatter', 'CSF'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=cosines),
MyStruct(outid='AROMANonAgg+03P',usearoma=False,n_init2drop=5,nonaggr=True,
         noise=['GlobalSignal', 'WhiteMatter', 'CSF'],expansion=0,
         spkreg=0,fdthr=99,dvrthr=99,addnoise=cosines) )



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
    #curgmprb= glob.glob(curdir.split('ses-')[0] + 'anat/*_T1w_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz')[0]

    if atlasis4d:
       masker = NiftiLabelsMasker(labels_img=atlas, detrend=True, standardize=True, mask_img=curmask) #, smoothing_fwhm=6)
    else:
       masker = NiftiMapsMasker(    maps_img=atlas, detrend=True, standardize=True, mask_img=curmask, smoothing_fwhm=6) 

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
     
     # maybe this check is unnecessary
     if not pipelines[jj].usearoma:
        noise     = pipelines[jj].noise
     else:
        noise     = pipelines[jj].noise                

     #noise columns
     if not noise:
         NoiseReg   = np.ones(shape=(timepoints,1))
     else:
         if len(noise)==1:
            NoiseReg = confounds[ [ col for col in list(confounds) if col.startswith(noise[0])]]
         else:
            NoiseReg = np.array(confounds[noise])
     if do_expansion is 1:
        NoiseReg   = np.concatenate(( NoiseReg,np.concatenate(([np.zeros(NoiseReg.shape[1])],np.diff(NoiseReg,axis=0)),axis=0) ),axis=1)
     if do_expansion is 2:
        NoiseReg   = np.concatenate(( NoiseReg,np.concatenate(([np.zeros(NoiseReg.shape[1])],np.diff(NoiseReg,axis=0)),axis=0) ),axis=1)
        NoiseReg   = np.concatenate( (NoiseReg,np.square(NoiseReg)), axis=1)
     if len(addnoise)>0:
        NoiseReg   = np.concatenate(( NoiseReg,confounds[addnoise]),axis=1)

     #replace NaNs w/ column means
     col_mean = np.nanmean(NoiseReg,axis=0)
     inds     = np.where(np.isnan(NoiseReg))
     NoiseReg[inds] = np.take(col_mean,inds[1])

     #spike columns - taken from another script, a bit kloogey
     SpikeReg = np.ones([timepoints,1])
     dvarnormstd = (confounds["non-stdDVARS"] - np.nanmean(confounds["non-stdDVARS"])) / np.std(confounds["non-stdDVARS"])
     dvarnormmad = 1.4826 * ((confounds["non-stdDVARS"] - np.nanmedian(confounds["non-stdDVARS"])) /  \
                     np.mean( np.abs( confounds["non-stdDVARS"] - np.nanmedian(confounds["non-stdDVARS"]) ) ))
     if do_spikereg is 1:
        SpikeReg = (((dvarnormstd > dvar_thresh) | (confounds.FramewiseDisplacement > fd_thresh))==False)*1
     if do_spikereg is 2:
        SpikeReg = (((dvarnormmad > dvar_thresh) | (confounds.FramewiseDisplacement > fd_thresh))==False)*1

     SpikeReg[0:(n_init2drop)] = 0 
     censorcols   = np.where(SpikeReg==0)[0]
     SpikeCols    = np.zeros((NoiseReg.shape[0],len(censorcols)))
     SpikeCols[censorcols,range(0,len(censorcols))] = 1
     if len(np.where(SpikeReg==0)[0])>0:
        NoiseReg  = np.concatenate((NoiseReg,SpikeCols),axis=1)

     #de-mean noise[/spike] matrix
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
           #THE ABOVE STEP SHOULD HAVE A LINEAR DETREND TOO!
           
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


