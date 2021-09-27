#some executable statement here
import getopt, glob, os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.io import savemat
from nibabel import load
#from nipype.utils.config import NUMPY_MMAP                 
from nipype.interfaces.afni import TProject
from nilearn.input_data import NiftiLabelsMasker    # pip install nilearn==0.5.0a0
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure

### SET-UP and CHECK REQUIREMENTS
prepdir = None
atlas = None
space = 'MNI152NLin2009cAsym'
mypipes = ['24P+aCompCor+4GSR','02P+AROMANonAgg','03P+AROMANonAgg','36P+SpkRegFD25']
cachedir = './tmpdir'
overwrite = False
funcpointer = '/*/*/*/*space-' + space + '_preproc*.nii*'

# add  highpass 

options, remainder = getopt.getopt(sys.argv[1:], "p:a:o:s:c:f:", ["prepdir=","atlas=","overwrite=","pipes=","cachedir=","funcpointer="])

for opt, arg in options:
    if opt in ('-p', '--prepdir'):
        prepdir = arg
    elif opt in ('-a', '--atlas'):
        atlas = arg
    elif opt in ('-s', '--space'):
        space = arg
    elif opt in ('-i', '--pipes'):
        mypipesstr = arg.replace(' ','')
        mypipes    = arg.replace(' ','').replace('[','').replace(']','').replace("'","").split(',')
        print(mypipesstr)
    elif opt in ('-o', '--overwrite'):
        overwrite = arg
    elif opt in ('-c', '--cachedir'):
        cachedir = arg
    elif opt in ('-f', '--funcpointer'):
        funcpointer = arg

print('#  #  #  #  #  #    FMRIPREP Denoiser    #  #  #  #  #  #')
print('FMRIPREP directory (--prepdir, str):                              '+prepdir)
print('ATLAS file (--atlas, str to *.nii):                               '+atlas)
print('PIPELINES (--pipes, list):                                        '+mypipesstr)
print('WRITE directory (--cachedir, str)              :                  '+cachedir)
print('OVERWRITE existing (--overwrite, bool)?                           '+overwrite)
print('FUNCTIONAL file pointer within prepdir root (--funcpointer, str): '+funcpointer)

if not os.path.exists(prepdir):
   sys.exit('   FMRIPREP Denoiser (Fatal error): Invalid or nonexistent prepdir path: '+prepdir)
elif not os.path.exists(atlas):
   sys.exit('   FMRIPREP Denoiser (Fatal error): Invalid or nonexistent atlas file: '+atlas)
elif not os.path.exists(cachedir):
   print('   FMRIPREP Denoiser (   Warning   ) : Nonexistent cachedir, making...')
   os.mkdir(cachedir)

nfunc = len(glob.glob(prepdir+funcpointer))
if not glob.glob(prepdir+funcpointer):
   sys.exit('   FMRIPREP Denoiser (Fatal error): Invalid --prepdir path OR invalid --funcpointer, no functional files found.')
else:
   funcdat = glob.glob(prepdir+funcpointer)
   print('   FMRIPREP Denoiser (   Running   ) : Found '+str(nfunc)+' functional files to denoise...')

##### END SETUP / CHECK

#if len(load(atlas, mmap=NUMPY_MMAP).shape)==4:
if len(load(atlas, mmap=True).shape)==4:
   atlasis4d = True
else:
   atlasis4d = False

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
    passband:   list

#for temporal filtering cosine functions, consider: https://nipype.readthedocs.io/en/latest/interfaces/generated/nipype.algorithms.confounds.html
baseregressors = ["NonSteadyStateOutlier*","non_steady_state_outlier*"]
allpipelines = (
MyStruct(outid='00P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=[],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='01P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['GlobalSignal','global_signal'],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='02P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['WhiteMatter','CSF','white_matter','csf'],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='03P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['GlobalSignal','WhiteMatter','CSF','global_signal','white_matter','csf'],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='06P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='24P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'],expansion=2,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='09P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','GlobalSignal','WhiteMatter','CSF',
                'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','global_signal','white_matter','csf'],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='36P',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','GlobalSignal','WhiteMatter','CSF',
                'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','global_signal','white_matter','csf'],expansion=2,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='09P+SpkRegFD20',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','GlobalSignal','WhiteMatter','CSF',
                'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','global_signal','white_matter','csf'],expansion=0,
         spkreg=1,fdthr=0.20,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='09P+SpkRegFD25',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','GlobalSignal','WhiteMatter','CSF',
                'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','global_signal','white_matter','csf'],expansion=0,
         spkreg=1,fdthr=0.25,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='09P+SpkRegFD30',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','GlobalSignal','WhiteMatter','CSF',
                'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','global_signal','white_matter','csf'],expansion=0,
         spkreg=1,fdthr=0.30,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='36P+SpkRegFD20',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','GlobalSignal','WhiteMatter','CSF',
                'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','global_signal','white_matter','csf'],expansion=2,
         spkreg=1,fdthr=0.20,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='36P+SpkRegFD25',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','GlobalSignal','WhiteMatter','CSF',
                'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','global_signal','white_matter','csf'],expansion=2,
         spkreg=1,fdthr=0.25,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='36P+SpkRegFD30',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','GlobalSignal','WhiteMatter','CSF',
                'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','global_signal','white_matter','csf'],expansion=2,
         spkreg=1,fdthr=0.30,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='00P+aCompCor',usearoma=False,n_init2drop=0,nonaggr=False,                                     
         noise=[],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors+['aCompCor*','a_comp_cor*',"Cosine*","cosine*"],passband=[.009,9999]),
MyStruct(outid='06P+aCompCor',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors+['aCompCor*','a_comp_cor*',"Cosine*","cosine*"],passband=[.009,9999]),
MyStruct(outid='12P+aCompCor',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'],expansion=1,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors+['aCompCor*','a_comp_cor*',"Cosine*","cosine*"],passband=[.009,9999]),
MyStruct(outid='24P+aCompCor',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'],expansion=2,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors+['aCompCor*','a_comp_cor*',"Cosine*","cosine*"],passband=[.009,9999]),
MyStruct(outid='06P+aCompCor+1GSR',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','GlobalSignal','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','global_signal'],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors+['aCompCor*','a_comp_cor*',"Cosine*","cosine*"],passband=[.009,9999]),
MyStruct(outid='12P+aCompCor+2GSR',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','GlobalSignal','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','global_signal'],expansion=1,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors+['aCompCor*','a_comp_cor*',"Cosine*","cosine*"],passband=[.009,9999]),
MyStruct(outid='24P+aCompCor+4GSR',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['X','Y','Z','RotX','RotY','RotZ','GlobalSignal','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','global_signal'],expansion=2,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors+['aCompCor*','a_comp_cor*',"Cosine*","cosine*"],passband=[.009,9999]),
MyStruct(outid='00P+AROMANonAgg',usearoma=True,n_init2drop=0,nonaggr=False,
         noise=[],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='01P+AROMANonAgg',usearoma=True,n_init2drop=0,nonaggr=False,
         noise=['GlobalSignal','global_signal'],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='02P+AROMANonAgg',usearoma=True,n_init2drop=0,nonaggr=False,
         noise=['WhiteMatter','CSF','white_matter','csf'],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='03P+AROMANonAgg',usearoma=True,n_init2drop=0,nonaggr=False,
         noise=['GlobalSignal','WhiteMatter','CSF','white_matter','csf'],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='08P+AROMANonAgg+4GSR',usearoma=True,n_init2drop=0,nonaggr=False,
         noise=['GlobalSignal','WhiteMatter','CSF','white_matter','csf'],expansion=2,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors,passband=[.009,9999]),
MyStruct(outid='00P+AROMAAgg',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=[],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors+['AROMAAggrComp*','aroma_motion*'],passband=[.009,9999]),
MyStruct(outid='01P+AROMAAgg',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['GlobalSignal','global_signal'],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors+['AROMAAggrComp*','aroma_motion*'],passband=[.009,9999]),
MyStruct(outid='02P+AROMAAgg',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['WhiteMatter','CSF','white_matter','csf'],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors+['AROMAAggrComp*','aroma_motion*'],passband=[.009,9999]),
MyStruct(outid='03P+AROMAAgg',usearoma=False,n_init2drop=0,nonaggr=False,
         noise=['GlobalSignal','WhiteMatter','CSF','global_signal','white_matter','csf'],expansion=0,
         spkreg=0,fdthr=999999,dvrthr=999999,addnoise=baseregressors+['AROMAAggrComp*','aroma_motion*'],passband=[.009,9999]) )


####### FINAL CHECKS
pipelines = []
pipenames = []
for jj in range(0,len(allpipelines)):
    if allpipelines[jj].outid in mypipes:
       pipelines.append(allpipelines[jj])
       pipenames.append(allpipelines[jj].outid)
if [item for item in mypipes if item not in pipenames]:
    pipeInvalid = [item for item in mypipes if item not in pipenames]
    invalidPipeNames = ' '.join([str(elem) for elem in pipeInvalid])
    sys.exit('   FMRIPREP Denoiser (Fatal error): Invalid pipelines requested: '+invalidPipeNames)

if not os.path.exists(os.system('which 3dTproject')):
    sys.exit('   FMRIPREP Denoiser (Fatal error): Invalid AFNI path to 3dTproject, check whether you''ve added AFNI and it contains 3dTproject')
elif not os.path.exists(os.system('which fsl_regfilt')):
    print('   FMRIPREP Denoiser (   Warning   ) : Cannot find FSL path to fsl_regfilt (crucial for AROMANonAgg pipelines, proceed with caution...')

######### FINAL CHECKS END 

idlist      = np.chararray((len(funcdat),len(pipelines)),itemsize=len(os.path.basename(funcdat[0]).split('_')[0]),unicode=True)
atlaslist   = np.chararray((len(funcdat),len(pipelines)),itemsize=len(atlas),unicode=True)
ses         = np.chararray((len(funcdat),len(pipelines)),itemsize=2,unicode=True)
task        = np.chararray((len(funcdat),len(pipelines)),itemsize=5,unicode=True)
run         = np.chararray((len(funcdat),len(pipelines)),itemsize=5,unicode=True)
fdthr       = np.zeros((len(funcdat),len(pipelines)))
dvthr       = np.zeros((len(funcdat),len(pipelines)))
ntr         = np.zeros((len(funcdat),len(pipelines)))
ntrabovethr = np.zeros((len(funcdat),len(pipelines)))
pctdflost   = np.zeros((len(funcdat),len(pipelines)))
mfd         = np.zeros((len(funcdat),len(pipelines)))
medfd       = np.zeros((len(funcdat),len(pipelines)))
maxfd       = np.zeros((len(funcdat),len(pipelines)))
mdv         = np.zeros((len(funcdat),len(pipelines)))
meddv       = np.zeros((len(funcdat),len(pipelines)))
maxdv       = np.zeros((len(funcdat),len(pipelines)))
if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
for ii in range(0,len(funcdat)): #range(0,len(funcdat)): 

   #get stuff for current case
   curfunc = funcdat[ii]
   curdir  = os.path.dirname(curfunc)
   curmask = glob.glob(curdir + '/*' +
                       curfunc.split('task-')[1].split('_')[0] + '*' +
                       curfunc.split('run-')[1].split('_')[0]  + '*' + '*space-' + space + '*brain*mask.nii*')[0]
   curconf = glob.glob(curdir + '/' + os.path.basename(curfunc)[0:11]+ '*' + 
                       curfunc.split('task-')[1].split('_')[0] + '*' + 
                       curfunc.split('run-')[1].split('_')[0]  + '*' + '*confounds*.tsv')[0]
   if not glob.glob(curdir.split('/ses-')[0]+'/anat/*space-' + space + '*dtissue*.nii*'):
      cursegm = glob.glob(curdir.split('/ses-')[0]+'/anat/*space-' + space + '*dseg*.nii*')[0]
   else:
      cursegm = glob.glob(curdir.split('/ses-')[0]+'/anat/*space-' + space + '*dtissue*.nii*')[0]
   curcache= cachedir + '/' + os.path.basename(curfunc)[0:11]
   dim1,dim2,dim3,timepoints = load(curfunc, mmap=True).shape #NUMPY_MMAP).shape
   t = time.time()
   print ('Current subject (' + str(ii) + '): ' + curfunc)

   # if the "atlas" is a set of weighted maps (e.g., ICA spatial maps), use the mapsMasker (with smoothing)
   if atlasis4d:
      masker = NiftiMapsMasker(    maps_img=atlas, detrend=True, standardize=True, mask_img=curmask, smoothing_fwhm=6)
   else:
      masker = NiftiLabelsMasker(labels_img=atlas, detrend=True, standardize=True, mask_img=curmask)

   # make subject output directory, if none exists
   if not os.path.isdir(curcache):
      os.mkdir(curcache)

   #select columns of confound tsv to reduce based upon
   confounds  = pd.read_csv(curconf,sep='\t')

   #loop "pipelines" to generate "denoised" data
   for jj in range(0,len(pipelines)):

     outfile = (curcache + '/' + os.path.basename(curfunc)[0:-7] + '_Proc-' + pipelines[jj].outid + '_ROI-' + os.path.basename(atlas)[0:-4] + '_TS.tsv')
     n_init2drop  = pipelines[jj].n_init2drop
     usearoma     = pipelines[jj].usearoma
     do_nonaggr   = pipelines[jj].nonaggr
     do_expansion = pipelines[jj].expansion
     do_spikereg  = pipelines[jj].spkreg
     addnoise     = pipelines[jj].addnoise
     fd_thresh    = pipelines[jj].fdthr
     dvar_thresh  = pipelines[jj].dvrthr
     bandpass     = pipelines[jj].passband
     
     # if usearoma==True, nullify any smoothing to be done beforehand
     # also, the functional file-derived signals should come from the existing AROMA.nii.gz, this section of code will
     # replace the contents of existing 'WhiteMatter', 'CSF', 'GlobalSignal' with new contents from the AROMA cleaned file
     nAROMAComps = 0
     tmpAROMA    = (curdir + '/tmpAROMA_' +
                      'task-' + curfunc.split('task-')[1].split('_')[0] + '_' +
                      'run-'  + curfunc.split('run-')[1].split('_')[0]  + '.nii.gz')
     tmpAROMAconf= (curdir + '/tmpAROMA_' +
                      'task-' + curfunc.split('task-')[1].split('_')[0] + '_' +
                      'run-'  + curfunc.split('run-')[1].split('_')[0]  + '_confounds.tsv')
     tmpAROMAwm  = (curcache + '/tmpAROMA_' +
                      'task-' + curfunc.split('task-')[1].split('_')[0] + '_' +
                      'run-'  + curfunc.split('run-')[1].split('_')[0]  + '_wm.nii.gz')
     tmpAROMAcsf = (curcache + '/tmpAROMA_' +
                      'task-' + curfunc.split('task-')[1].split('_')[0] + '_' +
                      'run-'  + curfunc.split('run-')[1].split('_')[0]  + '_csf.nii.gz')
     if usearoma:
        from nipype.interfaces.fsl.utils import FilterRegressor
        nAROMAComps = nAROMAComps + len(np.loadtxt(glob.glob(curdir + '/*'+
                            curfunc.split('task-')[1].split('_')[0] + '*' +
                            curfunc.split('run-')[1].split('_')[0]  + '*' + '*AROMAnoiseICs.csv')[0],delimiter=',').astype('int'))
        if (not os.path.isfile(outfile) or overwrite) or (not os.path.isfile(tmpAROMA) and overwrite):
           FilterRegressor(design_file=                   glob.glob(curdir + '/*'+
                                   curfunc.split('task-')[1].split('_')[0] + '*' +
                                   curfunc.split('run-')[1].split('_')[0]  + '*' + '*MELODIC*.tsv')[0],
                           filter_columns=list(np.loadtxt(glob.glob(curdir + '/*'+
                                   curfunc.split('task-')[1].split('_')[0] + '*' +
                                   curfunc.split('run-')[1].split('_')[0]  + '*' + '*AROMAnoiseICs.csv')[0],delimiter=',').astype('int')),
                           in_file=curfunc,
                           mask=curmask,
                           out_file=tmpAROMA).run()
        if not os.path.isfile(tmpAROMAconf):
           if not os.path.isfile(tmpAROMAwm) or not os.path.isfile(tmpAROMAcsf):
              from nipype.interfaces.fsl.maths import Threshold
              from nipype.interfaces.fsl.utils import ImageMeants
              Threshold(in_file=cursegm, thresh=2.5, out_file=tmpAROMAwm,  args=' -uthr 3.5 -kernel sphere 4 -ero -bin').run()
              Threshold(in_file=cursegm, thresh=0.5, out_file=tmpAROMAcsf, args=' -uthr 1.5 -kernel sphere 2 -ero -bin').run() 
           wmts = NiftiLabelsMasker(labels_img=tmpAROMAwm , detrend=False, standardize=False).fit_transform(tmpAROMA)
           csfts= NiftiLabelsMasker(labels_img=tmpAROMAcsf, detrend=False, standardize=False).fit_transform(tmpAROMA) 
           gsts = NiftiLabelsMasker(labels_img=curmask    , detrend=False, standardize=False).fit_transform(tmpAROMA)
           AROMAconfounds = np.concatenate( (csfts, wmts, gsts), axis=1)
           np.savetxt(tmpAROMAconf, AROMAconfounds, header='CSF\tWhiteMatter\tGlobalSignal',comments='',delimiter='\t')
        AROMAconfounds = pd.read_csv(tmpAROMAconf,sep='\t')
        if 'GlobalSignal' in list(confounds):
           confounds[['CSF','WhiteMatter','GlobalSignal']]   = AROMAconfounds[['CSF','WhiteMatter','GlobalSignal']]
        else:
           confounds[['csf','white_matter','global_signal']] = AROMAconfounds[['CSF','WhiteMatter','GlobalSignal']]

     # "noise" and "addnoise" are both regressed from the data, however, (optional) derivative and expansion terms are applied
     # to the "noise" columns, whereas no derivatives/expansions are applied to "addnoise" (i.e., which will be 0-lag/non-expanded)
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

     #spike columns - a bit kloogey
     SpikeReg = np.zeros([timepoints,1])
     if do_spikereg is 1:
        DVARS = confounds.filter(['stdDVARS','std_dvars']) 
        FD    = confounds.filter(['FramewiseDisplacement','framewise_displacement'])        
        SpikeReg = (np.sum(np.concatenate((DVARS>dvar_thresh,FD>fd_thresh),axis=1),axis=1)==1)*1 
     if n_init2drop>0:
        SpikeReg[0:(n_init2drop)] = 1 
     censorcols   = np.where(SpikeReg==1)[0]
     SpikeCols    = np.zeros((NoiseReg.shape[0],len(censorcols)))
     SpikeCols[censorcols,range(0,len(censorcols))] = 1
     if len(np.where(SpikeReg==1)[0])>0:
        NoiseReg  = np.concatenate((NoiseReg,SpikeCols),axis=1)

     #de-mean noise[/spike] matrix, delete columns of constants
     NoiseReg = NoiseReg - np.mean(NoiseReg,axis=0)
     if any (np.mean(NoiseReg,axis=0)==0): 
        NoiseReg = np.delete(NoiseReg,np.where(np.mean(NoiseReg,axis=0)==0)[0][0],1)
     noise_fn = curcache + "/NoiseReg" + "_Proc-" + pipelines[jj].outid + "_ROI-" + os.path.basename(atlas)[0:-4] + ".txt"
     np.savetxt(noise_fn,NoiseReg)

     #do the regression
     errts_fn = curcache + "/errts_3dtproject" + "_Proc-" + pipelines[jj].outid + "_ROI-" + os.path.basename(atlas)[0:-4] + ".nii"
     if (not os.path.isfile(outfile) or overwrite) and (NoiseReg.shape[1]/NoiseReg.shape[0] < .90):
        if os.path.isfile(errts_fn):
           os.remove(errts_fn)
        tproject = TProject()
        if usearoma: tproject.inputs.in_file = tmpAROMA
        else:        tproject.inputs.in_file = curfunc
        tproject.inputs.polort = 2 # 0th, 1st, 2nd-order terms
        if usearoma:
           tproject.inputs.automask = True
        else: 
           tproject.inputs.automask = False
           tproject.inputs.mask = curmask
        tproject.inputs.bandpass= tuple(bandpass)
        if NoiseReg.shape[1]>0:
           tproject.inputs.ort     = noise_fn 
        #tproject.inputs.censor  = curcache + "/SpikeReg.txt"
        #tproject.inputs.cenmode = 'NTRP'
        tproject.inputs.out_file= errts_fn   
        tproject.run()

        # get time-series
        print ('Regressed ' + str(NoiseReg.shape[1]+nAROMAComps) + ' parameters from ROI time-series...')
        roits = masker.fit_transform(errts_fn) 
        np.savetxt(outfile, roits, delimiter='\t') 
        elapsed = time.time() - t
        print ('Elapsed time (s) for ' + pipelines[jj].outid + ': ' + str(np.round(elapsed,1)))

     #store info into dataframe w/ 
     idlist[ii,jj]      = os.path.basename(curfunc).split('_')[0]
     atlaslist[ii,jj]   = atlas
     ses[ii,jj]         = curfunc.split('ses-')[1].split('/')[0]  
     task[ii,jj]        = curfunc.split('task-')[1].split('_')[0]    
     run[ii,jj]         = curfunc.split('run-')[1].split('_')[0]     
     ntr[ii,jj]         = float(timepoints)
     fdthr[ii,jj]       = float(pipelines[jj].fdthr)
     dvthr[ii,jj]       = float(pipelines[jj].dvrthr)
     ntrabovethr[ii,jj] = float(np.sum(SpikeReg==1)) - n_init2drop
     pctdflost[ii,jj]   = float(NoiseReg.shape[1]+nAROMAComps)/float(NoiseReg.shape[0])
     mfd[ii,jj]         = float(np.mean(confounds.filter(['FramewiseDisplacement','framewise_displacement'])[1:-1])) 
     medfd[ii,jj]       = float(np.median(confounds.filter(['FramewiseDisplacement','framewise_displacement'])[1:-1]))
     maxfd[ii,jj]       = float(np.max( confounds.filter(['FramewiseDisplacement','framewise_displacement'])[1:-1]))
     mdv[ii,jj]         = float(np.mean(confounds.filter(['stdDVARS','std_dvars'])[1:-1]))
     meddv[ii,jj]       = float(np.median(confounds.filter(['stdDVARS','std_dvars'])[1:-1]))
     maxdv[ii,jj]       = float(np.max( confounds.filter(['stdDVARS','std_dvars'])[1:-1]))

     if os.path.isfile(errts_fn):   os.remove(errts_fn)
     if os.path.isfile(noise_fn):   os.remove(noise_fn)

   if os.path.isfile(tmpAROMA   ):  os.remove(tmpAROMA)
   if os.path.isfile(tmpAROMAwm ):  os.remove(tmpAROMAwm)
   if os.path.isfile(tmpAROMAcsf):  os.remove(tmpAROMAcsf)

for jj in range(0,len(pipelines)):
   df = pd.DataFrame({'participant_id':idlist[:,jj], 
                      'ses_id':ses[:,jj],
                      'task_id':task[:,jj],
                      'run_id':run[:,jj],
                      'atlas':atlaslist[:,jj],
                      'TR':ntr[:,jj], 
                      'FDthr':fdthr[:,jj], 
                      'DVARthr':dvthr[:,jj], 
                      'TRabovethr':ntrabovethr[:,jj], 
                      'PctDFlost':np.around(pctdflost[:,jj],5), 
                      'meanFD':np.around(mfd[:,jj],5), 
                      'medFD':np.around(medfd[:,jj],5), 
                      'maxFD':np.around(maxfd[:,jj],5),
                      'meanDVARS':np.around(mdv[:,jj],5),
                      'medDVARS':np.around(meddv[:,jj],5),
                      'maxDVARS':np.around(maxdv[:,jj],5)})
   df.to_csv(path_or_buf= cachedir + '/' + pipelines[jj].outid + '.tsv',sep='\t',index=False)


