# fmriprep_denoising
Scripts for comparing different fmriprep denoising approaches on quality control functional connectivity (QC-FC) motion, QC-FC distance-dependence, and temporal degree of freedom loss.

The Python program denoise_fmriprep_output.py will execute any of several pre-set denoising pipelines (_description forthcoming_) which are mostly based on published, validated methods. For each functional file identified in its path, denoise_fmriprep_output.py will generate an RxT *tsv file containing the denoised time-series, where R is the number of "ROIs" (Regions Of Interest) in the atlas file and T is the number of TRs in the data. The atlas can be either a 3D *nii of discrete integers corresponding to ROIs, or a 4D *nii of probabalistic ROIs (e.g., ICA output).

# Example usage:
In Linux, first clone the GitHub repository into your local directory. Alternatively, you can use the GitHub GUI to download and unzip the code. 
```linux
git clone https://github.com/sjburwell/fmriprep_denoising.git
```

Next, move into the directory containing the cloned repository's code and create an Anaconda environment containing the necessary Python dependencies using the environment.yml file. The name of the created environment will be called python363 because it uses Python version 3.6.3, but also requires additional packages.
```linux
cd fmriprep_denoising/
conda env create -f environment.yml
conda activate python363
```

Next, run the denoising program. If you don't have AFNI and FSL neuroimaging toolboxes in your system path already, now is the time to load them. Below, they are added by the "load module" command, but things may be different on different machines. You can check whether these programs already exist in your path by using the "which fsl" or "which afni" command. 
```linux
module load afni fsl
```

Finally, once AFNI and FSL are loaded, run the program...
```linux
python denoise_fmriprep_output.py \
  --prepdir=/labs/burwellstudy/data/fmri/fmriprep-es2/fmriprep \
  --atlas=./atlases/Conn17f_atlas.nii \
  --pipes=['24P+aCompCor+4GSR','03P+AROMANonAgg'] \
  --overwrite=True \
  --cachedir=./tmpdir \
  --funcpointer=/sub-88302*/*/*/*space-MNI152NLin2009cAsym_preproc*.nii*
```
In the above code, the arguements explained below:
* 'prepdir' refers to the directory in which fmriprep output is saved
* 'atlas' points to a Nifti file containing either a 3-dimensional "atlas" where voxels' integer values pertain to different regions of interest (e.g., 0=nonbrain, 1=hippocampus, 2=amygdala, etc.) or 4-dimensional series of "maps" where the 4th dimension contains statistical maps of activations (e.g., ICA weights)
* 'pipes' refers to the denoising pipelines to be requested (see below for more info)
* 'overwrite' is a booean which determines whether previous output(s) will be overwritten (default: False)
* 'cachedir' is the directory in which output files will be written (default: './tmpdir')
* 'funcpointer' is a file-filtering string that can be used to match a selection of functional files for denoising (default: '/\*/\*/\*/\*space-MNI152NLin2009cAsym_preproc*.nii*' or ALL functional files in the fmriprep directory).

# Pipeline options:
The below *options* may be passed as a list (i.e., ['00P','09P']) to the "pipes" argument in denoise_fmriprep_output.py:
* *00P*: high-pass filter cosine functions, non-steady state outlier TR
* *01P*: 00P+global signal
* *02P*: 00P+white matter, csf
* *03P*: 01P+02P
* *06P*: 00P+motion parameters
* *09P*: 03P+06P
* *24P*: 06P+1st difference and quadratic expansion of the parameters
* *36P*: 09P+1st difference and quadratic expansion of the parameters
* *09P+SpkRegFD20*: 09P, plus spike regression of high motion (FramewiseDisplacement > 0.20) TRs
* *09P+SpkRegFD25*: 09P, plus spike regression of high motion (FramewiseDisplacement > 0.25) TRs
* *09P+SpkRegFD30*: 09P, plus spike regression of high motion (FramewiseDisplacement > 0.30) TRs
* *36P+SpkRegFD20*: 36P, plus spike regression of high motion (FramewiseDisplacement > 0.20) TRs
* *36P+SpkRegFD25*: 36P, plus spike regression of high motion (FramewiseDisplacement > 0.25) TRs
* *36P+SpkRegFD30*: 36P, plus spike regression of high motion (FramewiseDisplacement > 0.30) TRs
* *00P+aCompCor*: 00P, plus inclusion of the aCompCor columns
* *06P+aCompCor*: 06P, plus inclusion of the aCompCor columns
* *24P+aCompCor*: 24P, plus inclusion of the aCompCor columns
* *06P+aCompCor+1GSR*: 06P, plus inclusion of the aCompCor columns and global signal
* *24P+aCompCor+4GSR*: 24P, plus inclusion of the aCompCor columns, global signal its 1st difference and quadratic expansion
* *00P+AROMANonAgg*: non-aggressive ICA-AROMA filtering, done using fsl_regfilt
* *01P+AROMANonAgg*: non-aggressive ICA-AROMA filtering, done using fsl_regfilt, plus regression of global signal that was extracted after the initial ICA-AROMA filtering
* *02P+AROMANonAgg*: non-aggressive ICA-AROMA filtering, done using fsl_regfilt, plus regression of white matter and csf signals that were extracted after the initial ICA-AROMA filtering
* *03P+AROMANonAgg*: non-aggressive ICA-AROMA filtering, done using fsl_regfilt, plus regression of global signal, white matter, and csf signals that was extracted after the initial ICA-AROMA filtering
* *00P+AROMAAgg*: aggressive ICA-AROMA filtering
* *01P+AROMAAgg*: aggressive ICA-AROMA filtering, plus regression of global signal (i.e., 01P)
* *02P+AROMAAgg*: aggressive ICA-AROMA filtering, plus regression of white matter and csf signals (i.e., 02P)
* *03P+AROMAAgg*: aggressive ICA-AROMA filtering, plus regression of global signal, white matter, and csf signals (i.e., 03P)


For deeper explanation of _AROMANonAgg_ vs _AROMAAgg_ denoising, please see [discussion on Neurostars forum](https://neurostars.org/t/fmriprep-ica-aroma-filtering-including-wm-csf-etc-confounds-in-fsl-regfilt/3137/6). 

# Output:
In the cache directory, the program denoise_fmriprep_output.py generates subject directories containing tab-separated variable (TSV) files where each column reflects a single time-series for a requested region of interest (ROIs), and each row reflects the time-points of the scanning session (i.e., TRs). For each subject and each denoising approach, there should be one TSV file. Additionally, at the root of the cache directory, there are two TSV files containing meta-data from preprocessing. 
![alt text](https://github.com/sjburwell/fmriprep_denoising/blob/master/fmriprep_denoising_directory_output.JPG)
