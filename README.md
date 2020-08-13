# fmriprep_denoising
Scripts for comparing different fmriprep denoising approaches on quality control functional connectivity (QC-FC) motion, QC-FC distance-dependence, and temporal degree of freedom loss.

The Python program denoise_fmriprep_output.py will execute any of several pre-set denoising pipelines (_description forthcoming_) which are mostly based on published, validated methods. For each functional file identified in its path, denoise_fmriprep_output.py will generate an RxT *tsv file containing the denoised time-series, where R is the number of "ROIs" (Regions Of Interest) in the atlas file and T is the number of TRs in the data. The atlas can be either a 3D *nii of discrete integers corresponding to ROIs, or a 4D *nii of probabalistic ROIs (e.g., ICA output).
 
The script compare_denoised.py will generate QCFC motion, distance dependence, and tDOF loss metrics.

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

Next, run the denoising program. If you don't have AFNI and FSL neuroimaging toolboxes in your system path already, now is the time to load them. Below, they are added by the "load module" command, but things may be different on different machines. 
```linux
module load afni fsl
python denoise_fmriprep_output.py \
  --prepdir=/labs/burwellstudy/data/fmri/fmriprep-es2/fmriprep \
  --atlas=./atlases/Conn17f_atlas.nii \
  --pipes=['24P+aCompCor+4GSR','03P+AROMANonAgg'] \
  --overwrite=True \
  --cachedir=./tmp \
  --funcpointer=/sub-88302*/*/*/*space-MNI152NLin2009cAsym_preproc*.nii*
```
In the above code, the arguements explained below:
* 'prepdir' refers to the directory in which fmriprep output is saved
* 'atlas' points to a Nifti file containing either a 3-dimensional "atlas" where voxels' values pertain to different regions of interest (e.g., 0=nonbrain, 1=hippocampus, 2=amygdala, etc.) or 4-dimensional series of "maps" where the 4th dimension contains statistical maps of activations (e.g., ICA weights)
* 'pipes' refers to the denoising pipelines to be requested (see below for more info)
* 'overwrite' is a booean which determines whether previous output(s) will be overwritten (default: False)
* 'cachedir' is the directory in which output files will be written (default: './tmpdir')
* 'funcpointer' is a file-filtering string that can be used to match a selection of functional files for denoising (default: '/*/*/*/*space-MNI152NLin2009cAsym_preproc*.nii*' or ALL functional files in the fmriprep directory).



