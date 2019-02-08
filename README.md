# fmriprep_denoising
Scripts for comparing different fmriprep denoising approaches on quality control functional connectivity (QC-FC) motion, QC-FC distance-dependence, and temporal degree of freedom loss.

1) The two scripts check_maskoverlaps.py and check_allframewisedisplacement.py check for mismatched functional/anatomical masks and cull all framewise displacements (respectively). These steps are useful to identify poor alignments (check_maskoverlaps) and proper framewise-displacement/DVARS cutoffs.
2) The first script to run should be fmriprep2denoised_atlas.py, which will for each subject and each denoising scheme, generate an RxT *tsv file containing the denoised time-series, where R is the number of "ROIs" in the atlas file and T is the number of TRs in the data. The atlas can be either a 3D *nii of discrete integers corresponding to ROIs, or a 4D *nii of probabalistic ROIs (e.g., ICA output).
3) The script compare_denoised.py will generate QCFC motion, distance dependence, and tDOF loss metrics.
