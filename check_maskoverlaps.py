#this is a quick and dirty check to look for cases where the functional mask and anatomical mask poorly align 
import glob
from scipy import ndimage as nd
import nibabel as nib

anatmasks = glob.glob('/home/syliaw/shared/bids/fmriprep_output/fmriprep/*/anat/*space*brain*mask.nii.gz')    #these two columns should correspond 
funcmasks = glob.glob('/home/syliaw/shared/bids/fmriprep_output/fmriprep/*/*/func/*space*brain*mask.nii.gz') # to the same space (i.e., MNI-transformed)

idlist   = np.zeros((len(anatmasks),1))
overlaps = np.zeros((len(anatmasks),1))
overlapR = np.zeros((len(anatmasks),1))
overlapL = np.zeros((len(anatmasks),1))
overlapF = np.zeros((len(anatmasks),1))
overlapB = np.zeros((len(anatmasks),1))
overlapT = np.zeros((len(anatmasks),1))
for ii in range(0,len(anatmasks)):
  idlist[ii] = int(anatmasks[ii].split('fmriprep/sub-')[1][0:7])

  #cf. https://stackoverflow.com/questions/18386302/resizing-a-3d-image-and-resampling
  anat = nib.load(anatmasks[ii]) 
  func = nib.load(funcmasks[ii])
  src  = anat.get_data() 
  targ = func.get_data()

  dsfactor = [w/float(f) for w,f in zip(targ.shape, src.shape)]
  srctarg  = nd.interpolation.zoom(src, zoom=dsfactor)

  overlaps[ii] = np.sum((targ+srctarg)==2) / np.sum((srctarg)>=1)
  overlapR[ii] = np.sum((targ[75:-1,]+srctarg[75:-1,])==2) / np.sum((srctarg[75:-1,])>=1)
  overlapL[ii] = np.sum((targ[ 0:20,]+srctarg[ 0:20,])==2) / np.sum((srctarg[ 0:20,])>=1)
  overlapF[ii] = np.sum((targ[0:-1,100:-1,]+srctarg[0:-1,100:-1,])==2) / np.sum((srctarg[0:-1,100:-1,])>=1)
  overlapB[ii] = np.sum((targ[0:-1,0:20,]+srctarg[0:-1,0:20,])==2) / np.sum((srctarg[0:-1,0:20,])>=1)
  overlapT[ii] = np.sum((targ[0:-1,0:-1,70:-1]+srctarg[0:-1,0:-1,70:-1])==2) / np.sum((srctarg[0:-1,0:-1,70:-1])>=1)

print('Overlap in a particular direction: ')
np.unique((
idlist[np.where(overlapR<np.percentile(overlapR, 1))],
idlist[np.where(overlapL<np.percentile(overlapL, 1))],
idlist[np.where(overlapF<np.percentile(overlapF, 1))],
idlist[np.where(overlapB<np.percentile(overlapB, 1))],
idlist[np.where(overlapT<np.percentile(overlapT, 1))],
))

print('Overlap overall: '
idlist(np.where(overlap<np.percentile(overlap, 3))])

