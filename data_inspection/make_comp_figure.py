#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 12:28:39 2021

@author: manusdonahue
"""

import os

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import matplotlib.cm as cm
import sys
from matplotlib.colors import ListedColormap

mr_id = 'SCD_P034'
main_folder = '/Users/manusdonahue/Documents/Sky/t1_volumizers/'
figname = f'seg_comp_{mr_id}.png'

adder = 10
adjuster = 0

spm_slice = 75+adder
fs_slice = 75+adder
fs_seg_slice_num = 128-adder+adjuster


fs_folder = '/Volumes/DonahueDataDrive/freesurfer_subjects_scd/'
spm_folder = '/Users/manusdonahue/Documents/Sky/scd_t1s/'
sienax_folder = '/Users/manusdonahue/Documents/Sky/sienax_segmentations/'

outname = os.path.join(main_folder, figname)





the_t1 = os.path.join(spm_folder, f'{mr_id}.nii')
           
brain_data = nib.load(the_t1)
brain_mat = brain_data.get_fdata()
brain_shape = brain_mat.shape
brain_slice = brain_mat[:,:,spm_slice]

plt.style.use('dark_background')

#fig, axrows = plt.subplots(3, 1, figsize=(4,12))
fig, axrows = plt.subplots(2, 2, figsize=(12,6))

for i in axrows:
    for j in i:
        j.axis('off')
        j.imshow(np.rot90(brain_slice,1), cmap=matplotlib.cm.gray)
    

fs_t1 = os.path.join(fs_folder, mr_id, 'mri', 'orig.mgz')   

fs_brain_data = nib.load(the_t1)
fs_brain_mat = brain_data.get_fdata()
fs_brain_shape = brain_mat.shape
fs_brain_slice = brain_mat[:,:,fs_slice] 

"""
ax_with_fs_backer = [2]
for i in ax_with_fs_backer:
    axrows[i].axis('off')
    axrows[i].imshow(np.rot90(fs_brain_slice,1), cmap=matplotlib.cm.gray)
"""

# let's dp the SPM segmentations
# c1 is gm, c2 is wm, c3 is csf
tissue_types = ['gm', 'wm', 'csf']
tissue_numbers = ['c1', 'c2', 'c3']
cmaps = ['red', 'white', 'cornflowerblue']


for t_type, t_num, cmap in zip(tissue_types, tissue_numbers, cmaps):
    the_file = os.path.join(spm_folder, f'{t_num}{mr_id}.nii')
    
    use_cmap = colors.ListedColormap([cmap, cmap])
    
    the_data = nib.load(the_file)
    the_mat = the_data.get_fdata()
    the_slice = the_mat[:,:,spm_slice]
    
    the_slice[the_slice > 1.] = 1
    
    slicealphas = the_slice * 0.5
    
    tissue_mask = np.ma.masked_where(the_slice == 0, the_slice)
    
    axrows[0][1].imshow(np.rot90(tissue_mask), cmap=use_cmap, alpha=np.rot90(slicealphas))





# let's do the FS segmentations
fs_seg_file = os.path.join(fs_folder, mr_id, 'mri', 'aseg-in-rawavg.mgz')
fs_seg_data = nib.load(fs_seg_file)
fs_seg_mat = fs_seg_data.get_fdata()
fs_seg_shape = fs_seg_mat.shape
#fs_seg_slice = fs_seg_mat[:,fs_seg_slice_num,:]
fs_seg_slice = fs_seg_mat[:,:,fs_slice]

gm_codes = [3, 42, 8, 47, 11, 50]
csf_codes = [4, 43, 14, 15, 25]

not_wm_codes = [0]
not_wm_codes.extend(gm_codes)
not_wm_codes.extend(csf_codes)


gm = np.isin(fs_seg_slice, gm_codes)
csf = np.isin(fs_seg_slice, csf_codes)
wm = ~np.isin(fs_seg_slice, not_wm_codes)

cmaps = ['red', 'white', 'cornflowerblue']


for t_type, t_mat, cmap in zip(tissue_types, [gm, wm, csf], cmaps):
    
    use_cmap = colors.ListedColormap([cmap, 'white'])
    
    t_mat_int = t_mat.astype('int')
    tissue_mask = np.ma.masked_where(t_mat_int == 0, t_mat_int)
    
    axrows[1][0].imshow(np.rot90(tissue_mask, 1), cmap=use_cmap, alpha=0.5)
    
# now for the SIENAX segmentations
sienax_data_folder = os.path.join(sienax_folder, mr_id, 'bin', 'axT1_raw_sienax')

tissue_numbers = [1, 2, 0]
tissue_types = ['gm', 'wm', 'csf']
cmaps = ['red', 'white', 'cornflowerblue']

for t_num, cmap in zip(tissue_numbers, cmaps):
    the_file = os.path.join(sienax_data_folder, f'I_stdmaskbrain_pve_{t_num}.nii.gz')
    
    use_cmap = colors.ListedColormap([cmap, cmap])
    
    the_data = nib.load(the_file)
    the_mat = the_data.get_fdata()
    the_slice = the_mat[:,:,spm_slice]
    
    the_slice[the_slice > 1] = 1
    
    slicealphas = the_slice * 0.5
    
    tissue_mask = np.ma.masked_where(the_slice == 0, the_slice)
    
    axrows[1][1].imshow(np.rot90(tissue_mask), cmap=use_cmap, alpha=np.rot90(slicealphas))

    
    
axrows[0][0].set_title('T1-weighted image')
axrows[0][1].set_title('SPM')
axrows[1][0].set_title('FreeSurfer')
axrows[1][1].set_title('SIENAX')


plt.tight_layout()

plt.savefig(outname, dpi=400)