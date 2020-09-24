
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:34:40 2020

@author: skyjones
"""

import os
import re

import pandas as pd
from glob import glob
import nibabel as nib
import numpy as np
import shutil


out_csv = '/Users/manusdonahue/Documents/Sky/volume_testing/volume_comparisons.csv'
in_folder = '/Users/manusdonahue/Documents/Sky/volume_testing'


run_fast = True
run_sienax = True
run_freesurfer = True

over_fast = False
over_sienax = False
over_freesurfer = False

######

np.random.seed(0)

subjects_dir = os.environ['SUBJECTS_DIR']

folders = np.array(glob(os.path.join(in_folder, '*/'))) # list of all possible subdirectories
#folders = folders[np.random.choice(len(folders), size=10, replace=False)]


pts_of_interest = ['SCD_TRANSP_P001_01',
                   'SCD_TRANSP_P001_02',
                   'SCD_P004_01',
                   'SCD_P004_02',
                   'SCD_TRANSP_P005_01',
                   'SCD_TRANSP_P005_02',
                   'SCD_P035_01',
                   'SCD_P035_02',
                   'SCD_P009_01',
                   'SCD_P009_02',
                   'SCD_P013_01',
                   'SCD_P013_02']

folders = [os.path.join(in_folder, i) for i in pts_of_interest]


def get_terminal(path):
    """
    Takes a filepath or directory tree and returns the last file or directory
    

    Parameters
    ----------
    path : path
        path in question.

    Returns
    -------
    str of only the final file or directory.

    """
    return os.path.basename(os.path.normpath(path))


out_df = pd.DataFrame()
mrs = []
for i, f in enumerate(folders):
    mr = get_terminal(f)
    mrs.append(mr)
    print(f'{i+1} of {len(folders)}: {mr}')
    
    t1_path = os.path.join(f, 'processed', 'axT1.nii.gz')
    raw_t1_path = os.path.join(f, 'bin', 'axT1_raw.nii.gz')
    
    comp_folder = os.path.join(f, 'comp')
    fast_folder = os.path.join(comp_folder, 'fast')
    sienax_folder = os.path.join(comp_folder, 'sienax')
    freesurfer_folder = os.path.join(comp_folder, 'freesurfer')

    if os.path.exists(comp_folder):
        pass
    else:
        os.mkdir(comp_folder)
    
    all_volumes = {}
    blank_sub = {'wm':None,
                 'gm':None,
                 'csf':None,
                 'icv':None,
                 'det':None,
                 }

    # do FAST segmentation
    if run_fast:
        
        if os.path.exists(fast_folder) and over_fast:
            shutil.rmtree(fast_folder)
            
        if not os.path.exists(fast_folder):
            os.mkdir(fast_folder)
            fast_base = os.path.join(fast_folder, 'fast')
            fast_command = f'fast -S 1 -t 1 -n 3 -o {fast_base} {t1_path}'
            print(f'Running FAST:\n{fast_command}')
            os.system(fast_command)

        fast_pve_path = os.path.join(fast_folder, 'fast_pveseg.nii.gz')
        
        raw = nib.load(fast_pve_path)
        img = raw.get_fdata()
        
        header = raw.header
        voxel_dims = header['pixdim'][1:4]
        voxel_vol = np.product(voxel_dims)
        
        # 1 = csf, 2 = gm, 3 = wm
        # use partial voluems for calculation
        seg_types = {1: 'csf', 2: 'gm', 3:'wm'}
        
        fast_sub = blank_sub.copy()
        
        for num, matter_type in seg_types.items():
            
            subnum = num-1
            subseg_file = os.path.join(fast_folder, f'fast_pve_{subnum}.nii.gz')
            subraw = nib.load(subseg_file)
            subim = subraw.get_fdata()
            
            
            vol = float(subim.sum() * voxel_vol) / 1e3
            
            fast_sub[matter_type] = vol
            
            
        all_volumes['fast'] = fast_sub


    if run_sienax:   
                
        if os.path.exists(sienax_folder) and over_sienax:
            shutil.rmtree(sienax_folder)
            
        if not os.path.exists(sienax_folder):
            sienax_command = f'sienax {t1_path} -o {sienax_folder}'
            print(f'Running SIENAX:\n{sienax_command}')
            os.system(sienax_command)
        
        sienax_report = open(os.path.join(sienax_folder, 'report.sienax'))
        
        txt = sienax_report.read()
        lines = txt.split('\n')
        
        greys = lines[-4]
        whites = lines[-3]
        brains = lines[-2]
        
        grey_vol_norm = float(greys.split(' ')[-2])
        grey_vol_raw = float(greys.split(' ')[-1])
        
        white_vol_norm = float(whites.split(' ')[-2])
        white_vol_raw = float(whites.split(' ')[-1])
        
        brain_vol_norm = float(brains.split(' ')[-2])
        brain_vol_raw = float(brains.split(' ')[-1])
        
        
        sienax_sub = blank_sub.copy()
        sienax_sub['wm'] = white_vol_raw / 1e3
        sienax_sub['gm'] = grey_vol_raw / 1e3
        sienax_sub['det'] = brain_vol_norm / brain_vol_raw
        
        all_volumes['sienax'] = sienax_sub
        
        
    if run_freesurfer:
                        
        if os.path.exists(freesurfer_folder) and over_freesurfer:
            shutil.rmtree(freesurfer_folder)
            
        if not os.path.exists(freesurfer_folder):
            os.mkdir(freesurfer_folder)
        
        r1_cmd = f'/Applications/freesurfer/7.1.1/bin/recon-all -subjid {mr} -i {raw_t1_path} -autorecon1'
        print(f'Running Freesurfer -autorecon1:\n{r1_cmd}')
        os.system(r1_cmd)
        
        
        r2_cmd = f'/Applications/freesurfer/7.1.1/bin/recon-all -subjid {mr} -autorecon2'
        print(f'Running Freesurfer -autorecon2:\n{r2_cmd}')
        os.system(r2_cmd)
        
        
        stats_file = os.path.join(subjects_dir, mr, 'stats', 'aseg.stats')
        
        if not os.path.exists(stats_file):
            r3_cmd = f'/Applications/freesurfer/7.1.1/bin/recon-all -subjid {mr} -autorecon3'
            print(f'Running Freesurfer -autorecon3:\n{r3_cmd}')
            os.system(r3_cmd)
        else:
            print('autorecon3 already run. skipping')
        
        
        stats_report = open(stats_file)

        txt = stats_report.read()
        lines = txt.split('\n')
        
        wm_line = [i for i in lines if 'Total cerebral white matter volume' in i][0]
        gm_line = [i for i in lines if 'Total gray matter volume' in i][0]
        icv_line = [i for i in lines if 'Estimated Total Intracranial Volume' in i][0]
        
        wm_val = float(wm_line.split(', ')[-2])
        gm_val = float(gm_line.split(', ')[-2])
        icv_val = float(icv_line.split(', ')[-2])
        
        trans_mat_file = os.path.join(subjects_dir, mr, 'mri', 'transforms', 'talairach.xfm') 
        trans_report = open(trans_mat_file)
        trans_txt = trans_report.read()
        trans_lines = trans_txt.split('\n')
        
        mat_as_text = trans_lines[-4:-1]
        mat = [[float(a) for a in re.split(';| ', i) if a != ''] for i in mat_as_text]
        mat.append([0, 0, 0, 1])
        mat = np.array(mat)
        
        det = np.linalg.det(mat)
        
        freesurfer_sub = blank_sub.copy()
        freesurfer_sub['wm'] = wm_val / 1e3
        freesurfer_sub['gm'] = gm_val / 1e3
        freesurfer_sub['icv'] = icv_val / 1e3
        freesurfer_sub['det'] = det
        
        all_volumes['freesurfer'] = freesurfer_sub
        
        
    flat_vols = {'pt':mr}
    for key, sub in all_volumes.items():
        for subkey, val in sub.items():
            flat_vols[f'{key}_{subkey}'] = val
            
    out_df = out_df.append(flat_vols, ignore_index=True)
    out_df = out_df[flat_vols.keys()]
    out_df.to_csv(out_csv)
