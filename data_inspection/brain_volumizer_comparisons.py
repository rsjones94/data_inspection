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

over_fast = True
over_sienax = True
over_freesurfer = True

######

np.random.seed(0)

subjects_dir = os.environ['SUBJECTS_DIR']

folders = np.array(glob(os.path.join(in_folder, '*/'))) # list of all possible subdirectories
folders = folders[np.random.choice(len(folders), size=10, replace=False)]
    

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
            vol = int(img.sum() * voxel_vol) # uncomment this line to use partial volume estimates (2 of 2)
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
        sienax_sub['wm'] = white_vol_norm
        sienax_sub['gm'] = grey_vol_norm
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
        

        
        
        
        
print(f'Folders:')
for i in folders:
    print(i)
        
        
