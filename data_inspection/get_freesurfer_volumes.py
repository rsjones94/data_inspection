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


out_csv = '/Users/manusdonahue/Documents/Sky/volume_testing/freesurfer_volumes.csv'

######


subjects_dir = os.environ['SUBJECTS_DIR']

folders = np.array(glob(os.path.join(subjects_dir, '*/'))) # list of all possible subdirectories
    

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

for i, f in enumerate(folders):
    mr = get_terminal(f)
    print(f'{i+1} of {len(folders)}: {mr}')
    
    all_volumes = {}
    blank_sub = {'wm':None,
                 'gm':None,
                 'csf':None,
                 'icv':None,
                 'det':None,
                 }

    
    
    stats_file = os.path.join(f, 'stats', 'aseg.stats')
    
    if not os.path.exists(stats_file):
        print(f'{mr} is incomplete. skipping')
        continue
    
    
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
        
