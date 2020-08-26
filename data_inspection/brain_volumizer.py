#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:34:40 2020

@author: skyjones
"""

import os

import pandas as pd


in_csv = '/Users/manusdonahue/Documents/Sky/brain_volume_cleaning/brain_volumes.csv'
out_csv = '/Users/manusdonahue/Documents/Sky/brain_volume_cleaning/brain_volumes_calculated.csv'

seg_ext = '_3D_brain_seg.nii.gz' # scan id + seg_ext = the segmentation file
in_folder = 'Processed' # subfolder the segmentation file is in

# parent folder that house the scan folders
filefolder = '/Volumes/DonahueDataDrive/Data_sort/SCD_Grouped'


######


study_id_col = 'study_id'
mr_id_cols = ['mr1_mr_id', 'mr2_mr_id', 'mr3_mr_id']

cols_of_interest = [study_id_col]
cols_of_interest.extend(mr_id_cols)

df = pd.read_csv(in_csv, dtype={study_id_col:'object'})
df = df[cols_of_interest]

    
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


all_subdirectories = [x[0] for x in os.walk(filefolder)] # list of all possible subdirectories

for i, row in df.iterrows():
    study_id = row[study_id_col]
    print(f'\nStudy ID is {study_id}')
    for c in mr_id_cols:
        mr = row[c]
        if pd.isnull(mr):
            continue
        
        # try to find the folder we're interested in
        candidate_folders = [sub for sub in all_subdirectories if get_terminal(sub) == mr] # check if last subfolder is scan name
        n_cands = len(candidate_folders)
        
        if n_cands != 1:
            print(f'{mr} has {n_cands} candidate folders. Skipping')
            continue
        else:
            cand = candidate_folders[0]
        
        # try to find the segmentation file
        seg_file = os.path.join(cand, in_folder, f'{mr}{seg_ext}')
        
        if not os.path.exists(seg_file):
            print(f'{mr} has no seg file ({seg_file}). Skipping')
        else:
            print(f'{mr} is g2g')