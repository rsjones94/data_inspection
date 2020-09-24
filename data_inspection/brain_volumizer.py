#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:34:40 2020

@author: skyjones
"""

import os

import pandas as pd
from glob import glob
import nibabel as nib
import numpy as np

in_csv = '/Users/manusdonahue/Documents/Sky/brain_volume_cleaning/brain_volumes.csv'
out_csv = '/Users/manusdonahue/Documents/Sky/brain_volume_cleaning/brain_volumes_calculated.csv'

seg_ext = 'pveseg.nii.gz' # scan id + seg_ext = the segmentation file
in_folder = 'FAST' # subfolder the segmentation file is in

# parent folder that house the scan folders
filefolder = '/Users/manusdonahue/Documents/Sky/brain_volumes/'


######

np.random.seed(0)


study_id_col = 'study_id'
mr_id_cols = ['mr1_mr_id_real', 'mr2_mr_id_real', 'mr3_mr_id_real']

cols_of_interest = [study_id_col]
cols_of_interest.extend(mr_id_cols)

df_orig = pd.read_csv(in_csv, dtype={study_id_col:'object'})
df = df_orig[cols_of_interest].copy()

    
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

triers = [0,-1,1] # we will alter the scan number by these values in sequence to try to find the scan

nans_fixed = 0

for i, row in df.iterrows():
    study_id = row[study_id_col]
    print(f'\nStudy ID is {study_id}')
    scan_no = 0
    for c in mr_id_cols:
        scan_no += 1
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
        
        """
        try_index = 0
        found_file = False
        
        scan_number = int(mr[-1])
        
        while not found_file:
            # try to find the segmentation file
            
            try:
                adder = triers[try_index]
            except IndexError:
                print(f'No segmentation found for {mr}')
                break
            end_number = str(scan_number + adder)
            
            seg_file = os.path.join(cand, in_folder, f'{mr[:-1]+end_number}{seg_ext}')
            
            if not os.path.exists(seg_file):
                try_index += 1
            else:
                found_file = True
                print(f'{mr} is g2g')
        """
        
        globular = os.path.join(cand,'**', f'*{seg_ext}')
        seg_files = glob(globular, recursive=True)
        
        if len(seg_files) != 1:
            print(f'{len(seg_files)} found for segmentations for {mr}. Skipping!!!\n({seg_files})')
            print(globular)
            continue
        else:
            seg_file = seg_files[0]
        
        
        print(f'{mr} is g2g')
        
        raw = nib.load(seg_file)
        img = raw.get_fdata()
        
        header = raw.header
        voxel_dims = header['pixdim'][1:4]
        voxel_vol = np.product(voxel_dims)
        
        # 1 = csf, 2 = gm, 3 = wm
        
        
        # use partial voluems for calculation
        seg_types = {1: 'csf', 2: 'grey', 3:'white'}
        
        total_vol = int(img[img > 0].sum() * voxel_vol)
        print(f'Total vol for {mr}: {total_vol}')
        
        imaging_vol = int(np.product(img.size) * voxel_vol)
        print(f'Imaging vol: {imaging_vol}')
        
        fixed_a_nan = False 
        for num, matter_type in seg_types.items():
            
            col = f'mr{scan_no}_{matter_type}_cbv'
            orig = df_orig.iloc[i][col]
            
            if not pd.isnull(orig):
                continue
            
            subnum = num-1
            subseg_file = glob(os.path.join(cand,'**', f'*pve_{subnum}.nii.gz'), recursive=True)[0] # uncomment this line to use partial volume estimates (1 of 2)
            # subseg_file = glob(os.path.join(cand,'**', f'*pveseg.nii.gz'), recursive=True)[0] # uncomment this line to use discrete volume estimates (1 of 2)
            subraw = nib.load(subseg_file)
            subim = subraw.get_fdata()
            
            #vol = int((img==num).sum() * voxel_vol)
            vol = int(subim.sum() * voxel_vol) # uncomment this line to use partial volume estimates (2 of 2)
            #vol = int((subim==num).sum() * voxel_vol) # uncomment this line to use discrete volume estimates (2 of 2)
            
            if orig != vol:
                print(f'{mr} has a new value for {matter_type} ({orig} --> {vol})')
            else:
                print(f'{mr} value remains')
            
            if pd.isnull(vol):
                print(f'calculated NAN for {mr}-{matter_type}')
            elif pd.isnull(orig):
                fixed_a_nan = True
            
            df_orig.at[i,col] = vol
        
        nans_fixed += fixed_a_nan
            
df_orig.to_csv(out_csv)
print(f'Finished. Found data for {nans_fixed} new segmentations')
        
        
        
        
                
                
                