#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:34:40 2020

@author: skyjones
"""

import os
import itertools

import pandas as pd
from glob import glob
import nibabel as nib
import numpy as np

in_csv = '/Users/manusdonahue/Documents/Sky/brain_volume_cleaning/brain_volumes.csv'
out_csv = '/Users/manusdonahue/Documents/Sky/brain_volume_cleaning/brain_volumes_calculated.csv'

# parent folder that house the processed scan folders
filefolder = '/Users/manusdonahue/Documents/Sky/brain_volumes/'


######


study_id_col = 'study_id'
mr_id_cols = ['mr1_mr_id_real', 'mr2_mr_id_real', 'mr3_mr_id_real']

cols_of_interest = [study_id_col]
cols_of_interest.extend(mr_id_cols)

df_orig = pd.read_csv(in_csv, dtype={study_id_col:'object'})
df = df_orig[cols_of_interest].copy()

col_build_mr = ['mr1', 'mr2', 'mr3']
col_build_tissue = ['grey', 'white', 'brain']
col_build_norm = ['norm', 'raw']

new_cols = itertools.product(col_build_mr, col_build_tissue, col_build_norm)

for c in list(new_cols):
    col = '_'.join(c)
    df[col] = None

    
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

nans_fixed = 0

for i, row in df.iterrows():
    study_id = row[study_id_col]
    print(f'\nStudy ID is {study_id}')
    scan_no = 0
    for c, mr_ind in zip(mr_id_cols, col_build_mr):
        scan_no += 1
        mr = row[c]
        if pd.isnull(mr):
            continue
        
        print(f'---On {mr}---')
        # try to find the file we're interested in
        candidate_folder = os.path.join(filefolder, mr, 'bin', 'axT1_raw_sienax')
        report = os.path.join(candidate_folder, 'report.sienax')
        
        print(f'Report at: {report}')

        try:
            with open(report, "r") as rep:
                txt = rep.read()
                lines = txt.split('\n')
                
                greys = lines[-4]
                whites = lines[-3]
                brains = lines[-2]
                
                grey_vol_norm = greys.split(' ')[-2]
                grey_vol_raw = greys.split(' ')[-1]
                
                white_vol_norm = whites.split(' ')[-2]
                white_vol_raw = whites.split(' ')[-1]
                
                brain_vol_norm = brains.split(' ')[-2]
                brain_vol_raw = brains.split(' ')[-1]
                
                df.at[i,f'{mr_ind}_grey_norm'] = grey_vol_norm
                df.at[i,f'{mr_ind}_grey_raw'] = grey_vol_raw
                
                df.at[i,f'{mr_ind}_white_norm'] = white_vol_norm
                df.at[i,f'{mr_ind}_white_raw'] = white_vol_raw
                
                df.at[i,f'{mr_ind}_brain_norm'] = brain_vol_norm
                df.at[i,f'{mr_ind}_brain_raw'] = brain_vol_raw
                
        except IOError:
            print('\n\nNO REPORT AVAILABLE\n\n')
            continue
        
            
df.to_csv(out_csv)
print(f'Finished')
        
        
        
        
                
                
                