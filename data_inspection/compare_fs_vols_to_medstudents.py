
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:34:40 2020

@author: skyjones
"""

import os
import re
import shutil

import pandas as pd
from glob import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing

from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data

from parse_fs_stats import parse_freesurfer_stats

exclude_pts = ['SCD_K065', 'SCD_TRANSP_P001_01']

old_csv = '/Users/manusdonahue/Documents/Sky/freesurfer_volume_visualization/comp_tissue_vols_medstud.xlsx'
fs_csv = '/Users/manusdonahue/Documents/Sky/freesurfer_volume_visualization/collated.csv'

brain_vol_csv = '/Users/manusdonahue/Documents/Sky/normal_brain_vols.csv' # from Borzage, Equations to describe brain size across the continuum of human lifespan (2012)
# values originally reported as mass in g, converted to cc assuming rho = 1.04 g/cc

fs_folder = '/Volumes/DonahueDataDrive/freesurfer_subjects/'

parsed_folder = '/Users/manusdonahue/Documents/Sky/freesurfer_volume_visualization/parsed/'

out_folder = '/Users/manusdonahue/Documents/Sky/freesurfer_volume_visualization/'

# os.path.basename(os.path.normpath(path))

###########
            



old_voldata = pd.read_excel(old_csv, index_col='subj_id')[['wm_vol', 'gm_vol']] / 1000
fs_voldata = pd.read_csv(fs_csv, index_col='mr_id')[['wm_vol', 'gm_vol']]

vol_types = ['gm_vol', 'wm_vol', 'total_vol']

fs_gm = np.array([])
fs_wm = np.array([])

old_gm = np.array([])
old_wm = np.array([])

for pt in fs_voldata.index:
    
    fsrow = fs_voldata.loc[pt]
    try:
        oldrow = old_voldata.loc[pt]
    except KeyError:
        continue
    
    fs_gm = np.append(fs_gm, fsrow['gm_vol'])
    fs_wm = np.append(fs_wm, fsrow['wm_vol'])
    
    old_gm = np.append(old_gm, oldrow['gm_vol'])
    old_wm = np.append(old_wm, oldrow['wm_vol'])
    
fs_tot = fs_gm + fs_wm
old_tot = old_gm + old_wm

limits = ((300,1000), (300,800), (600,1700))
trulim = (0,1600)
tru_out = os.path.join(out_folder, f'FSvsMedStudVols.png')
plt.figure(figsize=(16,16))
for ser, name, lims in zip([[fs_gm, old_gm], [fs_wm, old_wm], [fs_tot, old_tot]], ['Grey matter', 'White matter', 'Total tissue'], limits):

    plt.scatter(ser[0], ser[1], label=name)
    
    outname = os.path.join(out_folder, f'FSvsMedStudVols_{name}.png')

plt.plot([trulim[0],trulim[1]], [trulim[0],trulim[1]], color='black')

plt.xlim(trulim)
plt.ylim(trulim)
plt.title('Comparison of Freesurfer and old segmentation volumes')
plt.xlabel('Freesurfer volume (cc)')
plt.ylabel('Mystery med student volume (cc)')

plt.legend()



plt.axes().set_aspect('equal')
plt.tight_layout()
plt.savefig(tru_out)