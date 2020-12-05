
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
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import scipy
import statsmodels.api as sm

from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data

from parse_fs_stats import parse_freesurfer_stats

exclude_pts = []

out_folder = '/Users/manusdonahue/Documents/Sky/freesurfer_volume_visualization_glymph/'
parsed_folder = '/Users/manusdonahue/Documents/Sky/freesurfer_volume_visualization_glymph/parsed'

brain_vol_csv = '/Users/manusdonahue/Documents/Sky/normal_brain_vols.csv' # from Borzage, Equations to describe brain size across the continuum of human lifespan (2012)
# values originally reported as mass in g, converted to cc assuming rho = 1.04 g/cc

fs_folder = '/Volumes/DonahueDataDrive/freesurfer_subjects_glymph/'

parse = True
collate = True
visualize = False 

# os.path.basename(os.path.normpath(path))

###########


def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None, color='#b9cfe7'):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color=color, edgecolor="", alpha=0.25)

    return ax


def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """
    if ax is None:
        ax = plt.gca()

    bootindex = scipy.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        pc = scipy.polyfit(xs, ys + resamp_resid, 1)                  
        # Plot bootstrap cluster
        ax.plot(xs, scipy.polyval(pc, xs), "b-", linewidth=2, alpha=3.0 / float(nboot))

    return ax


def filter_zeroed_axial_slices(nii_data, thresh=0.99):
    # removes slices if the number of pixels that are lesser than or equal to 0 exceeds a % threshold, and replaces NaN with -1
    the_data = nii_data.copy()
    wherenan = np.isnan(the_data)
    the_data[wherenan] = -1
    
    if thresh:
        keep = []
        for i in range(the_data.shape[2]):
            d = the_data[:,:,i]
            
            near_zero = np.isclose(d,0)
            less_zero = (d <= 0)
            
            bad_pixels = np.logical_or(near_zero, less_zero)
            
            perc_bad = bad_pixels.sum() / d.size
            
            if not perc_bad >= thresh:
                keep.append(True)
            else:
                keep.append(False)
        
        new = the_data[:,:,keep]
        return new
    else:
        return the_data


folders = np.array(glob(os.path.join(fs_folder, '*/'))) # list of all possible subdirectories
folders = [os.path.normpath(i) for i in folders]

mr_ids = [os.path.basename(f) for f in folders]


if parse:
    for i, mr in enumerate(mr_ids):
        
        print(f'\nParsing {mr} ({i+1} of {len(mr_ids)})')
        
        stats_file = os.path.join(fs_folder, mr, 'stats', 'aseg.stats')
        parsed_file = os.path.join(parsed_folder, f'{mr}.csv')
        
        try:
            parse_freesurfer_stats(stats_file, parsed_file)
        except FileNotFoundError:
            print(f'No completed Freesurfer folder for {mr} ({stats_file})')
            

parsed_csvs = np.array(glob(os.path.join(parsed_folder, '*.csv'))) # list of all possible subdirectories


if collate:
    out_csv = os.path.join(out_folder, 'collated.csv')
    out_df = pd.DataFrame()
    blank_dict = {'mr_id':None,
                  'wm_vol':None,
                  'gm_vol':None,
                  'total_vol':None,
                  'total_vol_custom':None,
                  'mask_vol_custom':None,
                  'supratent':None,
                  'etiv':None,
                  'csf_vol':None,
                  'gm_normal':None,
                  'wm_normal':None,
                  'total_normal':None,
                  'supratent_normal':None
                  }
    for i, csv in enumerate(parsed_csvs):
        pt_name = os.path.basename(os.path.normpath(csv))[:-4]
        print(f'Collating {pt_name} ({i+1} of {len(parsed_csvs)})')
        
        working = pd.Series(blank_dict.copy())
        parsed_csv = pd.read_csv(csv, index_col='short')
        
        working['mr_id'] = pt_name
        if pt_name in exclude_pts:
            working['exclude'] = 1
        working['gm_vol'] = parsed_csv.loc['TotalGrayVol']['value'] / 1e3
        working['total_vol'] = parsed_csv.loc['BrainSegVolNotVent']['value'] / 1e3
        working['wm_vol'] = working['total_vol'] - working['gm_vol']
        #working['age'] = in_table_indexed.loc[pt_name]['age']
        working['etiv'] = parsed_csv.loc['eTIV']['value'] / 1e3
        working['gm_normal'] = working['gm_vol'] / working['etiv']
        working['wm_normal'] = working['wm_vol'] / working['etiv']
        working['total_normal'] = working['total_vol'] / working['etiv']
        working['supratent'] = parsed_csv.loc['SupraTentorialVolNotVent']['value'] / 1e3
        working['supratent_normal'] = working['supratent'] / working['etiv']
        working['csf_vol'] = working['etiv'] - working['total_vol']
        
        """
        if in_table_indexed.loc[pt_name]['mri1_wml_drp'] == 1:
            working['white_matter_injury'] = 1
        else:
            working['white_matter_injury'] = 0
        """
        
        fs_seg_file = os.path.join(fs_folder, pt_name, 'mri', 'aseg.auto.mgz')
        fs_brain_file = os.path.join(fs_folder, pt_name, 'mri', 'brain.mgz')
        
        seg_data = nib.load(fs_seg_file)
        brain_data = nib.load(fs_brain_file)
        
        seg_voxel_vol = np.product(seg_data.header.get_zooms())
        brain_voxel_vol = np.product(seg_data.header.get_zooms())
        
        seg_mat = seg_data.get_fdata()
        brain_mat = brain_data.get_fdata()
        
        seg_mask = seg_mat > 0
        brain_mask = brain_mat > 0
        
        seg_vol = seg_mask.sum()*seg_voxel_vol
        brain_vol = brain_mask.sum()*brain_voxel_vol
        
        working['total_vol_custom'] = seg_vol / 1e3
        working['mask_vol_custom'] = brain_vol / 1e3
        
        
        for ind in parsed_csv.index:
            try: # if the index is an integer (or can be coerced to one) then the value is for a granular segmentation volume
                num = int(ind)
                keyname = parsed_csv.loc[ind]['long']
                val = parsed_csv.loc[ind]['value']
                if 'hypo' not in keyname:
                   working[keyname] = val / 1e3
                else:
                   working[keyname] = val
                if keyname not in blank_dict:
                    blank_dict[keyname] = None
            except ValueError:
                pass
    
        
                
        out_df = out_df.append(working, ignore_index=True)
    
    out_df = out_df[blank_dict.keys()]
    out_df.to_csv(out_csv, index=False)
        
    
if visualize:
    print('Visualizing')
    pass
        
            
            
            
            
            

        