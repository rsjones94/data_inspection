
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:34:40 2020

@author: skyjones
"""

import os
import re
import shutil
import sys

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
from scipy.stats import chisquare, ttest_ind
import statsmodels.api as sm
import redcap
import imageio
import itertools
from skimage import measure as meas
from pingouin import ancova

from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data
import matplotlib
import matplotlib.patheffects as pe

from fpdf import FPDF

from parse_fs_stats import parse_freesurfer_stats
from parse_sienax_stats import parse_sienax_stats



manual_excls = {
            
                'K001': ['stroke'],
                'K011': ['motion'],
                'K017': ['motion'],
                'SCD_C018': ['structural'],
                'SCD_C022': ['structural'],
                'SCD_C023': ['structural'],
                'SCD_K003': ['motion'],
                'SCD_K021': ['motion'],
                'SCD_K029': ['motion', 'bad_sienax'],
                'SCD_K031': ['bad_capture'],
                'SCD_K037': ['motion', 'bad_sienax'],
                'SCD_K050': ['motion', 'bad_sienax'],
                'SCD_K061': ['structural'],
                'SCD_K064_01': ['motion'],
                'SCD_P029': ['motion'],
                'SCD_TRANSF_012_01-SCD_TRANSP_P002_01': ['stroke'],
                'SCD_TRANSF_A001_01':['non_scd_anemia'],
                'SCD_TRANSF_A002_01':['non_scd_anemia'],
                'K004': ['control_sci'],
                'KA001_01': ['non_scd_anemia'],
                'SCD_C005_01': ['control_sci'],
                'SCD_C031': ['control_sci'],
                'SCD_C047': ['control_sci'],
                'SCD_K023_02': ['subsequent'],
                'SCD_K024_02': ['subsequent'],
                'SCD_P024_01': ['stroke'],
                'SCD_P027': ['stroke'],
                'SCD_P025': ['missing_flair'],
                'SCD_P053': ['missing_flair'],
                'SCD_K041': ['motion'], # motion in FLAIR
    
    }

"""
Participants who I manually converted PARREC to NiFTi

    SCD_C011_01
    SCD_K065
    SCD_P008
    SCD_P009_01
    SCD_P010_01
    SCD_P021
    SCD_P012
    SCD_P014
    SCD_P019
    SCD_TRANSF_P006_01
    SCD_P008
    SCD_P009_01
    SCD_P014
    SCD_P019

"""

"""
Participants with custom SIENAX params:
    
    SCD_K004 : -f 0.3
    SCD_K020 : flipped and run with -f 0.3 -B
    SCD_K024 : -f 0.3
    SCD_K025 : -f 0.25
    SCD_K027 : -f 0.3
    SCD_K034 : flipped and run with -f 0.3 -B
    SCD_K035 : flipped and run with -f 0.3 -B
    SCD_K036 : -f 0.1
    SCD_K039 : -f 0.1 -g 0.5 marginal
    SCD_K040 : -f 0.35 -B
    SCD_K041 : -f 0.23 -g 0.13 -R marginial
    SCD_K043 : flipped and run with -f 0.15
    SCD_K048 : flipped and run with -f 0.15 -R
    SCD_K050 : flipped and run with -f 0.4 -B
    SCD_K051 : flipped and run with -f 0.3 -B
    SCD_K052_01 : flipped and run with -f 0.3 -B
    SCD_K054_01 : flipped and run with -f 0.3 -B
    SCD_P014 : -f 0.2 -B
    
"""

"""
Participants with rotated images

SCD_K001
SCD_K020 flipped
SCD_K024
SCD_K027
SCD_K029
SCD_K030
SCD_K034 flipped
SCD_K035 flipped
SCD_K036
SCD_K037
SCD_K039
SCD_K040
SCD_K041
SCD_K042
SCD_K043 flipped
SCD_K046
SCD_K048 flipped
SCD_K050 flipped
SCD_K051 flipped
SCD_K052_01 flipped
SCD_K054_01 flipped
SCD_TRANSP_K001_01

"""


in_csv = '/Users/manusdonahue/Documents/Sky/stroke_status.csv'

out_folder = '/Users/manusdonahue/Documents/Sky/t1_volumizers/'
out_folder_orig = out_folder

lesion_mask_folder = '/Users/manusdonahue/Documents/Sky/brain_lesion_masks/combined/'

brain_vol_csv = '/Users/manusdonahue/Documents/Sky/normal_brain_vols.csv' # from Borzage, Equations to describe brain size across the continuum of human lifespan (2012)
# values originally reported as mass in g, converted to cc assuming rho = 1.04 g/cc

sienax_folder = '/Users/manusdonahue/Documents/Sky/sienax_segmentations/'
fs_folder = '/Volumes/DonahueDataDrive/freesurfer_subjects_scd/'
spm_folder = '/Users/manusdonahue/Documents/Sky/scd_t1s/'

parse = True
collate = True
quality_check = True
visualize = True
interrater = True
graphs_w_overt = True

# os.path.basename(os.path.normpath(path))

###########

programs = ['SPM', 'FS', 'SIENAX']
norm_columns = ['icv', 'icv', 'vscaling']
sub_outs = [os.path.join(out_folder, f'vis_{f}') for f in programs]
quality_folders = [os.path.join(f, 'quality') for f in sub_outs]
parsed_folders = [os.path.join(f, 'parsed') for f in sub_outs]
program_masters = [spm_folder, fs_folder, sienax_folder]

for big in [sub_outs, quality_folders, parsed_folders]:
    for l in big:
        if not os.path.exists(l):
            os.mkdir(l)

"""
quality_folder =  '/Users/manusdonahue/Documents/Sky/spm_volume_visualization/quality'
parsed_folder = '/Users/manusdonahue/Documents/Sky/spm_volume_visualization/parsed'

"""

exclude_pts = list(manual_excls.keys())

def adjust_for_perfusion(volume, cbf, coef=0.8, exp=0.5, tissue_density=1.041):
    """
    use Grubb's relationship to adjust a tissue volume
    that accounts for CBV (calulation from CBF)

    Parameters
    ----------
    volume : float
        The original volume of tissue.
    cbf : float
        the cererebral blood flow in ml/100g/min.
    coef : float, optional
        Grubb's coefficient. The default is 0.8.
    exp : float, optional
        Grubb's exponent. The default is 0.5.
    tissue_density : float, optional
        The tissue density, in g/cc
        https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1600-0404.1970.tb05606.x#:~:text=Data%20obtained%20in%20this%20investigation,grams%20per%20cubic%20centimeter%20(S.E.

    Returns
    -------
    The tissue volume, adjusted to account for the fact that some of the tissue 
    was actually blood.

    """
    cbv = coef * cbf ** exp # cbv in ml/100g tissue
    blood_frac = (cbv * tissue_density)/100  # ml / 100cc (dimensionless)
    adjusted = volume * (1-blood_frac)
    return adjusted
    
    

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
            
            bad_pixels = np.logical_or(near_zero, less_zero)> 5
            
            perc_bad = bad_pixels.sum() / d.size
            
            if not perc_bad >= thresh:
                keep.append(True)
            else:
                keep.append(False)
        
        new = the_data[:,:,keep]
        return new
    else:
        return the_data

in_table_indexed = pd.read_csv(in_csv, index_col='mr1_mr_id_real')

if parse:
    
    for program, parent_folder, parsed_folder, master in zip(programs, sub_outs, parsed_folders, program_masters):
        print(f'------------------------------ Parsing ({program}) ------------------------------')
        if program == 'SPM':
            
            files = np.array(glob(os.path.join(master, '*.nii'))) # list of all niftis
            parent_files = [f for f in files if os.path.basename(os.path.normpath(f))[0] != 'c'] # if the nifti starts with c it's a tissue probability map 
            for i, p in enumerate(parent_files):
                
                mr = os.path.basename(os.path.normpath(p))[:-4]
                
                print(f'\nParsing {mr} ({i+1} of {len(parent_files)})')
                
                parsed_file = os.path.join(parsed_folder, f'{mr}.csv')
                
                # c1 is gm, c2 is wm, c3 is csf
                tissue_types = ['gm', 'wm', 'csf']
                tissue_long = ['gray_matter_volume', 'white_matter_volume', 'cerebrospinal_fluid_volume']
                tissue_numbers = [1, 2, 3]
                
                blank = {'short':None, 'long':None, 'value':None, 'units':'mm^3'}
                df = pd.DataFrame()
                
                for t_type, t_num, long in zip(tissue_types, tissue_numbers, tissue_long):
                    the_row = blank.copy()
                    the_row['short'] = t_type
                    the_row['long'] = long
                    
                    tissue_file = os.path.join(spm_folder, f'c{t_num}{mr}.nii')
                    
                    seg_data = nib.load(tissue_file)
                    seg_voxel_vol = np.product(seg_data.header.get_zooms())
                    seg_mat = seg_data.get_fdata()
                    
                    raw_vol = seg_mat.sum()
                    vol = raw_vol * seg_voxel_vol
                    
                    the_row['value'] = vol
                    
                    df = df.append(the_row, ignore_index=True)
                    
                df = df[['short', 'long', 'value', 'units']]
                df.to_csv(parsed_file, index=False)
                
        elif program == 'SIENAX':
            
            folders = np.array(glob(os.path.join(master, '*/'))) # list of all folders
            for i, f in enumerate(folders):
                
                mr = os.path.basename(os.path.normpath(f))
                print(f'\nParsing {mr} ({i+1} of {len(folders)})')
                
                sienax_report = os.path.join(f, 'bin/axT1_raw_sienax/report.sienax')
                parsed_file = os.path.join(parsed_folder, f'{mr}.csv')
                
                try:
                    parse_sienax_stats(sienax_report, parsed_file)
                except FileNotFoundError:
                    print(f'No completed SIENAX report for {mr} ({sienax_report})')
                
                
        
        elif program == 'FS':
            
            folders = np.array(glob(os.path.join(master, '*/'))) # list of all folders
            for i, f in enumerate(folders):
                
                mr = os.path.basename(os.path.normpath(f))
                
                print(f'\nParsing {mr} ({i+1} of {len(folders)})')
                
                stats_file = os.path.join(fs_folder, mr, 'stats', 'aseg.stats')
                parsed_file = os.path.join(parsed_folder, f'{mr}.csv')
                
                try:
                    parse_freesurfer_stats(stats_file, parsed_file)
                except FileNotFoundError:
                    print(f'No completed Freesurfer folder for {mr} ({stats_file})')
            

parsed_csv_list = [np.array(glob(os.path.join(f, '*.csv'))) for f in parsed_folders]

if collate:
    
    print('Contacting REDCap')
    api_url = 'https://redcap.vanderbilt.edu/api/'
    token_loc = '/Users/manusdonahue/Desktop/Projects/redcaptoken_scd_real.txt'
    token = open(token_loc).read()
    
    project = redcap.Project(api_url, token)
    project_data_raw = project.export_records()
    project_data = pd.DataFrame(project_data_raw)
    print('Contacted...')
    
    mri_cols = ['mr1_mr_id',
                'alternate_mr_id_1'
                ]
    


    blank_dict = {'mr_id':None,
                  'wm_vol':None,
                  'gm_vol':None,
                  'total_vol':None,
                  'wm_vol_unadj':None,
                  'gm_vol_unadj':None,
                  'total_vol_unadj':None,
                  'icv':None,
                  'csf_vol':None,
                  'gm_normal':None,
                  'wm_normal':None,
                  'total_normal':None,
                  'vscaling':None,
                  'hct':None,
                  'gm_cbf':None,
                  'wm_cbf':None,
                  'ox_delivery':None,
                  'age':None,
                  'stroke_silent':None,
                  'stroke_overt':None,
                  'sci':None,
                  'transf':None,
                  'race':None,
                  'scd':None,
                  'anemia':None,
                  'control':None,
                  'lesion_burden':None,
                  'lesion_count':None,
                  'gender':None,
                  'intracranial_stenosis':None,
                  'hydroxyurea':None,
                  'hemoglobin':None,
                  'hemoglobin_s_frac':None,
                  'pulseox':None,
                  'bmi':None,
                  'diabetes':None,
                  'high_cholesterol':None,
                  'coronary_art_disease':None,
                  'smoker':None,
                  'exclude':0,
                  'excl_control_sci':None,
                  'excl_subsequent':None,
                  'excl_stroke':None,
                  'excl_excessive_burden':None,
                  'excl_transf':None,
                  'excl_transp':None,
                  'excl_bad_freesurfer':None,
                  'excl_bad_sienax':None,
                  'excl_bad_spm':None,
                  'excl_bad_anyseg':None,
                  'excl_missing_gm_cbf':None,
                  'excl_missing_wm_cbf':None}
    
    manual_exclusion_reasons = []
    for key,val in manual_excls.items():
        manual_exclusion_reasons.extend(val)
    manual_exclusion_reasons = set(manual_exclusion_reasons)
    manual_exclusion_addons = [f'excl_{i}' for i in manual_exclusion_reasons]
    
    for i in manual_exclusion_addons:
        if i not in blank_dict:
            blank_dict[i] = None
    
    """
        age
        sex
        race
        infarcted - outcome_mri1_sci
        intracranial stenosis > 50% - mra1_ic_stenosis_drp
        hydroxyurea therapy - current_med_hu
        chronic blood transfusions - reg_transf
        hemoglobin (g/dL) - initial_hgb_s
        bmi - bmi
        diabetes mellitus - mh_rf_diab
        coronary artery disease - mh_rf_cad
        high cholesterol - mh_rf_high_cholest
        smoking currently - mh_rf_act_smoke
    """
    
    for parsed_csvs, out_folder, prog in zip(parsed_csv_list, sub_outs, programs):
        
        missing_masks = []
        out_df = pd.DataFrame()
        print(f'Program is {prog}')
        for i, csv in enumerate(parsed_csvs):
            pt_name = os.path.basename(os.path.normpath(csv))[:-4]
            #print(f'Collating {pt_name} ({i+1} of {len(parsed_csvs)})')
            
            working = pd.Series(blank_dict.copy())
            parsed_csv = pd.read_csv(csv, index_col='short')
            
            # get index in database
            which_scan = [pt_name in list(project_data[i]) for i in mri_cols]
            
            if True not in which_scan:
                print(f'No name match found for {pt_name} ({prog})\n')
                continue
            
            scan_index = which_scan.index(True)
            scan_mr_col = mri_cols[scan_index]
            studyid_index_data = project_data.set_index('study_id')
            
            inds = studyid_index_data[scan_mr_col] == pt_name
            cands = studyid_index_data[inds]
            
            study_id = cands.index[0]
            
            
            hematocrit = float(cands.iloc[0][f'blood_draw_hct1'])/100
            working['hct'] = hematocrit
            
            working['mr_id'] = pt_name
            if pt_name in manual_excls:
                
                working['exclude'] = 1
                val = manual_excls[pt_name]
                for v in val:
                    working[f'excl_{v}'] = 1
                        
            if any([working['excl_bad_freesurfer'],working['excl_bad_spm'],working['excl_bad_sienax']]):
                working['excl_bad_anyseg']
                
            for ix in '23456':
                if f'_0{ix}' in pt_name:
                    working['exclude'] = 1
                    working['excl_subsequent'] = 1
            
            working['ox_delivery'] = float(cands.iloc[0][f'mr1_cao2'])
            
            """
            if in_table_indexed.loc[pt_name]['mri1_wml_drp'] == 1:
                working['white_matter_injury'] = 1
            else:
                working['white_matter_injury'] = 0
            """
            
            stroke_overt = (cands.iloc[0][f'outcome_mri1_overt_stroke'])
            #stroke_silent = (cands.iloc[0][f'mh_rf_prior_stroke_silent'])
            stroke_silent = 0
            
            if stroke_overt == 1 or stroke_overt == '1':
                working['exclude'] = 1
                working['excl_stroke'] = 1
            
            sci = (cands.iloc[0][f'outcome_mri1_sci'])
            #transf = (cands.iloc[0][f'enroll_sca_transfusion'])
            
            for val, name in zip([stroke_overt, stroke_silent, sci],
                                 ['stroke_overt', 'stroke_silent', 'sci']):
                if val == 1 or val == '1':
                    working[name] = 1
                else:
                    working[name] = 0
                    
            status = int(cands.iloc[0][f'case_control'])
            
            if status == 2:
                working['scd'] = 0
                working['anemia'] = 1
                working['control'] = 0
            elif status == 1:
                working['scd'] = 1
                working['anemia'] = 0
                working['control'] = 0
            else:
                working['scd'] = 0
                working['anemia'] = 0
                working['control'] = 1
                
            if working['scd'] == 0 and working['sci'] == 1 and 'a0' not in pt_name.lower():
                working['exclude'] = 1
                working['excl_control_sci'] = 1
                
            working['gender'] = int(cands.iloc[0][f'gender'])
            working['age'] =  float(cands.iloc[0][f'age'])
            working['race'] =  int(cands.iloc[0][f'race'])
            
            # additional 
            
            try:
                working['intracranial_stenosis'] = int(cands.iloc[0]['mra1_ic_stenosis_drp'])
            except ValueError:
                pass
            
            try:
                working['hydroxyurea'] = int(cands.iloc[0]['current_med_hu'])
            except ValueError:
                pass
            
            try:
                working['transf'] = int(cands.iloc[0]['reg_transf'])
            except ValueError:
                pass
            
            try:
                working['hemoglobin_s_frac'] = float(cands.iloc[0]['initial_hgb_s_value'])/100
            except ValueError:
                pass
        
            try:
                working['hemoglobin'] = float(cands.iloc[0]['blood_draw_hgb1'])
            except ValueError:
                pass
            
            try:
                working['bmi'] = float(cands.iloc[0]['bmi'])
            except ValueError:
                pass
            
            try:
                working['pulseox'] = float(cands.iloc[0]['mr1_pulse_ox_result'])
            except ValueError:
                pass
            
            try:
                working['diabetes'] = int(cands.iloc[0]['mh_rf_diab'])
            except ValueError:
                pass
            
            try:
                working['high_cholesterol'] = int(cands.iloc[0]['mh_rf_high_cholest'])
            except ValueError:
                pass
            
            try:
                working['coronary_art_disease'] = int(cands.iloc[0]['mh_rf_cad'])
            except ValueError:
                pass
            
            try:
                working['smoker'] = int(cands.iloc[0]['mh_rf_act_smoke'])
            except ValueError:
                pass            
            

            
            
            if 'transp' in pt_name.lower():
                working['exclude'] = 1
                working['excl_transp'] = 1
            
            """
            
                          'intracranial_stenosis':None,
                          'hydroxyurea':None,
                          'transf':None,
                          'hemoglobin':None,
                          'bmi':None,
                          'diabetes':None,
                          'high_cholesterol':None,
                          'coronary_art_disease':None,
                          'smoker':None,
                          'exclude':0}
            
                age
                sex
                race
                infarcted - outcome_mri1_sci
                intracranial stenosis > 50% - mra1_ic_stenosis_drp
                hydroxyurea therapy - current_med_hu
                chronic blood transfusions - reg_transf
                hemoglobin (g/dL) - initial_hgb_s_value
                bmi - bmi
                diabetes mellitus - mh_rf_diab
                coronary artery disease - mh_rf_cad
                high cholesterol - mh_rf_high_cholest
                smoking currently - mh_rf_act_smoke
            """
            
            #if working['age'] >= 18:
            #   working['exclude'] = 1
            
            if working['transf'] == 1 or 'transf' in pt_name.lower():
               working['exclude'] = 1
               working['excl_transf'] = 1
               working['transf'] == 1
            
            # software specific:
            if prog == 'SPM':
                working['gm_vol_unadj'] = parsed_csv.loc['gm']['value'] / 1e3
                working['wm_vol_unadj'] = parsed_csv.loc['wm']['value'] / 1e3
                working['total_vol_unadj'] = working['gm_vol_unadj'] + working['wm_vol_unadj']
                
                working['csf_vol'] = parsed_csv.loc['csf']['value'] / 1e3
                working['icv'] = working['gm_vol_unadj'] + working['wm_vol_unadj'] + working['csf_vol']
                working['gm_normal'] = working['gm_vol_unadj'] / working['icv']
                working['wm_normal'] = working['wm_vol_unadj'] / working['icv']
                working['total_normal'] = working['total_vol_unadj'] / working['icv']
                
            elif prog == 'SIENAX':
                working['gm_vol_unadj'] = parsed_csv.loc['gm']['value'] / 1e3
                working['wm_vol_unadj'] = parsed_csv.loc['wm']['value'] / 1e3
                working['total_vol_unadj'] = working['gm_vol_unadj'] + working['wm_vol_unadj']
                
                working['vscaling'] = parsed_csv.loc['scaling']['value']
                
            elif prog == 'FS':
                working['gm_vol_unadj'] = parsed_csv.loc['TotalGrayVol']['value'] / 1e3
                working['total_vol_unadj'] = parsed_csv.loc['BrainSegVolNotVent']['value'] / 1e3
                working['wm_vol_unadj'] = working['total_vol_unadj'] - working['gm_vol_unadj']
                working['icv'] = parsed_csv.loc['eTIV']['value'] / 1e3
                working['gm_normal'] = working['gm_vol_unadj'] / working['icv']
                working['wm_normal'] = working['wm_vol_unadj'] / working['icv']
                working['total_normal'] = working['total_vol_unadj'] / working['icv']
                working['csf_vol'] = working['icv'] - working['total_vol_unadj']
            
            else:
                raise Exception('you probably spelled something wrong')
                
            try:
                working['gm_cbf'] = float(cands.iloc[0]['mr1_recalc_gm_cbf'])
                working['gm_vol'] = adjust_for_perfusion(working['gm_vol_unadj'], working['gm_cbf'])
            except ValueError:
                working['exclude'] = 1
                working['excl_missing_gm_cbf'] = 1
               
            try:
                working['wm_cbf'] = float(cands.iloc[0]['mr1_recalc_wm_cbf'])
                working['wm_vol'] = adjust_for_perfusion(working['wm_vol_unadj'], working['wm_cbf'])
            except ValueError:
                working['exclude'] = 1
                working['excl_missing_wm_cbf'] = 1
                
            if working['wm_vol'] and working['gm_vol']:
                working['total_vol'] = working['wm_vol'] + working['gm_vol']
            
            
            # calculate lesion burden
            if any([working['stroke_overt'], working['stroke_silent'], working['sci']]):
                burden_mask = os.path.join(lesion_mask_folder, pt_name, 'axFLAIR_mask.nii.gz')
                try:
                    lesion_data = nib.load(burden_mask)
                    lesion_voxel_vol = np.product(lesion_data.header.get_zooms())
                    lesion_mat = lesion_data.get_fdata()
                    
                    raw_vol = lesion_mat.sum()
                    vol = raw_vol * lesion_voxel_vol
                    
                    working['lesion_burden'] = vol / 1e3
                    
                    
                    labeled = meas.label(lesion_mat)
                    working['lesion_count'] = labeled.max()
                    
                    
                    #if working['lesion_burden'] > 3:
                    #    working['exclude'] = 1
                    #    working['excl_excessive_burden'] = 1
                    
                except FileNotFoundError:
                    if working['exclude'] == 0:
                        print(f'please make a mask for {pt_name}')
                        missing_masks.append(pt_name)
            else:
                working['lesion_burden'] = 0
                working['lesion_count'] = 0
                
                
            out_df = out_df.append(working, ignore_index=True)
        
        out_df = out_df[blank_dict.keys()]
        out_csv = os.path.join(out_folder, f'collated.csv')
        out_df.to_csv(out_csv, index=False)
        
        # now make the demographic table
        
        the_cols = ['age', 'race', 'gender', 'sci', 'transf', 'intracranial_stenosis', 'hydroxyurea',
                    'hemoglobin', 'bmi', 'diabetes', 'high_cholesterol', 'coronary_art_disease', 'smoker', 'ox_delivery', 'hemoglobin_s_frac', 'pulseox']
        all_cols = the_cols.copy()
        all_cols.append('scd')
        all_cols.append('exclude')
        cut_df = out_df[all_cols]
        cut_df = cut_df[cut_df['exclude'] == 0]
        
        for col in the_cols:
            if col == 'hemoglobin':
                continue
            ser = cut_df[col]
            for i, val in ser.iteritems():
                if val == '' or val == None or np.isnan(val):
                    ser[i] = 0
                    
        scd_df = cut_df[cut_df['scd']==1]
        ctrl_df = cut_df[cut_df['scd']==0]
        
        categorical = ['race', 'gender', 'sci', 'transf', 'intracranial_stenosis', 'hydroxyurea', 'diabetes',
                       'high_cholesterol', 'coronary_art_disease', 'smoker']
        categorical_names = ['Black race', 'Male sex', 'Has SCI', 'Regular blood transfusions', 'Intracranial stenosis >50%',
                             'Hydroxyurea therapy', 'Diabetes mellitus', 'High cholesterol',
                             'Coronary artery disease', 'Smoking currently']
        
        cont = ['age', 'ox_delivery', 'hemoglobin', 'hemoglobin_s_frac', 'pulseox', 'bmi']
        cont_names = ['Age at MRI', 'CaO2 (mL/dL)', 'Hemoglobin, g/dL', 'Hemoglobin S fraction', 'Pulse oximeter reading', 'Body mass index, kg/m2']
        
        table_1 = pd.DataFrame()
        
        for col,name in zip(categorical, categorical_names):
            
            scd_ser = scd_df[col].dropna()
            ctrl_ser = ctrl_df[col].dropna()
            
            match_num = 1
            if col == 'race':
                match_num = 2
            
            scd_d = sum(scd_ser == match_num)
            ctrl_d = sum(ctrl_ser == match_num)
            
            freq_true = scd_d
            freq_false = len(scd_ser) - freq_true
            
            ctrl_true = ctrl_d
            ctrl_false = len(ctrl_ser) - ctrl_true
            true_frac = ctrl_true / (len(ctrl_ser))
            false_frac = ctrl_false / (len(ctrl_ser))
            
            expect_true = len(scd_ser) * true_frac
            expect_false = len(scd_ser) * false_frac
            
            scd_perc = round((scd_d / len(scd_ser) * 100), 2)
            ctrl_perc = round((ctrl_d / len(ctrl_ser) * 100), 2)
            
            chi, pval = chisquare([freq_true, freq_false], [expect_true, expect_false])
            
            dic = {f'SCD (n={len(scd_df)})': f'{scd_d} ({scd_perc}%)', f'Control (n={len(ctrl_df)})': f'{ctrl_d} ({ctrl_perc}%)', 'p-value':pval}
            ser = pd.Series(dic, name=name)
            table_1 = table_1.append(ser)
            
        for col,name in zip(cont, cont_names):
            
            scd_ser = scd_df[col].dropna()
            ctrl_ser = ctrl_df[col].dropna()
            
            scd_d = np.mean(scd_ser)
            ctrl_d = np.mean(ctrl_ser)
            
            t, pval = ttest_ind(scd_ser, ctrl_ser)
            
            scd_sd = round(np.std(scd_ser),2)
            ctrl_sd = round(np.std(ctrl_ser),2)
            
            dic = {f'SCD (n={len(scd_df)})': f'{round(scd_d,2)} (sd={scd_sd})', f'Control (n={len(ctrl_df)})': f'{round(ctrl_d,2)} (sd={ctrl_sd})', 'p-value':pval}
            ser = pd.Series(dic, name=name)
            table_1 = table_1.append(ser)
            
        table_1_name = os.path.join(out_folder, f'table1.csv')
        table_1.to_csv(table_1_name)
            
            
            
        #sys.exit()
            
            
        
        
        """
            scd vs control
            age
            sex
            race
            infarcted - outcome_mri1_sci
            intracranial stenosis > 50% - mra1_ic_stenosis_drp
            hydroxyurea therapy - current_med_hu
            chronic blood transfusions - reg_transf
            hemoglobin (g/dL) - initial_hgb_s
            bmi - bmi
            diabetes mellitus - mh_rf_diab
            coronary artery disease - mh_rf_cad
            high cholesterol - mh_rf_high_cholest
            smoking currently - mh_rf_act_smoke
        """
    
if quality_check:
    plt.style.use('dark_background')
    composite_quality_folder = os.path.join(out_folder, 'composite_quality')
    if not os.path.exists(composite_quality_folder):
        os.mkdir(composite_quality_folder)
        
    excluded_folder = os.path.join(out_folder, 'excluded_quality')
    if not os.path.exists(excluded_folder):
        os.mkdir(excluded_folder)
    
    """
    for program, parent_folder, quality_folder in zip(programs, program_masters, quality_folders):
        print(program)
        if program == 'SPM':
            files = np.array(glob(os.path.join(parent_folder, '*.nii'))) # list of all niftis
            parent_files = [f for f in files if os.path.basename(os.path.normpath(f))[0] != 'c'] # if the nifti starts with c it's a tissue probability map 
            
            for i, p in enumerate(parent_files):
                mr = os.path.basename(os.path.normpath(p))[:-4]
                
                print(f'\nQuality check {mr} ({i+1} of {len(parent_files)})')
                
                # c1 is gm, c2 is wm, c3 is csf
                tissue_types = ['gm', 'wm', 'csf']
                tissue_long = ['gray_matter_volume', 'white_matter_volume', 'cerebrospinal_fluid_volume']
                tissue_numbers = ['c1', 'c2', 'c3']
                
                
                t1_file = os.path.join(spm_folder, f'{mr}.nii')
                t1_data = nib.load(t1_file)
                t1_mat = t1_data.get_fdata()
                t1_shape = t1_mat.shape
                
                half_x = int(t1_shape[0] / 2 + 10)
                half_y = int(t1_shape[1] / 2 + 10)
                half_z = int(t1_shape[2] / 2 + 10)
                
                slice1 = half_x,slice(None),slice(None)
                slice2 = slice(None),half_y,slice(None)
                slice3 = slice(None),slice(None),half_z
                
                slices = [slice1,slice2,slice3]
                
                fig, axs = plt.subplots(2, 3, figsize=(12, 12))
                figname = os.path.join(quality_folder,f'{mr}.png')
                
                cmaps = ['Reds', 'Blues', 'Greens']
                
                rots = [2,2,2]
                
                for i,axrow in enumerate(axs):
                    for ax, slicer, rot in zip(axrow,slices, rots):
                        ax.axis('off')
                        t1_slice = t1_mat[slicer]
                        t1_slice = np.rot90(t1_slice, rot)
                        ax.imshow(t1_slice, cmap=matplotlib.cm.gray)
                    
                        if i == 1:
                            for t_type, t_num, long, colormap in zip(tissue_types, tissue_numbers, tissue_long, cmaps):
                                
                                tissue_file = os.path.join(spm_folder, f'{t_num}{mr}.nii')
                                tissue_data = nib.load(tissue_file)
                                tissue_mat = tissue_data.get_fdata()
                                tissue_slice = tissue_mat[slicer]
                                #tissue_slice = np.rot90(tissue_slice.T, rot)
                                tissue_slice = np.rot90(tissue_slice, rot)
                                
                                tissue_mask = np.ma.masked_where(tissue_slice == 0, tissue_slice)
                                
                                ax.imshow(tissue_mask, cmap=colormap, alpha=0.3)
                    
                plt.tight_layout()
                fig.savefig(figname, dpi=150)
                plt.close('all')
                
        elif program == 'SIENAX':
            folders = np.array(glob(os.path.join(parent_folder, '*/'))) # list of all folders
            for i, f in enumerate(folders):
                mr = os.path.basename(os.path.normpath(f))
                im = os.path.join(f, 'bin/axT1_raw_sienax/I_render.png')
                target = os.path.join(quality_folder, f'{mr}.png')
                shutil.copyfile(im, target)
                
        elif program == 'FS':
            fs_folders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
            mrs = [os.path.basename(f) for f in fs_folders]
            
            for i,(f,mr) in enumerate(zip(fs_folders,mrs)):
                print(f'\nQuality check {mr} ({i+1} of {len(fs_folders)})')
                mri_folder = os.path.join(f, 'mri')
                brain_file = os.path.join(mri_folder, 'orig.mgz')
                seg_file = os.path.join(mri_folder, 'aseg.mgz')
                
                brain_data = nib.load(brain_file)
                brain_mat = brain_data.get_fdata()
                brain_shape = brain_mat.shape
                
                half_x = int(brain_shape[0] / 2 + 10)
                half_y = int(brain_shape[1] / 2 + 10)
                half_z = int(brain_shape[2] / 2 + 10)
                
                slice1 = half_x,slice(None),slice(None)
                slice2 = slice(None),half_y,slice(None)
                slice3 = slice(None),slice(None),half_z
                
                slices = [slice1,slice2,slice3]
                
                seg_data = nib.load(seg_file)
                seg_mat = seg_data.get_fdata()
                seg_shape = seg_mat.shape
                
                # mat codes that are not wm
                # 3, 42: left and right cortex
                # 8, 47: left and right cerebellar cortex
                
                # 4, 43: left and right ventricle
                # 14, 15: 3rd and 4th ventricle
                # 24: csf
                
                gm_codes = [3, 42, 8, 47]
                csf_codes = [4, 43, 14, 15, 25]
                
                not_wm_codes = []
                not_wm_codes.extend(gm_codes)
                not_wm_codes.extend(csf_codes)
                
                
                
                half_x_seg = int(seg_shape[0] / 2 + 10)
                half_y_seg = int(seg_shape[1] / 2 + 10)
                half_z_seg = int(seg_shape[2] / 2 + 10)
                
                slice1_seg = half_x_seg,slice(None),slice(None)
                slice2_seg = slice(None),half_y_seg,slice(None)
                slice3_seg = slice(None),slice(None),half_z_seg
                
                slices_seg = [slice1_seg,slice2_seg,slice3_seg]
                
                fig, axs = plt.subplots(2, 3, figsize=(12, 12))
                figname = os.path.join(quality_folder,f'{mr}.png')
                
                cmaps = ['Reds', 'Blues', 'Greens']
                
                rots = [0,1,3]
                
                for i,axrow in enumerate(axs):
                    for ax, slicer, rot, slicer_seg in zip(axrow, slices, rots, slices_seg):
                        ax.axis('off')
                        brain_slice = brain_mat[slicer]
                        brain_slice = np.rot90(brain_slice, rot)
                        ax.imshow(brain_slice, cmap=matplotlib.cm.gray)
                    
                        if i == 1:
                            seg_slice = seg_mat[slicer_seg]
                            seg_slice = np.rot90(seg_slice, rot)
                            seg_slice[seg_slice == 0] = np.nan
                            ax.imshow(brain_slice, cmap=matplotlib.cm.gray)
                            ax.imshow(seg_slice, cmap='gist_rainbow')
                    
                plt.tight_layout()
                
                
                
                fig.savefig(figname, dpi=150)
                plt.close('all')
        
    """
    
    q_files_paths = np.array(glob(os.path.join(quality_folders[0], '*.png'))) # list of all pngs
    q_file_names = [os.path.basename(f) for f in q_files_paths]
    
    for i,f in enumerate(q_file_names):
        pdf = FPDF()
        mr = f[:-4]
        print(f'Composite report for {mr} ({i+1} of {len(q_file_names)})')
        for program, parent_folder, quality_folder in zip(programs, program_masters, quality_folders):
            try:
                print(f'\tadding program {program}')
                
                target_file = os.path.join(quality_folder, f)
                
                pdf.add_page()
                pdf.set_xy(0, 0)
                pdf.set_font('arial', 'B', 16)
                pdf.cell(210, 10, f"{program}: {mr}", 0, 2, 'C')
                pdf.image(target_file, x = None, y = None, w = 200, h = 0, type = '', link = '')
                
            except FileNotFoundError:
                pass
            
        pdf_out = os.path.join(composite_quality_folder, f'{mr}_composite.pdf')
        pdf.output(pdf_out, 'F')
            
        sienax_collated = '/Users/manusdonahue/Documents/Sky/t1_volumizers/vis_SIENAX/collated.csv'
        sienax_collated_df = pd.read_csv(sienax_collated)
        excluded_list = list(sienax_collated_df[sienax_collated_df['exclude']==1]['mr_id'])
        
        if mr in excluded_list:
            
            excluded_out = os.path.join(excluded_folder, f'{mr}_composite.pdf')
            shutil.move(pdf_out, excluded_out)
    
            
            
    
        
if visualize:
    matplotlib.rcdefaults()
    print('Visualizing')
    brain_vol_df = pd.read_csv(brain_vol_csv)
    
    for prog, norm_name, out_folder in zip(programs, norm_columns, sub_outs):
    
        collated_csv = os.path.join(out_folder, 'collated.csv')
        clean_table = pd.read_csv(collated_csv, index_col='mr_id')
        
        """
        # replace icv estimates with FreeSurfer's estimates
        fs_csv = '/Users/manusdonahue/Documents/Sky/t1_volumizers/vis_FS/collated.csv'
        fs_table = pd.read_csv(fs_csv)
        
        clean_table[norm_name] = None
        for i, row in fs_table.iterrows():
            the_mr_id = row['mr_id']
            clean_table.loc[the_mr_id,norm_name] = row['icv']
        
        clean_table[norm_name] = clean_table[norm_name].astype('float64')
        """
        
        clean_table = clean_table[clean_table['exclude'] != 1]
        
        '''
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.06)
    
        y_pred = clf.fit_predict(clean_table)
        #y_pred_unsort = y_pred.copy()
        x_scores = clf.negative_outlier_factor_
        #x_scores_unsort = x_scores.copy()
        clean_table['outlier'] = y_pred
        '''
        
        clean_table['normal_control'] = [all([i, not j]) for i,j in zip(clean_table['control'], clean_table['sci'])]        
        clean_table['sci_control'] = [all([i, j]) for i,j in zip(clean_table['control'], clean_table['sci'])]   
        
        
        clean_table['normal_scd'] = [all([i, not j]) for i,j in zip(clean_table['scd'], clean_table['sci'])]        
        clean_table['sci_scd'] = [all([i, j]) for i,j in zip(clean_table['scd'], clean_table['sci'])]
        
        
        ######## nice clean figures for publication
        pred_vars = ['age', 'ox_delivery', 'lesion_count']
        interest = ['total_vol', 'gm_vol', 'wm_vol']
        figname = os.path.join(out_folder, 'manuscript_scatter.png')
        fig, axs = plt.subplots(len(pred_vars), len(interest), figsize=(4*len(interest),4*len(pred_vars)))
        
        for pred_var, axrow in zip(pred_vars, axs):
            
            pt_type = ['control', 'scd']
            exprs = [clean_table[pt] == 1 for pt in pt_type]
            subdfs = [clean_table[expr] for expr in exprs]
            
            for col, ax in zip(interest, axrow):
                
                subcolors = ['red', 'blue']
                int_colors = ['red', 'blue']
                markers = ['o', '^']
                
                for subcolor, subd, icolor, patient_type, mark in zip(subcolors, subdfs, int_colors, pt_type, markers):
                    
                    if patient_type == 'control' and pred_var in ['lesion_burden', 'lesion_count']:
                        continue
                    
                    print(f'pred_var: {pred_var}, col: {col}')
                    
                    exes = subd[pred_var]
                    whys = subd[col]
                    
                    hold = [(x,y) for x,y in zip(exes,whys) if not np.isnan(x)]
                    
                    exes = [x for x,y in hold]
                    whys = [y for x,y in hold]
            
            
            
                    ## BOOT STRAPPING. courtesy of pylang from stackoverflow
                    
                    x, y = exes, whys
                    
                    # Modeling with Numpy
                    def equation(a, b):
                        """Return a 1D polynomial."""
                        return np.polyval(a, b)
                    # Data
                    ax.plot(
                        x, y, "o", color="#b9cfe7", markersize=4,
                        markeredgewidth=1, markeredgecolor="black", markerfacecolor=subcolor,
                        marker=mark, alpha=0.3, label=patient_type
                        )
                    ax.plot(
                        x, y, "o", color="#b9cfe7", markersize=4,
                        markeredgewidth=1, markeredgecolor="black", markerfacecolor="None",
                        marker=mark
                        )
                    try:
                        p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
                        y_model = equation(p, x)                                   # model using the fit parameters; NOTE: parameters here are coefficients
                        
                        # Statistics
                        n = len(exes)                                         # number of observations
                        m = p.size                                                 # number of parameters
                        dof = n - m                                                # degrees of freedom
                        t = stats.t.ppf(0.975, n - m)                              # used for CI and PI bands
                        
                        # Estimates of Error in Data/Model
                        resid = y - y_model                          
                        chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
                        chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
                        s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error
                        
                        
                        # Fit
                        ax.plot(x, y_model, "-", color=icolor, linewidth=1.5, alpha=0.25)  
                        
                        x2 = np.linspace(np.min(x), np.max(x), 100)
                        y2 = equation(p, x2)
                        
                        # Confidence Interval (select one)
                        plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax, color=icolor)
                        #plot_ci_bootstrap(x, y, resid, ax=ax)
                        
                        # Prediction Interval
                        pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))  
                        ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
                        ax.plot(x2, y2 - pi, "--", color=icolor, alpha=0.3)#, label="95% Prediction Limits")
                        ax.plot(x2, y2 + pi, "--", color=icolor, alpha=0.3)
                        
                    except np.linalg.LinAlgError:
                        print('Linear algebra error, likely due to singular matrix')
                        pass
                    
                    #ax.scatter(exes, whys, color=subcolor, alpha=0.2, s=4, label=patient_type, marker=mark)
                    #ax.scatter(exes, whys, color=subcolor, alpha=0.2, s=4, label=patient_type, marker=mark)
                    ax.legend()
                
                if col == 'total_vol':
                    ax.set_title(f'Total brain volume')
                elif col == 'wm_vol':
                    ax.set_title(f'White matter volume')
                if col == 'gm_vol':
                    ax.set_title(f'Gray matter volume')
    
                ax.set_ylabel('Tissue volume (cc)')
                ax.set_ylim(200,1450)
                
                if pred_var == 'age':
                    ax.set_xlim(0,50)
                    ax.set_xlabel('Age (years)')
                elif pred_var == 'hct':
                    ax.set_xlim(0.15,0.55)
                    ax.set_xlabel('Hematocrit')
                elif pred_var == 'ox_delivery':
                    ax.set_xlim(4.5,23)
                    ax.set_xlabel('Arterial oxygen content (mL O2 / dL blood)')
                elif pred_var == 'lesion_burden':
                    #ax.set_xlim(0,6e4)
                    ax.set_xlabel('Lesion burden (cc)')
                elif pred_var == 'lesion_count':
                    #ax.set_xlim(0,6e4)
                    ax.set_xlabel('Lesion count')
           
            plt.tight_layout()
            plt.savefig(figname, dpi=400)
        
        
        ######## statistical significance of slopes
        '''
        pred_vars = ['age', 'hct', 'lesion_burden', 'ox_delivery']
        
        for pred_var in pred_vars:
            
            print(f'\n\n\nPRED VAR = {pred_var}\n\n\n')
        
            interest = ['gm_vol', 'wm_vol', 'total_vol'] # 'gm_vol', 'wm_vol', 'supratent', 'total_vol'], 
            pt_type_pairs = [['control', 'scd'], ['normal_control', 'normal_scd'] ,['sci_control','sci_scd']]
            fig, axs = plt.subplots(len(pt_type_pairs), len(interest), figsize=(4*len(interest),4*len(pt_type_pairs)))
            for pt_type, axrow in zip(pt_type_pairs, axs):
                
                print(f'Z-testing: {pt_type}')
                
                exprs = [clean_table[pt] == 1 for pt in pt_type]
                subdfs = [clean_table[expr] for expr in exprs]
                
                for col, ax in zip(interest, axrow):
                    
                    subcolors = ['red', 'blue']
                    int_colors = ['red', 'blue']
                    bs = []
                    ses = []
                    ens = []
                    for subcolor, subd, icolor, patient_type in zip(subcolors, subdfs, int_colors, pt_type):
                        print(f'Pt type = {patient_type}')
                        
                        exes = subd[pred_var]
                        whys = subd[col]
                        
                        hold = [(x,y) for x,y in zip(exes,whys) if not np.isnan(x)]
                        
                        exes = [x for x,y in hold]
                        whys = [y for x,y in hold]
                        
                        
                        
                        if not exes:
                            ens.append(0)
                            continue
                        
                        ens.append(len(exes))
                        
                        ## BOOT STRAPPING. courtesy of pylang from stackoverflow
                        
                        x, y = exes, whys
                        
                        # Modeling with Numpy
                        def equation(a, b):
                            """Return a 1D polynomial."""
                            return np.polyval(a, b)
                        # Data
                        ax.plot(
                            x, y, "o", color="#b9cfe7", markersize=4,
                            markeredgewidth=1, markeredgecolor="black", markerfacecolor="None"
                            )
                        try:
                            p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
                            y_model = equation(p, x)                                   # model using the fit parameters; NOTE: parameters here are coefficients
                            
                            # Statistics
                            n = len(exes)                                         # number of observations
                            m = p.size                                                 # number of parameters
                            dof = n - m                                                # degrees of freedom
                            t = stats.t.ppf(0.975, n - m)                              # used for CI and PI bands
                            
                            # Estimates of Error in Data/Model
                            resid = y - y_model                          
                            chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
                            chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
                            s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error
                            
                            bs.append(p[0])
                            ses.append(s_err)
                            
                            
                            # Fit
                            ax.plot(x, y_model, "-", color=icolor, linewidth=1.5, alpha=0.25)  
                            
                            x2 = np.linspace(np.min(x), np.max(x), 100)
                            y2 = equation(p, x2)
                            
                            # Confidence Interval (select one)
                            plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax, color=icolor)
                            #plot_ci_bootstrap(x, y, resid, ax=ax)
                            
                            # Prediction Interval
                            pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))  
                            ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
                            ax.plot(x2, y2 - pi, "--", color=icolor, alpha=0.3)#, label="95% Prediction Limits")
                            ax.plot(x2, y2 + pi, "--", color=icolor, alpha=0.3)
                            
                            """
                            # Figure Modifications --------------------------------------------------------
                            # Borders
                            ax.spines["top"].set_color("0.5")
                            ax.spines["bottom"].set_color("0.5")
                            ax.spines["left"].set_color("0.5")
                            ax.spines["right"].set_color("0.5")
                            ax.get_xaxis().set_tick_params(direction="out")
                            ax.get_yaxis().set_tick_params(direction="out")
                            ax.xaxis.tick_bottom()
                            ax.yaxis.tick_left()
                            
                            # Labels
                            plt.title("Fit Plot for Weight", fontsize="14", fontweight="bold")
                            plt.xlabel("Height")
                            plt.ylabel("Weight")
                            plt.xlim(np.min(x) - 1, np.max(x) + 1)
                            
                            # Custom legend
                            handles, labels = ax.get_legend_handles_labels()
                            display = (0, 1)
                            anyArtist = plt.Line2D((0, 1), (0, 0), color="#b9cfe7")    # create custom artists
                            legend = plt.legend(
                                [handle for i, handle in enumerate(handles) if i in display] + [anyArtist],
                                [label for i, label in enumerate(labels) if i in display] + ["95% Confidence Limits"],
                                loc=9, bbox_to_anchor=(0, -0.21, 1., 0.102), ncol=3, mode="expand"
                            )  
                            frame = legend.get_frame().set_edgecolor("0.5")
                            
                            # Save Figure
                            plt.tight_layout()
                            plt.savefig("filename.png", bbox_extra_artists=(legend,), bbox_inches="tight")
                            
                            plt.show()
                            """
                        except np.linalg.LinAlgError:
                            print('Linear algebra error, likely due to singular matrix')
                            pass
                        
                        ax.scatter(exes, whys, color=subcolor, alpha = 0.2, s=4, label=patient_type)
                        ax.legend()
                       
                    try:
                        z_stat = abs((bs[0] - bs[1]) / np.sqrt(ses[0]**2 + ses[1]**2))
                    except IndexError:
                        # this happens if you only have one of SCD or control because you cant compares slopes
                        z_stats = 'n/a'
                    # Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). Applied multiple regression/correlation analysis for the behavioral sciences (3rd ed.)
                    # Paternoster, R., Brame, R., Mazerolle, P., & Piquero, A. R. (1998). Using the Correct Statistical Test for the Equality of Regression Coefficients. Criminology, 36(4), 859–866.
                    ax.set_title(f'Groups: {pt_type}\nn = {ens}\nCriterion: {col}\n(zstat = {round(z_stat,2)})')
                    
                    if 'norm' in col:
                        ax.set_ylabel('Normalized volume')
                        ax.set_ylim(-0.1,1.1)
                    elif 'lesion' in col:
                        ax.set_ylabel('Lesion burden (cc)')
                        ax.set_ylim(0,1)
                    else:
                        ax.set_ylabel('Tissue volume (cc)')
                        ax.set_ylim(0,1500)
                    
                    if pred_var == 'age':
                        ax.set_xlim(0,50)
                        ax.set_xlabel('Age (years)')
                    elif pred_var == 'hct':
                        ax.set_xlim(0.15,0.55)
                        ax.set_xlabel('Hematocrit')
                    elif pred_var == 'ox_delivery':
                        ax.set_xlim(7,23)
                        ax.set_xlabel('Oxygen delivery (mL O2 / dL blood)')
                    elif pred_var == 'lesion_burden':
                        #ax.set_xlim(0,6e4)
                        ax.set_xlabel('Lesion burden (cc)')
               
            plt.tight_layout()
            nice_name = os.path.join(out_folder, f'sig_testing_{pred_var}.png')
            plt.savefig(nice_name, dpi=400)
            '''
        

        # multiple linear regression looking at brain vol vs x,y,z
        
        factors = ['gm_vol', 'wm_vol', 'total_vol', 'lesion_count'] # lesion count / lesion burden
        
        """
        controlling = [
               ['age','gender', norm_name, 'scd', 'ox_delivery'],
               ['age','gender', norm_name, 'ox_delivery'],
               ['age','gender', norm_name, 'ox_delivery', 'lesion_count']
        ]
        
        keep_nonscd = [True, False, False]
        """
        
        controlling = [
               ['age','gender', norm_name, 'scd'],
               ['age','gender', norm_name, 'ox_delivery'],
               ['age','gender', norm_name, 'ox_delivery'],
               ['age','gender', norm_name, 'lesion_count']
        ]
        
        keep_nonscd = [True, True, False, False]
        
        '''
        controlling = [
               ['age','gender', 'scd', 'ox_delivery'],
               ['age','gender', 'lesion_count']
        ]
        keep_nonscd = [True, False]
        '''
        
        p_df = pd.DataFrame()
        p_df_name = os.path.join(out_folder, f'pvals.csv')
        
        corr_check = ['age', 'gender', norm_name, 'scd', 'ox_delivery', 'lesion_count', 'gm_vol', 'wm_vol', 'total_vol']
        corr_base = clean_table[corr_check].dropna()
        corr_mat_file = os.path.join(out_folder, f'correlation_matrix_{prog}.csv')
        #corr_file = open(corr_mat_file, 'w')
        corr_mat = corr_base.corr()
        corr_mat.to_csv(corr_mat_file)
        #corr_file.write(str(corr_mat))
        #corr_file.close()
            
        for controller, keeper in zip(controlling, keep_nonscd):
            summary_file = os.path.join(out_folder, f'signficance_summary_{"_".join(controller)}.txt')
            summary = open(summary_file, 'w')
            
            print('QUAD')
            for f in factors:
                #print(f'\n\n\nFACTOR: {f}\n')
                
                if f in controller:
                    continue # doesn't make sense to run a regression where something is both a predictor and the criterion
                
                
               
                pars = controller.copy()
                if f not in pars:
                    pars.append(f)
                    
                if 'scd' not in pars:
                    pars.append('scd')
                
                tabby = clean_table[pars].dropna()
                
                if 'transf' in controller:# if we're testing transfusion status, we need to evaluate SCD pts only
                    keeper = False
                    
                if not keeper:
                    tabby = tabby[tabby['scd']==1] # only keep SCD participants
                
                X = tabby[controller]
                Y = tabby[f]
                
                X2 = sm.add_constant(X)
                est = sm.OLS(Y, X2)
                est2 = est.fit()
                summary.write(str(est2.summary()))
                summary.write('\n\n\n\n--------\n--------\n\n\n\n')
                
                results_summary = est2.summary()
                
                results_as_html = results_summary.tables[1].as_html()
                as_df = pd.read_html(results_as_html, header=0)[0]
                as_df['criterion'] = f
                as_df['covariates'] = '+'.join(controller)
                as_df['keep_nonscd'] = keeper
                
                as_df = as_df.rename(columns={'Unnamed: 0':'predictor'})
                
                #droppers = ['const', 'age', 'gender']
                #for d in droppers:
                #    as_df = as_df[as_df['predictor'] != d]
                
                
                p_df = p_df.append(as_df, ignore_index=True)
                
                
            
            
            summary.close()
            p_df.to_csv(p_df_name)
            
            
        # violin plots of icv for SCD vs control
        def set_axis_style(ax, labels):
            ax.get_xaxis().set_tick_params(direction='out')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_xlim(0.25, len(labels) + 0.75)
            #ax.set_xlabel('Sample name')
            
        violin_name = os.path.join(out_folder, f'icv_violins_{prog}.png')
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        #ax1.set_title('BMI')
        if prog == 'SIENAX':
            ax.set_ylabel('VSCALING')
            fac = 'vscaling'
        else:
            ax.set_ylabel('ICV (cc)')
            fac = 'icv'
        
        
        data_icv = [clean_table[clean_table['control']==1][fac], clean_table[clean_table['scd']==1][fac]]
        icv_labs = ['Control', 'SCD']
        parts = ax.violinplot(data_icv, showmeans=True, showmedians=True)
        set_axis_style(ax, icv_labs)
        
        med_col = 'cornflowerblue'
        mean_col = 'darkorange'
        
        lwdth = 1
        
        custom_lines = [Line2D([0], [0], color=med_col, lw=lwdth),
                        Line2D([0], [0], color=mean_col, lw=lwdth)]
        
        ax.legend(custom_lines, ['Median', 'Mean'])
        if prog != 'SIENAX':
            ax.set_ylim(1000,1800)
        ax.set_title(prog)
        
        
        for parts in [parts]:
            for pc in parts['bodies']:
                pc.set_facecolor('green')
                pc.set_edgecolor('black')
                pc.set_alpha(0.2)
                
            parts['cbars'].set_color('black')
            parts['cmaxes'].set_color('black')
            parts['cmins'].set_color('black')
            
            parts['cmedians'].set_color(med_col)
            parts['cmeans'].set_color(mean_col)
            
            parts['cmedians'].set_linewidth(lwdth)
            parts['cmeans'].set_linewidth(lwdth)
            
        plt.tight_layout()
        plt.savefig(violin_name, dpi=200)
            
            
            
if interrater:
    
    
    interrater_folder = os.path.join(out_folder_orig, 'interrater')
    if not os.path.exists(interrater_folder):
        os.mkdir(interrater_folder)
    
    matplotlib.rcdefaults()
    print('Interrater!')
    
    data_dicts = {}
    for prog, norm_name, out_folder in zip(programs, norm_columns, sub_outs):
    
        collated_csv = os.path.join(out_folder, 'collated.csv')
        clean_table = pd.read_csv(collated_csv, index_col='mr_id')
        clean_table = clean_table[clean_table['exclude'] != 1]
        
        inner = {'data':clean_table, 'norm':norm_name}
        data_dicts[prog] = inner
            
    '''
    
    # triangle plot of raw volumes
    triangle_folder = os.path.join(interrater_folder, 'triangles')
    if os.path.exists(triangle_folder):
        shutil.rmtree(triangle_folder)
    os.mkdir(triangle_folder)
    
    exes_l = [data_dicts[p]['data']['gm_vol'] for p in programs]
    whys_l = [data_dicts[p]['data']['wm_vol'] for p in programs]
    

    
    n_triangles = len(exes_l[1])
    
    parents_exes = exes_l[1].sample(n_triangles) # sample from FS since it's the most limited patient-wise
    names = list(parents_exes.index)
    
    cor_ex = []
    cor_why = []
    
    for i in range(len(programs)):
        cor_ex.append([exes_l[i].loc[name] for name in names])
        cor_why.append([whys_l[i].loc[name] for name in names])
        
    for i, na in enumerate(names):
        
        fig, axs = plt.subplots(1, 1, figsize=(12,8))
        
        backlines = np.arange(200,2000,200)
        for num in backlines:
            axs.plot([0, num], [num, 0], color='red', alpha=0.4)
        
        
        for p, exes, whys in zip(programs, exes_l, whys_l):
            axs.scatter(exes,whys,label=p,alpha=0.5) # doing this within the loop because we're going to make a GIF
            axs.set_title(f'Volumetric comparison: {na}')
        
        

        
        the_exes = []
        the_whys = []
        for li_ex, li_why in zip(cor_ex, cor_why):
            the_exes.append(li_ex[i])
            the_whys.append(li_why[i])
        axs.scatter(the_exes, the_whys, facecolors='none', edgecolors='black')

        pairs = [[a,b] for a,b, in zip(the_exes,the_whys)]
        possibles = itertools.combinations(pairs, 2)
        
        for pos in possibles:
            egs = [pos[0][0], pos[1][0]]
            whi = [pos[0][1], pos[1][1]]
            axs.plot(egs, whi, color='black')
            
        axs.set_xlim(400,1000)
        axs.set_ylim(200,700)
        
        axs.set_xlabel('Gray matter volume (cc)')
        axs.set_ylabel('White matter volume (cc)')   
        plt.gca().set_aspect('equal', adjustable='box')
        axs.legend()
        plt.tight_layout()
            
        outname = os.path.join(triangle_folder, f'{na}.png')
        plt.savefig(outname, dpi=70)
        
        
    imglob = glob(os.path.join(triangle_folder, '*'))
    images = []
    for filename in imglob:
        images.append(imageio.imread(filename))
        
    triangle_path = os.path.join(interrater_folder, 'brainvol_triangle.gif')
    imageio.mimsave(triangle_path, images, duration=0.35)
    
    ####### static triangle plot
    
    cor_ex = []
    cor_why = []
    
    for i in range(len(programs)):
        cor_ex.append([exes_l[i].loc[name] for name in names])
        cor_why.append([whys_l[i].loc[name] for name in names])
    

    fig, axs = plt.subplots(1, 1, figsize=(12,8))
    
    backlines = np.arange(200,2000,200)
    for num in backlines:
        axs.plot([0, num], [num, 0], color='red', alpha=0.4)
    
    
    for p, exes, whys in zip(programs, exes_l, whys_l):
        axs.scatter(exes,whys,label=p,alpha=0.5)
        axs.set_title(f'Volumetric comparison')

            
        axs.set_xlim(400,1000)
        axs.set_ylim(200,700)
        
        axs.set_xlabel('Gray matter volume (cc)')
        axs.set_ylabel('White matter volume (cc)')   
        plt.gca().set_aspect('equal', adjustable='box')
        axs.legend()
        plt.tight_layout()
    
    
    triangle_path_static = os.path.join(interrater_folder, 'brainvol_triangle_static.png')
    plt.savefig(triangle_path_static, dpi=200)

    '''
    
    # Polynomial Regression
    def custom_polyfit(x, y, degree):
        """
        Adapated frm holocronweaver's answer on stackoverflow
        """
        results = {}
    
        coeffs = np.polyfit(x, y, degree)
    
         # Polynomial Coefficients
        results['polynomial'] = coeffs.tolist()
    
        # r-squared
        p = np.poly1d(coeffs)
        # fit values, and mean
        yhat = p(x)                         # or [p(z) for z in x]
        ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
        ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
        results['r2'] = ssreg / sstot
    
        return results
    
    # now plot the vscaling factor against corresponding ICVs
    scaling_path = os.path.join(interrater_folder, 'volumetric_scaling_agreement.png')
    fig, axs = plt.subplots(1, 1, figsize=(12,8))
    
    color_list = ['green', 'blue']
    progs = ['SPM', 'FS']
    add_to_legend = 2
    
    has_black = False
    has_blue = False
    has_red = False
    
    for val, name in zip(data_dicts['SIENAX']['data']['vscaling'], data_dicts['SIENAX']['data'].index):
        
        pexes = []
        pwhys = []
        
        for color, pro in zip(color_list, progs):
            try:
                px = val
                py = data_dicts[pro]['data']['icv'].loc[name]
                
                pexes.append(px)
                pwhys.append(py)
                
                if add_to_legend != 0:
                    axs.scatter(px, py, c=color, alpha=0.5, label=pro)
                    add_to_legend -= 1
                else:
                    axs.scatter(px, py, c=color, alpha=0.5)
            except KeyError:
                pass
            

        try:
            the_diff = abs(np.diff(pwhys)[0])
            if the_diff < 100:
                linecol = 'black'
                if not has_black:
                    lab = 'ICV difference < 100'
                else:
                    lab = ''
                has_black = True
            elif the_diff < 200:
                linecol = 'blue'
                if not has_blue:
                    lab = 'ICV difference > 100'
                else:
                    lab = ''
                has_blue = True
            else:
                linecol = 'red'
                if not has_red:
                    lab = 'ICV difference > 200'
                else:
                    lab = ''
                has_red = True
        except IndexError:
            linecol = 'black'
            lab = ''
        
        axs.plot(pexes, pwhys, color=linecol, alpha=0.2, label=lab)
        
    axs.set_xlim(1,2)
    axs.set_ylim(1000,2000)
    
    axs.legend()
    
    axs.set_xlabel('SIENAX VSCALING factor')
    axs.set_ylabel('Estimated ICV (cc)')
    axs.set_title('Agreement of volume-normalizing covariates')
    
    plt.tight_layout()
    plt.savefig(scaling_path)
    
    # plot FS ICV vs SPM ICV
    icv_path = os.path.join(interrater_folder, 'icv_SPMvsFS.png')
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    
    exes1 = data_dicts['FS']['data']['icv']
    exes2 = data_dicts['SPM']['data']['icv']
    
    ax.scatter(exes1,exes2, color='cornflowerblue', edgecolors='black', alpha=0.8)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('FS ICV (cc)')
    ax.set_ylabel('SPM ICV (cc)')
    ax.set_xlim(1000,1800)
    ax.set_ylim(1000,1800)
    ax.plot([0,2000],[0,2000], color='black', alpha=0.25)
    
    fit = custom_polyfit(exes1,exes2,1)
            
    coefs = fit['polynomial']
    r2 = fit['r2']
    
    ax.set_title(f'$r^2$ = {round(r2,3)}')
    
    fit_exes = [0,2000]
    fit_whys = [x*coefs[0] + coefs[1] for x in fit_exes]
    ax.plot(fit_exes, fit_whys, c='black')
    
    plt.tight_layout()
    plt.savefig(icv_path, dpi=200)
    
    
    ##### scatter+bland-altman plots
    
    def bland_altman_plot(data1, data2, ax, left_loc=None, *args, **kwargs):
        """
        Based on Neal Fultz' answer on Stack Overflow
        """
        
        data1     = np.asarray(data1)
        data2     = np.asarray(data2)
        mean      = np.mean([data1, data2], axis=0)
        diff      = data1 - data2                   # Difference between data1 and data2
        md        = np.mean(diff)                   # Mean of the difference
        sd        = np.std(diff)            # Standard deviation of the difference
        
        ax.scatter(mean, diff, *args, **kwargs)
        ax.axhline(md,           color='gray')
        ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
        ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
        
        if not left_loc:
            left_loc = min(mean)
        
        ax.annotate(f'Mean diff: {round(md,2)}', (left_loc,md+6), path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.75)])
        ax.annotate(f'-SD 1.96: {round(md-1.96*sd,2)}', (left_loc,md-1.96*sd+6), path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.75)])
        ax.annotate(f'+SD 1.96: {round(md+1.96*sd,2)}', (left_loc,md+1.96*sd+6), path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.75)])
        
        #ax.text(0.5, 0.5, f'Mean diff: {round(md,2)}',
        #      size=20,
        #      color='white',
        #      path_effects=[pe.withStroke(linewidth=4, foreground="red")])
        
        ax.set_xlabel("Mean (cc)")
        ax.set_ylabel("Difference (cc)")
        
    
    
    out_of_spec = []
    
    vol_measures = ['total_vol', 'gm_vol', 'wm_vol']
    formal_measures = ['Total volume', 'Gray matter volume', 'White matter volume']
    lim_list =[[700,1400],[400,900],[200,700]]
    lim_list_bland_ex = [[700,1300],[450,900],[250,650]]
    lim_list_bland_why = [[-200,200],[-100,200],[-150,50]]
    program_pairs = list(itertools.combinations(data_dicts.keys(), 2))
    
    for lims, measure, f_measure, bl_x, bl_y in zip(lim_list, vol_measures, formal_measures, lim_list_bland_ex, lim_list_bland_why):
        fig, axs = plt.subplots(len(program_pairs), 2, figsize=(12,24))
        for (p1, p2), axrow in zip(program_pairs, axs):
            
            if p1 == 'FS':
                progname1 = 'FreeSurfer'
            else:
                progname1 = p1
            
            if p2 == 'FS':
                progname2 = 'FreeSurfer'
            else:
                progname2 = p2
            
            exes1 = []
            exes2 = []
            
            d1 = data_dicts[p1]['data']
            d2 = data_dicts[p2]['data']
            
            inds = list(d1.index)
            
            for ind in inds:
                exes1.append(d1[measure].loc[ind])
                exes2.append(d2[measure].loc[ind])
                
            exes1 = np.array(exes1)
            exes2 = np.array(exes2)
            
            the_diff = exes2 - exes1
            the_mean = np.mean(the_diff)
            the_std = np.std(the_diff)*2
            upper_lim = the_mean + the_std
            lower_lim = the_mean - the_std
            
            out_of = the_diff > upper_lim
            outters = [name for name,boo in zip(inds,out_of) if boo]
            
            out_of_spec.extend(outters)
            
            axrow[0].plot([-100,10000], [-100,10000], c='gray', alpha=0.3)
                
            axrow[0].scatter(exes1, exes2, c='salmon', edgecolors='black', alpha=0.75)
            axrow[0].set_xlim(lims[0], lims[1])
            axrow[0].set_ylim(lims[0], lims[1])
            axrow[0].set_xlabel(f'{progname1} (cc)')
            axrow[0].set_ylabel(f'{progname2} (cc)')
            axrow[0].set_aspect('equal', 'box')
            
            if measure == 'total_vol':
                the_by = 200
            else:
                the_by = 100
            
            axrow[0].set_xticks(np.arange(lims[0], lims[1]+1, the_by))
            axrow[0].set_yticks(np.arange(lims[0], lims[1]+1, the_by))
            
            fit = custom_polyfit(exes1,exes2,1)
            
            coefs = fit['polynomial']
            r2 = fit['r2']
            
            fit_exes = lims.copy()
            fit_whys = [x*coefs[0] + coefs[1] for x in fit_exes]
            axrow[0].plot(fit_exes, fit_whys, c='black')
            
            axrow[0].set_title(f'{progname2} vs. {progname1} ($r^2$ = {round(r2,2)})')
            
            bland_altman_plot(exes1, exes2, ax=axrow[1], c='cornflowerblue', left_loc=bl_x[0]+10, edgecolors='black', alpha=0.75)
            axrow[1].set_xlim(bl_x[0],bl_x[1])
            axrow[1].set_ylim(bl_y[0],bl_y[1])
            
        
        fig.suptitle(f_measure)
        fig.tight_layout(rect=[0.01, 0.03, 1, 0.95])
        figname =  os.path.join(interrater_folder, f'agreement_{measure}.png')
        plt.savefig(figname, dpi=400)
        
    unique_out = set(out_of_spec)
        
        
    
    
    
    
        
        
        
        
             

if graphs_w_overt:
    
    collated_csvs = [os.path.join(out_folder_orig, f'vis_{program}', 'collated.csv') for program in programs]
    csv_dfs = [pd.read_csv(i)[~pd.read_csv(i)['mr_id'].isin(exclude_pts)] for i in collated_csvs]

        
    all_df = pd.DataFrame()
    for df, program in zip(csv_dfs, programs):
        df['program'] = program
        all_df = all_df.append(df, ignore_index=True)
    
    box_folder = os.path.join(out_folder_orig, 'boxes')
    if not os.path.exists(box_folder):
        os.mkdir(box_folder)
        
    
    figname = os.path.join(box_folder, 'boxes.png')
    
    cats = ['wm_vol', 'gm_vol', 'total_vol']
    pt_types = ['healthy', 'sci', 'overt']
    
    fig, axs = plt.subplots(len(cats), len(pt_types), figsize=(len(cats)*4,len(pt_types)*8))
    
    pt_type_str = []
    for ind, i in all_df.iterrows():
        
        if i['sci'] == 0 and i['stroke_overt'] == 0:
            a = 'healthy'
        elif i['sci'] == 1 and i['stroke_overt'] == 0:
            a = 'sci'
        elif i['stroke_overt'] == 1:
            a = 'stroke_overt'
            
        pt_type_str.append(a)
        
    all_df['pt_type_str'] = pt_type_str
    
    for row, prog in zip(axs, programs):
        for ax, cat in zip(row, cats):
            
            """         
            if pt_type == 'healthy':
                boo = [all(i) for i in (zip(all_df['sci'] == 0, all_df['stroke_overt'] == 0))]
            elif pt_type == 'sci':
                boo = [all(i) for i in (zip(all_df['sci'] == 1, all_df['stroke_overt'] == 0))]
            elif pt_type == 'overt':
                boo = all_df['stroke_overt'] == 1"""
            subdf = all_df[all_df['program']==prog]
            
            subdf.boxplot(column=cat, by='pt_type_str', ax=ax, grid=False)
            
            the_title = f'{prog}\n{cat} (n={int(len(subdf))})'
            ax.set_title(the_title)
            ax.set_ylim(0,1500)
            ax.set_ylabel('Volume (cc)')
    
    fig.tight_layout()
    fig.savefig(figname, dpi=400)
        
        
        
            

        