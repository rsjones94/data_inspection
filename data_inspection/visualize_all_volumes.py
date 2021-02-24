
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

from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data
import matplotlib
import matplotlib.patheffects as pe

from parse_fs_stats import parse_freesurfer_stats
from parse_sienax_stats import parse_sienax_stats


exclude_pts_bad_seg_spm = ['SCD_C011_01', 'SCD_K003', 'SCD_K033_02', 'SCD_K061', 'SCD_K065',
               'SCD_P007', 'SCD_P008', 'SCD_P009_01', 'SCD_P010_01', 'SCD_P012', 'SCD_P014',
               'SCD_P019', 'SCD_P021', 'SCD_TRANSF_P006_01', 'SCD_TRANSF_P007_01', 'K011',
               'K017', 'SCD_C038', 'SCD_K009', 'SCD_K021']

exclude_pts_bad_seg_fs = ['K018_01', 'K019_01', 'K020_01', 'K021_01', 'K022_01', 'K023_01', 'K024_01']

exclude_pts_bad_seg_sienax = ['K001', 'SCD_C011_01', 'SCD_K003', 'SCD_K020',
                              'SCD_K024', 'SCD_K027', 'SCD_K029', 'SCD_K033_02',
                              'SCD_K034', 'SCD_K035', 'SCD_K037', 'SCD_K039', 'SCD_K040',
                              'SCD_K041', 'SCD_K043', 'SCD_K048', 'SCD_K050', 'SCD_K054_01',
                              'SCD_K061', 'SCD_K065', 'SCD_P007', 'SCD_P008', 'SCD_P009_01',
                              'SCD_P010_01', 'SCD_P012', 'SCD_P014', 'SCD_P019', 'SCD_P021',
                              'SCD_TRANSF_012_01-SCD_TRANSP_P002_01', 'SCD_TRANSF_P006_01',
                              'SCD_TRANSF_P007_01', 'SCD_TRANSF_P011_01', 'SCD_TRANSF_P017_01',
                              'SCD_TRANSF_P022_01', 'SCD_TRANSP_P004_01', 'SCD_K004', 'SCD_K022', 'SCD_K025',
                              'SCD_K028', 'SCD_K036']

exclude_pts_manual = ['SCD_C018', 'SCD_C022', 'SCD_C023', 'K001', 'SCD_K029', 'SCD_K031', 'SCD_K050', 'SCD_K061'] # based on REDCap exclusions, with some exceptions

exclude_pts_manual.extend(['SCD_K028', 'SCD_TRANSF_K017_01', 'SCD_C038', 'SCD_K064_01',
                           'SCD_K009', 'SCD_TRANSF_K018_01', 'SCD_K042', 'SCD_K046', 'K017', 'SCD_K022',
                           'SCD_P029', 'SCD_K004', 'SCD_K051', 'SCD_K073', 'SCD_TRANSP_P014_01', 'SCD_K025',
                           'SCD_K021', 'SCD_K052_01', 'K011', 'SCD_K036', 'SCD_TRANSP_K001_01', 'SCD_K030',
                           'K016', 'SCD_TRANSF_A001_01']) # pts with >2 SD disagreement between raters

in_csv = '/Users/manusdonahue/Documents/Sky/stroke_status.csv'

out_folder = '/Users/manusdonahue/Documents/Sky/t1_volumizers/'
out_folder_orig = out_folder

lesion_mask_folder = '/Users/manusdonahue/Documents/Sky/brain_lesion_masks/combined/'

brain_vol_csv = '/Users/manusdonahue/Documents/Sky/normal_brain_vols.csv' # from Borzage, Equations to describe brain size across the continuum of human lifespan (2012)
# values originally reported as mass in g, converted to cc assuming rho = 1.04 g/cc

sienax_folder = '/Users/manusdonahue/Documents/Sky/sienax_segmentations/'
fs_folder = '/Volumes/DonahueDataDrive/freesurfer_subjects_scd/'
spm_folder = '/Users/manusdonahue/Documents/Sky/scd_t1s/'

parse = False
collate = True
quality_check = False
visualize = False
interrater = False
graphs_w_overt = False

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

exclude_pts = []
exclude_pts.extend(exclude_pts_manual)
for l in [exclude_pts_bad_seg_spm, exclude_pts_bad_seg_fs, exclude_pts_bad_seg_sienax]:
        exclude_pts.extend(l)


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
                  'icv':None,
                  'csf_vol':None,
                  'gm_normal':None,
                  'wm_normal':None,
                  'total_normal':None,
                  'vscaling':None,
                  'hct':None,
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
                  'gender':None,
                  'intracranial_stenosis':None,
                  'hydroxyurea':None,
                  'transf':None,
                  'hemoglobin':None,
                  'hemoglobin_s_frac':None,
                  'pulseox':None,
                  'bmi':None,
                  'diabetes':None,
                  'high_cholesterol':None,
                  'coronary_art_disease':None,
                  'smoker':None,
                  'exclude':0}
    
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
            if pt_name in exclude_pts:
                working['exclude'] = 1
                
            for ix in '23456':
                if f'_0{ix}' in pt_name:
                    working['exclude'] = 1
            
            working['ox_delivery'] = float(cands.iloc[0][f'mr1_cao2'])/100
            
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
            
            sci = (cands.iloc[0][f'outcome_mri1_sci'])
            transf = (cands.iloc[0][f'enroll_sca_transfusion'])
            
            for val, name in zip([stroke_overt, stroke_silent, sci, transf],
                                 ['stroke_overt', 'stroke_silent', 'sci', 'transf']):
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
                
            if working['scd'] == 0 and working['sci'] == 1:
                working['exclude'] = 1
                
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
            
            # software specific:
            if prog == 'SPM':
                working['gm_vol'] = parsed_csv.loc['gm']['value'] / 1e3
                working['wm_vol'] = parsed_csv.loc['wm']['value'] / 1e3
                working['total_vol'] = working['gm_vol'] + working['wm_vol']
                
                working['csf_vol'] = parsed_csv.loc['csf']['value'] / 1e3
                working['icv'] = working['gm_vol'] + working['wm_vol'] + working['csf_vol']
                working['gm_normal'] = working['gm_vol'] / working['icv']
                working['wm_normal'] = working['wm_vol'] / working['icv']
                working['total_normal'] = working['total_vol'] / working['icv']
                
            elif prog == 'SIENAX':
                working['gm_vol'] = parsed_csv.loc['gm']['value'] / 1e3
                working['wm_vol'] = parsed_csv.loc['wm']['value'] / 1e3
                working['total_vol'] = working['gm_vol'] + working['wm_vol']
                
                working['vscaling'] = parsed_csv.loc['scaling']['value']
                
            elif prog == 'FS':
                working['gm_vol'] = parsed_csv.loc['TotalGrayVol']['value'] / 1e3
                working['total_vol'] = parsed_csv.loc['BrainSegVolNotVent']['value'] / 1e3
                working['wm_vol'] = working['total_vol'] - working['gm_vol']
                working['icv'] = parsed_csv.loc['eTIV']['value'] / 1e3
                working['gm_normal'] = working['gm_vol'] / working['icv']
                working['wm_normal'] = working['wm_vol'] / working['icv']
                working['total_normal'] = working['total_vol'] / working['icv']
                working['csf_vol'] = working['icv'] - working['total_vol']
            
            else:
                raise Exception('you probably spelled something wrong')
            
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
                    
                    if working['lesion_burden'] > 3:
                        working['exclude'] = 1
                    
                except FileNotFoundError:
                    pass
            else:
                working['lesion_burden'] = 0
                
            
                    
            out_df = out_df.append(working, ignore_index=True)
        
        out_df = out_df[blank_dict.keys()]
        out_csv = os.path.join(out_folder, f'collated.csv')
        out_df.to_csv(out_csv, index=False)
        
        # now make the demographic table
        
        the_cols = ['age', 'race', 'gender', 'sci', 'intracranial_stenosis', 'hydroxyurea',
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
        
        categorical = ['race', 'gender', 'sci', 'intracranial_stenosis', 'hydroxyurea', 'diabetes',
                       'high_cholesterol', 'coronary_art_disease', 'smoker']
        categorical_names = ['Black race', 'Male sex', 'Has SCI', 'Intracranial stenosis >50%',
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
                fig.savefig(figname, dpi=400)
                plt.close('all')
        elif program == 'SIENAX':
            folders = np.array(glob(os.path.join(parent_folder, '*/'))) # list of all folders
            for i, f in enumerate(folders):
                mr = os.path.basename(os.path.normpath(f))
                im = os.path.join(f, 'bin/axT1_raw_sienax/I_render.png')
                target = os.path.join(quality_folder, f'{mr}.png')
                shutil.copyfile(im, target)
                
        elif program == 'FS':
            pass
            
            
    
        
if visualize:
    matplotlib.rcdefaults()
    print('Visualizing')
    brain_vol_df = pd.read_csv(brain_vol_csv)
    
    for prog, norm_name, out_folder in zip(programs, norm_columns, sub_outs):
    
        collated_csv = os.path.join(out_folder, 'collated.csv')
        clean_table = pd.read_csv(collated_csv, index_col='mr_id')
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
        pred_vars = ['age', 'ox_delivery', 'lesion_burden']
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
                
                for subcolor, subd, icolor, patient_type in zip(subcolors, subdfs, int_colors, pt_type):
                    
                    if patient_type == 'control' and pred_var == 'lesion_burden':
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
                    
                    ax.scatter(exes, whys, color=subcolor, alpha = 0.2, s=4, label=patient_type)
                    ax.legend()
                
                if col == 'total_vol':
                    ax.set_title(f'Total volume')
                elif col == 'wm_vol':
                    ax.set_title(f'White matter')
                if col == 'gm_vol':
                    ax.set_title(f'Gray matter')
    
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
                    ax.set_xlabel('Oxygen delivery (mL O2 / dL blood)')
                elif pred_var == 'lesion_burden':
                    #ax.set_xlim(0,6e4)
                    ax.set_xlabel('Lesion burden (cc)')
           
            plt.tight_layout()
            plt.savefig(figname, dpi=400)
        
        
        ######## statistical significance of slopes
        
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
        
        
        
        # multiple linear regression looking at brain vol vs x,y,z
        
        factors = ['gm_vol', 'wm_vol', 'total_vol', 'ox_delivery', 'lesion_burden']
        
        """
        print('TRI')
        for f in factors:
            print(f'\n\n\nFACTOR: {f}\n\n')
            X = clean_table[['age','gender','scd']]
            Y =  clean_table[f]
            
            X2 = sm.add_constant(X)
            est = sm.OLS(Y, X2)
            est2 = est.fit()
            print(est2.summary())
        """
        controlling = [
               ['age','gender', norm_name, 'scd', 'ox_delivery'],
               ['age','gender', norm_name, 'ox_delivery'],
               ['age','gender', norm_name, 'ox_delivery', 'lesion_burden']
        ]
        
        """
        controlling = [
                       ['age','gender', norm_name, 'transf', 'lesion_burden', 'hct'],
                       ['age','gender', norm_name, 'transf', 'hct'],
                       ['age','gender', norm_name, 'scd', 'lesion_burden', 'hct'],
                       ['age','gender', norm_name, 'scd', 'hct'],
                       ['age','gender', norm_name, 'scd', 'lesion_burden'],
                       ['age','gender', norm_name, 'scd', 'ox_delivery'],
                       ['age','gender', norm_name, 'ox_delivery']
                       ]
        """
        
        """
        
            controlling = [['age','gender','scd', 'sci'],
                       ['age','gender','hct'],
                       ['age','gender','lesion_burden'],
                       ['age','gender','lesion_burden', 'hct'],
                       ['age','gender', 'scd', 'lesion_burden', 'hct'],
                       ['age','gender','lesion_burden', 'hct', 'icv'],
                       ['age','gender','scd', 'lesion_burden'],
                       ['age','gender','scd'],
                       ['age','gender', 'transf'],
                       ['age','gender', 'transf', 'hct'],
                       ['age','gender', 'scd', 'transf', 'lesion_burden', 'hct']]
        """
        
        p_df = pd.DataFrame()
        p_df_name = os.path.join(out_folder, f'pvals.csv')
            
        for controller in controlling:
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
                if 'transf' in controller:
                    tabby = tabby[tabby['scd']==1] # if we're testing transfusion status, we need to evaluate SCD pts only
                
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
                
                as_df = as_df.rename(columns={'Unnamed: 0':'predictor'})
                
                droppers = ['const', 'age', 'gender']
                for d in droppers:
                    as_df = as_df[as_df['predictor'] != d]
                
                
                p_df = p_df.append(as_df, ignore_index=True)
                
                
            summary.close()
            p_df.to_csv(p_df_name)
            
            
            
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
    
    '''
    
    
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
        
        ax.set_xlabel("Average of 2 measures (cc)")
        ax.set_ylabel("Difference between 2 measures (cc)")
        
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
    
    out_of_spec = []
    
    vol_measures = ['total_vol', 'gm_vol', 'wm_vol']
    formal_measures = ['Total volume', 'Gray matter volume', 'White matter volume']
    lim_list =[[700,1500],[400,900],[200,700]]
    lim_list_bland_ex = [[800,1400],[450,900],[250,650]]
    lim_list_bland_why = [[-200,200],[-100,200],[-150,50]]
    program_pairs = list(itertools.combinations(data_dicts.keys(), 2))
    
    for lims, measure, f_measure, bl_x, bl_y in zip(lim_list, vol_measures, formal_measures, lim_list_bland_ex, lim_list_bland_why):
        fig, axs = plt.subplots(len(program_pairs), 2, figsize=(12,24))
        for (p1, p2), axrow in zip(program_pairs, axs):
            
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
            axrow[0].set_xlabel(f'{p1} (cc)')
            axrow[0].set_ylabel(f'{p2} (cc)')
            axrow[0].set_aspect('equal', 'box')
            
            fit = custom_polyfit(exes1,exes2,1)
            
            coefs = fit['polynomial']
            r2 = fit['r2']
            
            fit_exes = lims.copy()
            fit_whys = [x*coefs[0] + coefs[1] for x in fit_exes]
            axrow[0].plot(fit_exes, fit_whys, c='black')
            
            axrow[0].set_title(f'{p2} vs. {p1} ($r^2$ = {round(r2,2)})')
            
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
        
        
        
            

        