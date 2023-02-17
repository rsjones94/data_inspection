
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:34:40 2020

@author: skyjones
"""
import warnings
warnings.filterwarnings("ignore")

import os
import re
import shutil
import sys
import string
import re

import pandas as pd
from glob import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import scipy
from scipy.stats import chisquare, ttest_ind, mannwhitneyu, fisher_exact, iqr
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as mtt
import redcap
import imageio
import itertools
from skimage import measure as meas
from pingouin import ancova
import pingouin as pg
import num2words

from mpl_toolkits import mplot3d
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data
import matplotlib
import matplotlib.patheffects as pe

from fpdf import FPDF

from parse_fs_stats import parse_freesurfer_stats, parse_freesurfer_stats_lobular
from parse_sienax_stats import parse_sienax_stats

max_ctrl_age = 32
min_scd_age = 7.25


manual_excls = {
            
                'K001': ['stroke'],
                'K011': ['motion'],
                'K017': ['motion'],
                'K018': ['motion'],
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
SCD_TRANSP_K001_01interactio

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

parse = False
collate = False
quality_check = False
visualize = True
interrater = False
graphs_w_overt = False
other_plots = False

# os.path.basename(os.path.normpath(path))

###########

'''
programs = ['SPM', 'FS', 'SIENAX']
norm_columns = ['icv', 'icv', 'vscaling']
sub_outs = [os.path.join(out_folder, f'vis_{f}') for f in programs]
quality_folders = [os.path.join(f, 'quality') for f in sub_outs]
parsed_folders = [os.path.join(f, 'parsed') for f in sub_outs]
program_masters = [spm_folder, fs_folder, sienax_folder]
'''

programs = ['FS']
norm_columns = ['icv']
sub_outs = [os.path.join(out_folder, f'vis_{f}') for f in programs]
quality_folders = [os.path.join(f, 'quality') for f in sub_outs]
parsed_folders = [os.path.join(f, 'parsed') for f in sub_outs]
program_masters = [fs_folder]

for big in [sub_outs, quality_folders, parsed_folders]:
    for l in big:
        if not os.path.exists(l):
            os.mkdir(l)

"""
quality_folder =  '/Users/manusdonahue/Documents/Sky/spm_volume_visualization/quality'
parsed_folder = '/Users/manusdonahue/Documents/Sky/spm_volume_visualization/parsed'

"""

exclude_pts = list(manual_excls.keys())

def univariatize(controlling, response, independent, dataframe, equivalents=None):
    """
    Have you ever run a multiple linear regression but were only interested in
    one particular independent variable and the response variable? Frustrated
    with your inability to plot the relationship between the independent variable
    and the response variable in a clean 2d way?
    
    Try univariatizing your data! Adjusts the response variable so each data point
    is as if all the controlling variables are held constant. When you plot
    the adjusted response against the independent variable, the slope as the independent
    variable's slope in the full multiple linear regression. Warranty void if used
    with interaction terms.
    

    Parameters
    ----------
    controlling : list of str
        list of strings indicating variables to be controlled for.
    response : str
        the response variable you are adjusting.
    independent : str
        the independent variable you intend to plot.
    dataframe : pandas DataFrame
        a df of the data.
    equivalents : list of float, optional
        The equivalent value for each controlling variable. If none, then
        0 is used for all

    Returns
    -------
    adjusted : panas Series
        the response univariatized response variable.

    """
    
    if equivalents is None:
        equivalents = [0 for i in controlling]
    
    statement = f'{response} ~ {" + ".join(controlling)} + {independent}'
    #print(f'Statement is: {statement}')
    est = smf.ols(statement, data=dataframe).fit()
    
    adjusted = dataframe[response].copy()
    for c, eq in zip(controlling, equivalents):
        slope = est.params[c]
        
        c_diff = dataframe[c] - eq
        adjust_mag = c_diff * slope
        
        adjusted = adjusted - adjust_mag
        
    test_df = pd.DataFrame()
    created_var = f'{response}_adj'
    test_df[created_var] = adjusted.copy()
    test_df[independent] = dataframe[independent].copy()
    
    test_statement = f'{created_var} ~ {independent}'
    test = smf.ols(test_statement, data=test_df).fit()
    
        
    return adjusted, est, test
        

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
    blood_frac = (cbv * tissue_density)/100  # ml blood / ml tissue (dimensionless)
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
    ax.fill_between(x2, y2 + ci, y2 - ci, color=color, edgecolor=None, alpha=0.25)

    return ax, ci


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

lob_keys_o = ['bankssts',
 'caudalanteriorcingulate',
 'caudalmiddlefrontal',
 'cuneus',
 'entorhinal',
 'fusiform',
 'inferiorparietal',
 'inferiortemporal',
 'isthmuscingulate',
 'lateraloccipital',
 'lateralorbitofrontal',
 'lingual',
 'medialorbitofrontal',
 'middletemporal',
 'parahippocampal',
 'paracentral',
 'parsopercularis',
 'parsorbitalis',
 'parstriangularis',
 'pericalcarine',
 'postcentral',
 'posteriorcingulate',
 'precentral',
 'precuneus',
 'rostralanteriorcingulate',
 'rostralmiddlefrontal',
 'superiorfrontal',
 'superiorparietal',
 'superiortemporal',
 'supramarginal',
 'frontalpole',
 'temporalpole',
 'transversetemporal',
 'insula']

    
subcort_keys_o = ['Left-Lateral-Ventricle',
 'Left-Inf-Lat-Vent',
 'Left-Cerebellum-White-Matter',
 'Left-Cerebellum-Cortex',
 'Left-Thalamus',
 'Left-Caudate',
 'Left-Putamen',
 'Left-Pallidum',
 '3rd-Ventricle',
 '4th-Ventricle',
 'Brain-Stem',
 'Left-Hippocampus',
 'Left-Amygdala',
 'CSF',
 'Left-Accumbens-area',
 'Left-VentralDC',
 'Left-vessel',
 'Left-choroid-plexus',
 'Right-Lateral-Ventricle',
 'Right-Inf-Lat-Vent',
 'Right-Cerebellum-White-Matter',
 'Right-Cerebellum-Cortex',
 'Right-Thalamus',
 'Right-Caudate',
 'Right-Putamen',
 'Right-Pallidum',
 'Right-Hippocampus',
 'Right-Amygdala',
 'Right-Accumbens-area',
 'Right-VentralDC',
 'Right-vessel',
 'Right-choroid-plexus',
 '5th-Ventricle',
 'WM-hypointensities',
 'Left-WM-hypointensities',
 'Right-WM-hypointensities',
 'non-WM-hypointensities',
 'Left-non-WM-hypointensities',
 'Right-non-WM-hypointensities',
 'Optic-Chiasm',
 'CC_Posterior',
 'CC_Mid_Posterior',
 'CC_Central',
 'CC_Mid_Anterior',
 'CC_Anterior']


lob_keys = [i.replace('-', '_') for i in lob_keys_o]
subcort_keys = [i.replace('-', '_') for i in subcort_keys_o]

lob_keys = [re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), i) for i in lob_keys]
subcort_keys = [re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), i) for i in subcort_keys]

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
                    print(f'No ppleted SIENAX report for {mr} ({sienax_report})')
                
                
        
        elif program == 'FS':
            
            folders = np.array(glob(os.path.join(master, '*/'))) # list of all folders
            for i, f in enumerate(folders):
                
                mr = os.path.basename(os.path.normpath(f))
                
                print(f'\nParsing {mr} ({i+1} of {len(folders)})')
                
                stats_file = os.path.join(fs_folder, mr, 'stats', 'aseg.stats')
                lobular_file_lh = os.path.join(fs_folder, mr, 'stats', 'lh.aparc.stats')
                lobular_file_rh = os.path.join(fs_folder, mr, 'stats', 'rh.aparc.stats')
                
                parsed_file = os.path.join(parsed_folder, f'{mr}.csv')
                
                try:
                    odf = parse_freesurfer_stats(stats_file)
                except FileNotFoundError:
                    print(f'No completed Freesurfer folder for {mr} ({stats_file})')
                    continue
                
                try:
                    odf2 = parse_freesurfer_stats_lobular(lobular_file_lh)
                except FileNotFoundError:
                    print(f'No completed Freesurfer folder for {mr} ({lobular_file_lh})')
                    continue
                
                try:
                    odf3 = parse_freesurfer_stats_lobular(lobular_file_rh)
                except FileNotFoundError:
                    print(f'No completed Freesurfer folder for {mr} ({lobular_file_rh})')
                    continue
                
                lob_df = pd.DataFrame()
                
                for i,row in odf2.iterrows():
                    the_row = row.copy()
                    
                    try:
                        the_row['value'] = float(odf3[odf3['short']==the_row['short']]['value'])
                    except TypeError:
                        continue
                    
                    lob_df = lob_df.append(the_row, ignore_index=True)
                
                #odf_lob = odf2.copy()
                #odf_lob['value'] = odf2['value'] + odf3['value']
                
                odf_out = odf.append(lob_df, ignore_index=True)
                
                odf_out.to_csv(parsed_file, index=False)
            

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
                  'sid':None,
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
                  'cao2':None,
                  'oef':None,
                  'shunt_score':None,
                  'wm_oxdel':None,
                  'gm_oxdel':None,
                  'age':None,
                  'stroke_silent':None,
                  'stroke_overt':None,
                  'sci':None,
                  'transf':None,
                  'race':None,
                  'scd':None,
                  'is_ss':None,
                  'anemia':None,
                  'control':None,
                  'lesion_burden':None,
                  'lesion_count':None,
                  'lesion_burden_log':None,
                  'lesion_count_log':None,
                  'lesion_burden_log_nz':None,
                  'lesion_count_log_nz':None,
                  'gender':None,
                  'intracranial_stenosis':None,
                  'hydroxyurea':None,
                  'hemoglobin':None,
                  'hemoglobin_s_frac':None,
                  'hemoglobin_nons_frac':None,
                  'healthy_hemoglobin':None,
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
                  'excl_missing_wm_cbf':None,
                  'excl_non_ss':None,
                  'excl_missing_bloodwork':None,
                  'excl_weird_hbs':None,
                  'excl_lobcalc':None,
                  'excl_ctrl_old':None,
                  'excl_scd_young':None}
    

    
    for lk in lob_keys:
        blank_dict[f'{lk}_vol'] = None
        blank_dict[f'{lk}_vol_adj'] = None
        
    for sk in subcort_keys:
        blank_dict[f'{sk}_vol'] = None
        blank_dict[f'{sk}_vol_adj'] = None
    
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
            
            working['sid'] = study_id
            
            
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
            
            working['cao2'] = float(cands.iloc[0][f'mr1_cao2'])
            
            try:
                working['oef'] = float(cands.iloc[0][f'mr1_whole_oef_mean'])/100
            except ValueError:
                pass
            
                        
            try:
                working['shunt_score'] = int(cands.iloc[0][f'outcome_avshunt_consensus'])
                if working['shunt_score'] < 0:
                    working['shunt_score'] = None
            except ValueError:
                pass
            
            
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
            
            
            if working['scd'] == 1 and working['age'] < min_scd_age:
                working['excl_scd_young'] = 1
                working['exclude'] = 1
                
            if working['scd'] == 0 and working['age'] > max_ctrl_age:
                working['excl_ctrl_old'] = 1
                working['exclude'] = 1
            
            
            
            if working['scd'] == 1:
                try:
                    ss_status = int(cands.iloc[0][f'enroll_sca_incl_type'])
                except ValueError:
                    ss_status = 1
                    
                if ss_status == 1:
                    working['is_ss'] = 1
                else:
                    working['is_ss'] = 0
                    #working['excl_non_ss'] = 1
                    #working['exclude'] = 1
            
            # additional 
            
            try:
                working['intracranial_stenosis'] = int(cands.iloc[0]['mra1_ic_stenosis_drp'])
            except ValueError:
                working['intracranial_stenosis'] = 0
            
            try:
                working['hydroxyurea'] = int(cands.iloc[0]['current_med_hu'])
            except ValueError:
                working['hydroxyurea'] = 0
            
            try:
                working['transf'] = int(cands.iloc[0]['reg_transf'])
            except ValueError:
                working['transf'] = 0
            
            try:
                working['hemoglobin_s_frac'] = float(cands.iloc[0]['blood_draw_hbs1'])/100
                working['hemoglobin_nons_frac'] = 1-working['hemoglobin_s_frac']
                
                
                if working['hemoglobin_s_frac'] < 0.5:
                    #working['exclude'] = 1
                    #working['excl_weird_hbs'] = 1
                    pass
                    
            except ValueError:
                if working['is_ss'] == 1:
                    working['exclude'] = 1
                    working['excl_missing_bloodwork'] = 1
                    working['hemoglobin_s_frac'] = None
                    working['hemoglobin_nons_frac'] = None
                else:
                    working['hemoglobin_s_frac'] = 0
                    working['hemoglobin_nons_frac'] = 1
        
            try:
                working['hemoglobin'] = float(cands.iloc[0]['blood_draw_hgb1'])
                
                if working['hemoglobin_nons_frac'] is not None:
                    working['healthy_hemoglobin'] = working['hemoglobin_nons_frac'] * working['hemoglobin']
                else:
                    working['healthy_hemoglobin'] = working['hemoglobin']
            except ValueError:
                working['hemoglobin'] = None
                working['healthy_hemoglobin'] = None
                
            
            try:
                working['bmi'] = float(cands.iloc[0]['bmi'])
            except ValueError:
                working['bmi'] = None
            
            try:
                working['pulseox'] = float(cands.iloc[0]['mr1_pulse_ox_result'])
            except ValueError:
                 working['pulseox'] = None
            
            try:
                working['diabetes'] = int(cands.iloc[0]['mh_rf_diab'])
            except ValueError:
                working['diabetes'] = None
            
            try:
                working['high_cholesterol'] = int(cands.iloc[0]['mh_rf_high_cholest'])
            except ValueError:
                working['high_cholesterol'] = None
            
            try:
                working['coronary_art_disease'] = int(cands.iloc[0]['mh_rf_cad'])
            except ValueError:
                working['coronary_art_disease'] = None
            
            try:
                working['smoker'] = int(cands.iloc[0]['mh_rf_act_smoke'])
            except ValueError:
                working['smoker'] = None            
            

            
            
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
                
                try:
                    
                    
                    for sk, sko in zip(subcort_keys, subcort_keys_o):
                        working[f'{sk}_vol'] = parsed_csv[parsed_csv['long']==sko].iloc[0]['value']
                        
                    
                    for lk, lko in zip(lob_keys, lob_keys_o):
                        working[f'{lk}_vol'] = parsed_csv.loc[lko]['value']
                    
                    

                        
                except KeyError as ke:
                    print(f'Lobular calculation issue: {ke}')
                    working['exclude'] = 1
                    working['excl_lobcalc'] = 1

            else:
                raise Exception('you probably spelled something wrong')
                
            try:
                working['gm_cbf'] = float(cands.iloc[0]['mr1_recalc_gm_cbf'])
                working['gm_vol'] = adjust_for_perfusion(working['gm_vol_unadj'], working['gm_cbf'])
                working['gm_oxdel'] = working['gm_cbf'] * working['cao2'] / 100 # mL 02 / min / 100g tissue
            except ValueError:
                working['exclude'] = 1
                working['excl_missing_gm_cbf'] = 1
               
            try:
                working['wm_cbf'] = float(cands.iloc[0]['mr1_recalc_wm_cbf'])
                working['wm_vol'] = adjust_for_perfusion(working['wm_vol_unadj'], working['wm_cbf'])
                working['wm_oxdel'] = working['wm_cbf'] * working['cao2'] / 100 # mL 02 / min / 100g tissue
            except ValueError:
                working['exclude'] = 1
                working['excl_missing_wm_cbf'] = 1
                
            if working['wm_vol'] and working['gm_vol']:
                working['total_vol'] = working['wm_vol'] + working['gm_vol']
                
            if prog == 'FS' and working['excl_lobcalc'] != 1:
                    
                '''
                white_structs =  []
                grey_structs = ['subfrontal', 'frontalpole', 'caudalmiddlefrontal', 'rostralmiddlefrontal', 'superiorfrontal', 'hippocampus']
                
                for struct in white_structs:
                    working[f'{struct}_vol'] = adjust_for_perfusion(working[f'{struct}_unadj'], working['wm_cbf'])
                    
                for struct in grey_structs:
                    working[f'{struct}_vol'] = adjust_for_perfusion(working[f'{struct}_unadj'], working['gm_cbf'])
                '''
                pass
                
                
            
            
            
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
                    
                    '''
                    if working['lesion_burden'] > 8:
                        working['exclude'] = 1
                        working['excl_excessive_burden'] = 1
                    '''
                    
                except FileNotFoundError:
                    if working['exclude'] == 0:
                        print(f'please make a mask for {pt_name}')
                        missing_masks.append(pt_name)
                        raise Exception
            else:
                working['lesion_burden'] = 0
                working['lesion_count'] = 0
                
                
            out_df = out_df.append(working, ignore_index=True)
            
                                
        for cn in ['lesion_burden', 'lesion_count']:
            nozeroes = [i for i in out_df[cn] if i > 0]
            minv = np.min(nozeroes)
            adder = minv/2
            
            out_df[f'{cn}_log'] = out_df[f'{cn}'] + adder
            out_df[f'{cn}_log'] = np.log10(out_df[f'{cn}_log'])
            
            baselog = np.log10(out_df[cn])
            
            corrected = [i  if i != -np.inf else None for i in baselog]
            out_df[f'{cn}_log_nz'] = corrected
        
        out_df = out_df[blank_dict.keys()]
        out_csv = os.path.join(out_folder, f'collated.csv')
        out_df.to_csv(out_csv, index=False)
        
        # now make the demographic table
        
        the_cols = ['age', 'race', 'gender', 'sci', 'transf', 'intracranial_stenosis', 'hydroxyurea',
                    'hemoglobin', 'bmi', 'diabetes', 'high_cholesterol', 'coronary_art_disease', 'smoker', 'cao2', 'hemoglobin_s_frac', 'pulseox',
                    'wm_vol', 'gm_vol', 'total_vol', 'wm_vol_unadj', 'gm_vol_unadj', 'total_vol_unadj', 'lesion_count', 'lesion_burden']
        
        
                
        categorical = ['race', 'gender', 'sci', 'transf', 'intracranial_stenosis', 'hydroxyurea', 'diabetes',
                       'high_cholesterol', 'coronary_art_disease', 'smoker']
        categorical_names = ['Black race', 'Male sex', 'Has SCI', 'Regular blood transfusions', 'Intracranial stenosis >50%',
                             'Hydroxyurea therapy', 'Diabetes mellitus', 'Hypercholesterolemia',
                             'Coronary artery disease', 'Smoking currently']
        
        cont = ['age', 'cao2', 'hemoglobin', 'hemoglobin_s_frac', 'pulseox', 'bmi', 'wm_vol', 'gm_vol', 'total_vol', 'wm_vol_unadj', 'gm_vol_unadj', 'total_vol_unadj', 'lesion_count', 'lesion_burden',
                'gm_cbf', 'wm_cbf']
        cont_names = ['Age at MRI', 'CaO2 (mL/dL)', 'Hemoglobin, g/dL', 'Hemoglobin S fraction', 'SaO2', 'Body mass index, kg/m2', 'White matter volume, mL', 'Gray matter volume, mL', 'Total brain volume, mL',
                      'White matter volume (unadjusted), mL', 'Gray matter volume (unadjusted), mL', 'Total brain volume (unadjusted), mL', 'Lesion count', 'Lesion burden, mL',
                      'GM CBF, ml/100g/min', 'WM CBF, ml/100g/min']
        
        
        for colu in categorical:
            if colu not in the_cols:
                the_cols.append(colu)
        for colu in cont:
            if colu not in the_cols:
                the_cols.append(colu)
        
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
            
            if col not in ['smoker', 'diabetes']: # note that sometimes race needs to be included here due to small cell count
                chi, pval = chisquare([freq_true, freq_false], [expect_true, expect_false])
            else:
                contingency_table = [[freq_true, freq_false],[ctrl_true, ctrl_false]]
                chi, pval = fisher_exact(contingency_table)
            
            dic = {f'SCD (n={len(scd_df)})': f'{scd_d} ({scd_perc}%)', f'Control (n={len(ctrl_df)})': f'{ctrl_d} ({ctrl_perc}%)', 'p-value':pval}
            print(dic)
            ser = pd.Series(dic, name=name)
            table_1 = table_1.append(ser)
            
            if col == 'race':
                sys.exit()
            
        for col,name in zip(cont, cont_names):
            
            scd_ser = scd_df[col].dropna()
            if col in ['lesion_count', 'lesion_burden']:
                scd_ser = scd_ser[scd_ser > 0]
            ctrl_ser = ctrl_df[col].dropna()
            
            
            figure, axs = plt.subplots(1, 2, figsize=(8, 8))
            axs[0].hist(scd_ser, bins=12)
            axs[1].hist(ctrl_ser, bins=12)
            axs[0].set_title('SCD')
            axs[1].set_title('Controls')
            
            axs[0].set_xlabel(name)
            axs[1].set_xlabel(name)
            
            axs[0].set_ylabel('Count')
            axs[1].set_ylabel('Count')
            
            hisplot_name = os.path.join(out_folder, f'hisplot_{col}.png')
            plt.savefig(hisplot_name)
            

            
            if col not in ['pulseox', 'age', 'hemoglobin_s_frac', 'lesion_count', 'lesion_burden']:
                scd_d = np.mean(scd_ser)
                ctrl_d = np.mean(ctrl_ser)
                
                t, pval = ttest_ind(scd_ser, ctrl_ser)
                #we want sd
                stat = 'sd'
                scd_stat = round(np.std(scd_ser),2)
                ctrl_stat = round(np.std(ctrl_ser),2)
                
            else:
                scd_d = np.median(scd_ser)
                ctrl_d = np.median(ctrl_ser)
                
                t, pval = mannwhitneyu(scd_ser, ctrl_ser)
                # we want iqr
                stat = 'iqr'
                scd_stat = round(scipy.stats.iqr(scd_ser),2)
                ctrl_stat = round(scipy.stats.iqr(ctrl_ser),2)
            
            dic = {f'SCD (n={len(scd_df)})': f'{round(scd_d,2)} ({stat}={scd_stat})', f'Control (n={len(ctrl_df)})': f'{round(ctrl_d,2)} ({stat}={ctrl_stat})', 'p-value':pval}
            print(dic)
            ser = pd.Series(dic, name=name)
            table_1 = table_1.append(ser)
        
        

        
        table_1_name = os.path.join(out_folder, f'table1.csv')
        table_1.to_csv(table_1_name)
            
            
            
        
            
            
        
        
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
            
        fs_collated = '/Users/manusdonahue/Documents/Sky/t1_volumizers/vis_FS/collated.csv'
        fs_collated_df = pd.read_csv(fs_collated)
        excluded_list = list(fs_collated_df[fs_collated_df['exclude']==1]['mr_id'])
        
        if mr in excluded_list:
            
            excluded_out = os.path.join(excluded_folder, f'{mr}_composite.pdf')
            shutil.move(pdf_out, excluded_out)
    
            
            
    
        
if visualize:
    matplotlib.rcdefaults()
    print('Visualizing')
    brain_vol_df = pd.read_csv(brain_vol_csv)
    
    
    
    
    alphabet = list(string.ascii_uppercase)
    
    #indies = ['hemoglobin', 'gm_cbf', 'lesion_burden_log_nz']
    #responses = ['gm_vol_unadj', 'wm_vol_unadj']
    
    indies = ['hemoglobin', 'gm_cbf', 'lesion_burden_log_nz']
    responses = ['gm_vol_unadj', 'wm_vol_unadj']
    
    for i, (prog, norm_name, out_folder) in enumerate(zip(programs, norm_columns, sub_outs)):
        
        figname = os.path.join(out_folder_orig, f'manuscript_scatters_withadjustments_{prog}.png')
        fig, axs = plt.subplots(len(indies)+1, len(responses), figsize=(4*len(responses),5*len(indies)+1))
        letters = alphabet[:len(indies)+1]
    
        collated_csv = os.path.join(out_folder, 'collated.csv')
        clean_table = pd.read_csv(collated_csv, index_col='mr_id')
        clean_table = clean_table[clean_table['exclude'] != 1]
        
    
        controlling = ['age', 'gender', norm_name]
        eqs = [25, 0, 1400]
        
        indie='scd'
        # controlled box plots
        for g, (ax_row, let) in enumerate(zip(axs[:1], letters[:1])):
            for j, (resp, ax) in enumerate(zip(responses, ax_row)):
                
                use_table = clean_table.copy()
                
                keeper_cols = controlling.copy()
                keeper_cols.append(resp)
                keeper_cols.append(indie)
                
                sub_table = use_table[keeper_cols].dropna()
                
                adj_response, est, tester = univariatize(controlling, resp, indie, sub_table, equivalents=eqs)
                
                use_table['resp'] = adj_response
                
                scds = use_table[use_table['scd']==1]
                hcs = use_table[use_table['scd']==0]
                
                scd_resp = scds['resp']
                hc_resp = hcs['resp']
                
                
                data_blob = [hc_resp, scd_resp]
                width = 0.2
                
                exers = [1,1.6]
                medianprops = dict(color="black",linewidth=1.5)
                boxprops = dict(color="black",linewidth=1.5)
                bp = ax.boxplot([hc_resp, scd_resp], positions=exers, patch_artist=Trne, widths=width, showfliers=False, medianprops=medianprops, boxprops=boxprops, zorder=10)
                ax.set_xlim(.7, 1.9)
                
                boxes = bp['boxes']
                        
                        
                
                colors = ['red', 'blue']
                for b,co in zip(boxes, colors):
                    b.set_facecolor(co)
                    b.set_alpha(0.4)
                
                mover = [-width/2, width/2]
                labels = ['HC', 'SCD']
                markers = ['o', '^']
                the_arms = ['Affected arm', 'Contralateral arm']
                for k, (the_data, move, co, ta, mk) in enumerate(zip(data_blob, mover, colors, labels, markers)):
                    data = the_data
                    
                    coverage = 1.1
                    exes = exers[k] + np.random.random(data.size) * width*coverage - (width*coverage/2)
                    
                    if i==0:
                        labeler=ta
                    else:
                        labeler=None
                    
                    ax.scatter(exes, data, color=co, alpha=0.15, edgecolor='black', marker=mk, label=None, zorder=5, s=20)
                    ax.scatter(exes, data, color='None', alpha=0.4, edgecolor='black', marker=mk, zorder=5, s=20)
                    
                
                ax.set_xticks(exers)
                ax.set_xticklabels(labels)
                #ax.legend()
                
                ax.set_ylim(-0.25,.25)
                
                ax.plot(ax.get_xlim(),[0,0],c='red')
                
                    
                if 'vol' in resp:
                    if prog == 'FS':
                        proname = 'FreeSurfer'
                    else:
                        proname = prog
                    
                    if 'gm' in resp:
                        ao = 'GM'
                        ax.set_ylim(475,725)
                    elif 'wm' in resp:
                        ao = 'WM'
                        ax.set_ylim(350,550)
                    
                    
                    ax.set_ylabel(f'{ao} volume (mL)\n(controlled for age, gender, ICV)')
                else:
                    ax.set_ylabel(col)
                
                
                
                if j == 0:
                    print(f'PRINT THAT LETTER: {let}')
                    ax.text(-.1, 1.1, let, horizontalalignment='center',
                         verticalalignment='center', transform=ax.transAxes, size=24, fontweight='bold')
        
        
    
        
        # controlled univariate plots
        
        for g, (indie, ax_row, let) in enumerate(zip(indies, axs[1:], letters[1:])):
            
            
            #use_table = clean_table[clean_table['scd']==1]
            use_table = clean_table
            
            subcolor='blue'
            mark='^'
            icolor=subcolor
            for j, (resp, ax) in enumerate(zip(responses, ax_row)):
                
                keeper_cols = controlling.copy()
                keeper_cols.append(resp)
                keeper_cols.append(indie)
                
                allcols = keeper_cols
                allcols.append('scd')
                
                f_table = use_table[allcols].dropna()
                sub_table = f_table[keeper_cols]
                
                adj_response, est, tester = univariatize(controlling, resp, indie, sub_table, equivalents=eqs)
                
                                    
                exes = sub_table[indie]
                whys = adj_response
                zees = f_table['scd']
                
                hold = [(x,y,z) for x,y,z in zip(exes,whys,zees) if not np.isnan(x) and not np.isnan(y)]
                
                exes = [x for x,y,z in hold]
                whys = [y for x,y,z in hold]
        
        
                ## BOOT STRAPPING. courtesy of pylang from stackoverflow
                
                x, y = exes, whys
                # Modeling with Numpy
                def equation(a, b):
                    ''''Return a 1D polynomial.'''
                    return np.polyval(a, b)
                # Data
                
                p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
                perr = np.sqrt(np.diag(cov))    
                the_int, the_int_err = (p[1]), (perr[1])*1.96
                the_slope, the_slope_err = (p[0]), (perr[0])*1.96
                
                
                labeler = f'Slope: ${the_slope:#.3g}\pm{the_slope_err:#.3g}$\nIntercept: ${the_int:#.3g}\pm{the_int_err:#.3g}$'
                #labeler = prop
                
                mini_df = pd.DataFrame()
                mini_df['exes'] = exes
                mini_df['whys'] = whys
                mini_df['zees'] = list(zees)
                for num, fill_color, marker_shape, sublabel in zip([0,1], ['red', 'blue'], ['o', '^'], ['HC', 'SCD']):
                    
                    cut_mini_df = mini_df[mini_df['zees']==num]
                    
                    if len(cut_mini_df) == 0:
                        continue
                    
                    bex = cut_mini_df['exes']
                    bwhy = cut_mini_df['whys']
                    ax.plot(
                        bex, bwhy, "o", color="#b9cfe7", markersize=4,
                        markeredgewidth=1, markeredgecolor="black", markerfacecolor=fill_color,
                        marker=marker_shape, alpha=0.3, label=sublabel
                        )
                    ax.plot(
                        bex, bwhy, "o", color="#b9cfe7", markersize=4,
                        markeredgewidth=1, markeredgecolor="black", markerfacecolor="None",
                        marker=marker_shape
                        )
                    
                
                #p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
                #perr = np.sqrt(np.diag(cov))                               # standard-deviation estimates for each coefficient
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
                ax.plot(x, y_model, "-", color='gray', linewidth=1.5, alpha=0.25, label=labeler)
                
                xij = np.arange(min(x),max(x)+0.01, 0.01)
                yij = equation(p, xij)
                #ax.plot(xij[1:], yij[1:], "-", color=icolor, linewidth=1.5, alpha=0.25) 
                
                x2 = np.linspace(np.min(x), np.max(x), 100)
                y2 = equation(p, x2)
                
                # Confidence Interval (select one)
                nax, the_ci = plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax, color='gray')
                #plot_ci_bootstrap(x, y, resid, ax=ax)
                
                # now calculate the numerical 95% conf interval for slope and intercept
                #ci_high = y2+the_ci
                #ci_low = y2-the_ci
                
                
                
                
                # Prediction Interval
                pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))  
                ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
                ax.plot(x2, y2 - pi, "--", color='gray', alpha=0.3)#, label="95% Prediction Limits")
                ax.plot(x2, y2 + pi, "--", color='gray', alpha=0.3)
                
                
                
                ax.legend(fontsize=6)
                
                if indie == 'hemoglobin':
                    ax.set_xlim(6,17)
                    ax.set_xlabel('Hemoglobin (g/dL)')
                elif indie == 'lesion_burden_log_nz':
                    ax.set_xlabel('Log$_{10}$ lesion burden')
                elif indie == 'gm_cbf':
                    ax.set_xlabel('GM CBF (ml/100g/min)')
                else:
                    ax.set_xlabel(indie)
                    
                if 'vol' in resp:
                    if prog == 'FS':
                        proname = 'FreeSurfer'
                    else:
                        proname = prog
                    
                    if 'gm' in resp:
                        ao = 'GM'
                        ax.set_ylim(475,725)
                    elif 'wm' in resp:
                        ao = 'WM'
                        ax.set_ylim(350,550)
                    
                    ax.set_ylabel(f'{ao} volume (mL)\n(controlled for age, gender, ICV)')
                else:
                    ax.set_ylabel(col)
                
                
                    
                if j == 0:
                    print(f'PRINT THAT LETTER: {let}')
                    ax.text(-.1, 1.1, let, horizontalalignment='center',
                         verticalalignment='center', transform=ax.transAxes, size=24, fontweight='bold')
                
                
                

                
            
            plt.tight_layout()
            plt.savefig(figname, dpi=400)
            
            
    
    #interest = ['gm_vol_dhdj', 'wm_vol_unadj', 'superiorfrontal_unadj', 'rostralmiddlefrontal_unadj', 'caudalmiddlefrontal_unadj', 'frontalpole_unadj', 'hippocampus_unadj']
    #interest = ['gm_vol', 'wm_vol', 'subfrontal_vol', 'hippocampus_vol', 'gm_cbf']
    #interest = ['gm_vol', 'wm_vol', 'gm_cbf']
    interest = ['gm_vol_unadj', 'wm_vol_unadj']
    #pred_vars = ['age', 'cao2', 'lesion_count', 'lesion_burden', 'hemoglobin']
    #pred_vars = ['age', 'hemoglobin', 'lesion_burden_log_nz']
    pred_vars = ['hemoglobin', 'lesion_burden_log_nz']
    
    
    
    for i, (prog, norm_name, out_folder) in enumerate(zip(programs, norm_columns, sub_outs)):
        figname = os.path.join(out_folder_orig, f'manuscript_scatters_{prog}.png')
        fig, axs = plt.subplots(len(pred_vars), len(interest), figsize=(4*len(interest),4*len(pred_vars)))
        letters = alphabet[:len(pred_vars)]
        
        
        for g, (pred_var, ax_row, let) in enumerate(zip(pred_vars, axs, letters)):
            
    
            collated_csv = os.path.join(out_folder, 'collated.csv')
            clean_table = pd.read_csv(collated_csv, index_col='mr_id')
            
            clean_table = clean_table[clean_table['exclude'] != 1]
            
            clean_table['normal_control'] = [all([i, not j]) for i,j in zip(clean_table['control'], clean_table['sci'])]        
            clean_table['sci_control'] = [all([i, j]) for i,j in zip(clean_table['control'], clean_table['sci'])]   
            
            
            clean_table['normal_scd'] = [all([i, not j]) for i,j in zip(clean_table['scd'], clean_table['sci'])]        
            clean_table['sci_scd'] = [all([i, j]) for i,j in zip(clean_table['scd'], clean_table['sci'])]
            
            pt_type = ['control', 'scd']
            pt_type_proper = ['Control', 'SCD']
            exprs = [clean_table[pt] == 1 for pt in pt_type]
            subdfs = [clean_table[expr] for expr in exprs]
            
            for j, (col, ax) in enumerate(zip(interest, ax_row)):
                
                subcolors = ['red', 'blue']
                int_colors = ['red', 'blue']
                markers = ['o', '^']
                
                all_exes = []
                all_whys = []
                for subcolor, subd, icolor, patient_type, mark, prop in zip(subcolors, subdfs, int_colors, pt_type, markers, pt_type_proper):
                        
                    if patient_type == 'control' and pred_var in ['lesion_burden', 'lesion_count', 'hemoglobin_s_frac', 'sci', 'lesion_count_log', 'lesion_burden_log', 'lesion_count_log_nz', 'lesion_burden_log_nz']:
                        continue
                    
                    if patient_type == 'control' and col in ['sci', 'lesion_count', 'lesion_burden', 'lesion_count_log', 'lesion_burden_log', 'lesion_count_log_nz', 'lesion_burden_log_nz']:
                        continue
                    
                                
                    if pred_var == 'hemoglobin_s_frac':
                        # if we're looking at hbs frac, need to remove transfused pts
                        use_table = subd[subd['transf']==0]
                    else:
                        use_table = subd.copy()
                    
                    print(f'pred_var: {pred_var}, col: {col}')
                    
                    exes = use_table[pred_var]
                    whys = use_table[col]
                    
                    hold = [(x,y) for x,y in zip(exes,whys) if not np.isnan(x) and not np.isnan(y)]
                    
                    exes = [x for x,y in hold]
                    whys = [y for x,y in hold]
                    
                    if pred_var == 'lesion_burden':
                        ax.set_xscale('log')
                    if col == 'lesion_burden':
                        #ax.set_yscale('log')
                        pass
            
            
            
                    ## BOOT STRAPPING. courtesy of pylang from stackoverflow
                    
                    x, y = exes, whys
                        
                    all_exes.extend(x)
                    all_whys.extend(y)
                    
                    
                    # Modeling with Numpy
                    def equation(a, b):
                        ''''Return a 1D polynomial.'''
                        return np.polyval(a, b)
                    # Data
                    
                    p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
                    perr = np.sqrt(np.diag(cov))    
                    the_int, the_int_err = (p[1]), (perr[1])*1.96
                    the_slope, the_slope_err = (p[0]), (perr[0])*1.96
                    
                    
                    #labeler = f'{prop}\n\tSlope: ${the_slope:#.3g}\pm{the_slope_err:#.3g}$\n\tIntercept: ${the_int:#.3g}\pm{the_int_err:#.3g}$'
                    labeler = prop
                    
                    ax.plot(
                        x, y, "o", color="#b9cfe7", markersize=4,
                        markeredgewidth=1, markeredgecolor="black", markerfacecolor=subcolor,
                        marker=mark, alpha=0.3, label=labeler
                        )
                    ax.plot(
                        x, y, "o", color="#b9cfe7", markersize=4,
                        markeredgewidth=1, markeredgecolor="black", markerfacecolor="None",
                        marker=mark
                        )
                    try:
                        #p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
                        #perr = np.sqrt(np.diag(cov))                               # standard-deviation estimates for each coefficient
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
                        #ax.plot(x, y_model, "-", color=icolor, linewidth=1.5, alpha=0.25)
                        
                        xij = np.arange(min(x),max(x)+0.01, 0.01)
                        yij = equation(p, xij)
                        #ax.plot(xij[1:], yij[1:], "-", color=icolor, linewidth=1.5, alpha=0.25) 
                        
                        x2 = np.linspace(np.min(x), np.max(x), 100)
                        y2 = equation(p, x2)
                        
                        # Confidence Interval (select one)
                        #nax, the_ci = plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax, color=icolor)
                        #plot_ci_bootstrap(x, y, resid, ax=ax)
                        
                        # now calculate the numerical 95% conf interval for slope and intercept
                        #ci_high = y2+the_ci
                        #ci_low = y2-the_ci
                        
                        
                        
                        
                        # Prediction Interval
                        pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))  
                        #ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
                        #ax.plot(x2, y2 - pi, "--", color=icolor, alpha=0.3)#, label="95% Prediction Limits")
                        #ax.plot(x2, y2 + pi, "--", color=icolor, alpha=0.3)
                        
                    except np.linalg.LinAlgError:
                        print(f'Linear algebra error, likely due to singular matrix ({pred_var} vs. {col})')
                        print(exes,whys)
                        pass
                    
                x, y = all_exes, all_whys
                # Modeling with Numpy
                def equation(a, b):
                    ''''Return a 1D polynomial.'''
                    return np.polyval(a, b)
                # Data
                
                p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
                perr = np.sqrt(np.diag(cov))    
                the_int, the_int_err = (p[1]), (perr[1])*1.96
                the_slope, the_slope_err = (p[0]), (perr[0])*1.96
                
                
                labeler = f'Slope: ${the_slope:#.3g}\pm{the_slope_err:#.3g}$\nIntercept: ${the_int:#.3g}\pm{the_int_err:#.3g}$'
                #labeler = prop
                try:
                    #p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
                    #perr = np.sqrt(np.diag(cov))                               # standard-deviation estimates for each coefficient
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
                    #ax.plot(x, y_model, "-", color=icolor, linewidth=1.5, alpha=0.25)
                    
                    xij = np.arange(min(x),max(x)+0.01, 0.01)
                    yij = equation(p, xij)
                    ax.plot(xij[1:], yij[1:], "-", color='gray', linewidth=1.5, alpha=0.5, label=labeler) 
                    
                    x2 = np.linspace(np.min(x), np.max(x), 100)
                    y2 = equation(p, x2)
                    
                    # Confidence Interval (select one)
                    nax, the_ci = plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax, color='gray')
                    #plot_ci_bootstrap(x, y, resid, ax=ax)
                    
                    # now calculate the numerical 95% conf interval for slope and intercept
                    ci_high = y2+the_ci
                    ci_low = y2-the_ci
                    
                    
                    
                    
                    # Prediction Interval
                    pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))  
                    ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
                    ax.plot(x2, y2 - pi, "--", color='gray', alpha=0.3)#, label="95% Prediction Limits")
                    ax.plot(x2, y2 + pi, "--", color='gray', alpha=0.3)
                    
                except np.linalg.LinAlgError:
                    print(f'Linear algebra error, likely due to singular matrix ({pred_var} vs. {col})')
                    print(exes,whys)
                    pass
                
                    #ax.scatter(exes, whys, color=subcolor, alpha=0.2, s=4, label=patient_type, marker=mark)
                    #ax.scatter(exes, whys, color=subcolor, alpha=0.2, s=4, label=patient_type, marker=mark)
                ax.legend(fontsize=6)
                

                #if 'gm_vol' in col or 'wm_vol' in col:
                if 'vol' in col:
                    if prog == 'FS':
                        proname = 'FreeSurfer'
                    else:
                        proname = prog
                    
                    if 'gm' in col:
                        ao = 'GM'
                        ax.set_ylim(200,850)
                    elif 'wm' in col:
                        ao = 'WM'
                        ax.set_ylim(200,850)
                    elif 'subfrontal' in col:
                        ao = 'Subfrontal lobe'
                    elif 'hippocampus' in col:
                        ao = 'Hippocampus'
                    
                    ax.set_ylabel(f'{ao} volume (mL)')
                    
                elif 'gm_cbf' in col:
                    ax.set_ylabel(f'Gray matter CBF (ml/100g/min)')
                    ax.set_ylim(25,130)
                else:
                    ax.set_ylabel(col)
                

                
                if pred_var == 'age':
                    ax.set_xlim(0,50)
                    ax.set_xlabel('Age (years)')
                elif pred_var == 'hct':
                    ax.set_xlim(0.15,0.55)
                    ax.set_xlabel('Hematocrit')
                elif pred_var == 'cao2':
                    ax.set_xlim(5,25)
                    ax.set_xlabel('Arterial oxygen content (mL O2 / dL blood)')
                elif pred_var == 'lesion_burden':
                    #ax.set_xlim(0,6e4)
                    ax.set_xlabel('Lesion burden (mL)')
                elif pred_var == 'lesion_count':
                    #ax.set_xlim(0,6e4)
                    ax.set_xlabel('Lesion count')
                elif pred_var == 'hemoglobin_s_frac':
                    ax.set_xlabel('Hemoglobin S fraction')
                    ax.set_xlim(0.5,1)
                elif pred_var == 'wm_oxdel':
                    ax.set_xlabel('WM art. ox. delivery (mL O2 / minute / 100g tissue)')
                elif pred_var == 'wm_cbf':
                    ax.set_xlabel('WM CBF (mL blood / minute / 100g tissue)')
                elif pred_var == 'gm_oxdel':
                    ax.set_xlabel('GM art. ox. delivery (mL O2 / minute / 100g tissue)')
                elif pred_var == 'gm_cbf':
                    ax.set_xlabel('GM CBF (mL blood / minute / 100g tissue)')
                elif pred_var == 'oef':
                    ax.set_xlabel('OEF')
                elif pred_var == 'hemoglobin':
                    ax.set_xlabel('Hemoglobin (g/dL)')
                    #ax.set_xlim(0,1)
                elif pred_var == 'lesion_burden_log':
                    ax.set_xlabel('Log10 lesion burden')
                    #ax.set_xlim(0,1)
                elif pred_var == 'lesion_count_log':
                    ax.set_xlabel('Log10 lesion count')
                    #ax.set_xlim(0,1)
                elif pred_var == 'lesion_burden_log_nz':
                    ax.set_xlabel('Log10 lesion burden (no zeroes)')
                    #ax.set_xlim(0,1)
                elif pred_var == 'lesion_count_log_nz':
                    ax.set_xlabel('Log10 lesion count (no zeroes)')
                    #ax.set_xlim(0,1)
                else:
                    ax.set_xlabel(pred_var)
                '''
                if col == 'total_vol':
                    ax.set_title(f'Total brain volume')
                elif col == 'wm_vol':
                    ax.set_title(f'White matter volume')
                if col == 'gm_vol':
                    ax.set_title(f'Gray matter volume')
                '''
                    
                if j == 0:
                    print(f'PRINT THAT LETTER: {let}')
                    # need to label the rows with letters
                    xlims = ax.get_xlim()
                    ylims = ax.get_ylim()
                    
                    perc_adj_x = (xlims[1]-xlims[0])*0.1
                    perc_adj_y = (ylims[1]-ylims[0])*0.0
                    
                    spot_x = xlims[0] - perc_adj_x
                    spot_y = ylims[1] + perc_adj_y
                    
                    #ax.text(spot_x, spot_y, let, size=24, fontweight='bold')
                    #ax.text(spot_x, spot_y, let, size=24, fontweight='bold', transform=ax.transAxes)
                    ax.text(-.1, 1.05, let, horizontalalignment='center',
                         verticalalignment='center', transform=ax.transAxes, size=24, fontweight='bold')
                    
                    print(spot_x, spot_y)
            
            plt.tight_layout()
            plt.savefig(figname, dpi=400)
            
            
    
    
            
    for i, (prog, norm_name, out_folder) in enumerate(zip(programs, norm_columns, sub_outs)):
    
        collated_csv = os.path.join(out_folder, 'collated.csv')
        clean_table = pd.read_csv(collated_csv, index_col='mr_id')
        
        clean_table = clean_table[clean_table['exclude'] != 1]
        
        clean_table['normal_control'] = [all([i, not j]) for i,j in zip(clean_table['control'], clean_table['sci'])]        
        clean_table['sci_control'] = [all([i, j]) for i,j in zip(clean_table['control'], clean_table['sci'])]   
        
        
        clean_table['normal_scd'] = [all([i, not j]) for i,j in zip(clean_table['scd'], clean_table['sci'])]        
        clean_table['sci_scd'] = [all([i, j]) for i,j in zip(clean_table['scd'], clean_table['sci'])]
        
        pt_type = ['control', 'scd']
        pt_type_proper = ['Control', 'SCD']
        exprs = [clean_table[pt] == 1 for pt in pt_type]
        subdfs = [clean_table[expr] for expr in exprs]
        # multiple linear regression looking at brain vol vs x,y,z
        

        '''        
        controlling = [
               ['age','gender', norm_name, 'scd'],
               ['age','gender', norm_name, 'cao2'],
               ['age','gender', norm_name, 'cao2'],
               ['age','gender', norm_name, 'cao2', 'scd'],
               ['age','gender', norm_name, 'wm_oxdel'],
               ['age','gender', norm_name, 'wm_oxdel'],
               ['age','gender', norm_name, 'wm_oxdel', 'scd'],
               ['age','gender', norm_name, 'wm_cbf'],
               ['age','gender', norm_name, 'wm_cbf'],
               ['age','gender', norm_name, 'wm_cbf', 'scd'],
               ['age','gender', norm_name, 'oef'],
               ['age','gender', norm_name, 'oef'],
               ['age','gender', norm_name, 'oef', 'scd'],
               ['age','gender', norm_name, 'shunt_score'],
               ['age','gender', norm_name, 'shunt_score'],
               ['age','gender', norm_name, 'shunt_score', 'scd'],
               ['age','gender', norm_name, 'lesion_count'],
               ['age','gender', norm_name, 'gm_cbf']
        ]
        
        factor_sets = [
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol', 'wm_vol', 'total_vol'],
            ['gm_vol_unadj', 'wm_vol_unadj', 'total_vol_unadj', 'lesion_count']
        ]
        
        keep_param = [None, 0, 1, None, 0, 1, None, 0, 1, None, 0, 1, None, 0, 1, None, 1, 1] # 1 to keep only scd, 0 to keep controls only, None to keep both
        
        #interactions = [None, None, 'scd:cao2', None, None, None]
        interactions = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        '''
        '''
            blank_dict = {'mr_id':None,
                  'wm_vol':None,
                  'gm_vol':None,
                  'total_vol':None,
                  'wm_vol_unadj':None,
                  'gm_vol_unadj':None,
                  'total_vol_unadj':None,
                  'superiorfrontal_unadj':None,
                  'rostralmiddlefrontal_unadj':None,
                  'caudalmiddlefrontal_unadj':None,
                  'frontalpole_unadj':None,
                  'hippocampus_unadj':None,
                  'subfrontal_unadj':None,
                  'icv':None,
                  'csf_vol':None,
                  'gm_normal':None,
                  'wm_normal':None,
                  'total_normal':None,
                  'superiorfrontal_normal':None,
                  'rostralmiddlefrontal_normal':None,
                  'caudalmiddlefrontal_normal':None,
                  'frontalpole_normal':None,
                  'hippocampus_normal':None,
                  'subfrontal_normal':None,
                  'vscaling':None,
                  'hct':None,
                  'gm_cbf':None,
                  'wm_cbf':None,
                  'cao2':None,
                  'oef':None,
                  'shunt_score':None,
                  'wm_oxdel':None,
                  'gm_oxdel':None,
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
                  'excl_missing_wm_cbf':None,
                  'excl_lobcalc':None}
        '''
        
        logit_factors = ['sci']
        
        
        ct_1 = ['age','gender', norm_name, 'scd']
        
        ct_2 = ['age','gender', norm_name, 'hemoglobin']
        ct_3 = ['age','gender', norm_name, 'gm_cbf']
        ct_4 = ['age','gender', norm_name, 'wm_cbf']
        ct_5 = ['age','gender', norm_name, 'cao2']
        
        ct_6 = ['age','gender', norm_name, 'lesion_burden_log_nz']
        
        
        
        controlling = [
            
              ct_1,
              ct_2,
              ct_3,
              ct_4,
              ct_5,
              ct_6,
              
              ct_1,
              ct_2,
              ct_3,
              ct_4,
              ct_5,
              ct_6
               
        ]
        
        
        #fs_1 = ['gm_vol', 'wm_vol', 'subfrontal_vol', 'superiorfrontal_vol', 'rostralmiddlefrontal_vol',
        #        'caudalmiddlefrontal_vol', 'frontalpole_vol', 'hippocampus_vol']
        
        #fs_1 = ['gm_vol_unadj', 'wm_vol_unadj', 'subfrontal_unadj', 'hippocampus_unadj']
        
        #fs_1 = ['gm_vol', 'wm_vol']
        fs_2 = ['gm_vol_unadj', 'wm_vol_unadj']
        
        fs_1 = []
        ad1 = [f'{i}_vol' for i in lob_keys]
        ad2 = [f'{i}_vol' for i in subcort_keys]
        fs_1.extend(ad1)
        fs_1.extend(ad2)
        
        no_indy = [
                    'Left_Lateral_Ventricle_vol',
                     'Left_non_WM_hypointensities_vol',
                     'Right_Lateral_Ventricle_vol',
                     'Right_non_WM_hypointensities_vol',
                     'WM_hypointensities_vol',
                     'fiveth_Ventricle_vol',
                     'fourth_Ventricle_vol',
                     'non_WM_hypointensities_vol',
                     'threerd_Ventricle_vol',
                     'Left_WM_hypointensities_vol',
                     'Right_WM_hypointensities_vol',
                     'Left_Inf_Lat_Vent_vol',
                     'Right_Inf_Lat_Vent_vol'
                 ]
        
        
        fs_1 = [i for i in fs_1 if i not in no_indy]
        
        
        fs_2 = ['gm_vol_unadj', 'wm_vol_unadj']
        
        
        fs_3 = []
        fs_3.extend(fs_1)
        fs_3.extend(fs_2)
        
        #fs_2 = ['subfrontal_unadj', 'superiorfrontal_unadj', 'rostralmiddlefrontal_unadj',
        #        'caudalmiddlefrontal_unadj', 'frontalpole_unadj', 'hippocampus_unadj']
        
        fs_4 = ['lesion_count_log_nz', 'lesion_burden_log_nz', 'sci']
        
        #fs_1 = ['gm_vol', 'wm_vol']
        #fs_2 = ['gm_vol', 'wm_vol', 'subfrontal_vol', 'hippocampus_vol', 'sci', 'lesion_count', 'lesion_burden']
        #fs_2 = ['gm_vol_unadj', 'wm_vol_unadj', 'subfrontal_unadj', 'hippocampus_unadj']
        #fs_3 = ['lesion_count', 'lesion_burden']
        
        factor_sets = [
            fs_2,
            fs_2,
            fs_2,
            fs_2,
            fs_2,
            fs_2,
            
            fs_1,
            fs_1,
            fs_1,
            fs_1,
            fs_1,
            fs_1
        ]
        
        keep_param = [None, None, None, None, None, 1, None, None, None, None, None, 1] # 1 to keep only scd, 0 to keep controls only, None to keep both
        
        #interactions = [None, None, 'scd:cao2', None, None, None]
        interactions = [None, None, None, None, None, None, None, None, None, None, None, None]
        
        
        descrip = np.arange(len(controlling))
        
        if len(descrip) != len(interactions) != len(keep_param) != len(factor_sets) != len(controlling):
            raise Exception('lengths are not the same for statistical regression params')
        
        
        p_df = pd.DataFrame()
        p_df_name = os.path.join(out_folder, f'pvals.xlsx')
        
        corr_check = ['age', 'gender', norm_name, 'scd', 'cao2', 'lesion_count', 'gm_vol', 'wm_vol', 'total_vol']
        corr_base = clean_table[corr_check].dropna()
        corr_mat_file = os.path.join(out_folder, f'correlation_matrix_{prog}.csv')
        #corr_file = open(corr_mat_file, 'w')
        corr_mat = corr_base.corr()
        corr_mat.to_csv(corr_mat_file)
        #corr_file.write(str(corr_mat))
        #corr_file.close()
        
        
        preds_of_interest = ['scd', 'hemoglobin', 'lesion_burden_log_nz', 'gm_cbf', 'cao2', 'gm_oxdel', 'wm_cbf']
            
        for controller, keeper, interact, des, factor_set in zip(controlling, keep_param, interactions, descrip, factor_sets):
            
            c_df = pd.DataFrame()
            
            print(f'Model: {des}')
            
            summary_file = os.path.join(out_folder, f'signficance_summary_{"_".join(controller)}.txt')
            summary = open(summary_file, 'w')
            
            if 'hemoglobin_s_frac' in controller:
                # if we're looking at hbs frac, need to remove transfused pts
                use_table = clean_table[clean_table['transf']==0]
            else:
                use_table = clean_table.copy()
                
            
            print('QUAD')
            for f in factor_set:
                #print(f'\n\n\nFACTOR: {f}\n')
                
                if f in controller:
                    continue # doesn't make sense to run a regression where something is both a predictor and the criterion
                
                
                
                print(controller)
                
                cont_copy = controller.copy()
                
               
                pars = controller.copy()
                if f not in pars:
                    pars.append(f)
                    
                if 'scd' not in pars:
                    pars.append('scd')
                
                interact_drop = []
                if interact is not None:
                    for term in interact.split(':'):
                        
                        #if term == 'sci':
                        #    sys.exit()
                        
                        if term not in pars:
                            pars.append(term)
                        if term not in controller:
                            cont_copy.append(term)
                            interact_drop.append(term)
                    
                print(f'\tModel - dep var {f}, controlling for {pars} (descrip [{des}])')
                
                tabby = use_table[pars].dropna()
                    
                keep_text = 'all'
                if keeper != None:
                    tabby = tabby[tabby['scd']==keeper] # only keep SCD participants
                    if keeper == 1:
                        keep_text = 'scd_only'
                    elif keeper == 0:
                        keep_text = 'controls_only'
                     
                print(f'\t{keep_text}')
                
                
                X = tabby[cont_copy]
                Y = tabby[f]
                
                X2 = sm.add_constant(X)
                
                smashed_df = X2.copy()
                smashed_df[f] = Y
                
                cols_to_use = list(X2.columns)
                cols_to_use = [i for i in cols_to_use if i not in interact_drop]
                regression_statement = f'{f} ~ {" + ".join(cols_to_use)}'
                if interact:
                    regression_statement = regression_statement + ' + ' + interact
                    
                regression_statement = regression_statement + ' - 1' # remove intercept since we already added a constant
                
                
                #est = sm.OLS(Y, X2, hasconst=True)
                if f in logit_factors:
                    is_logit = True
                    est = smf.logit(regression_statement, data=smashed_df)
                    est2 = est.fit()
                    
                    
                else:
                    is_logit = False
                    est = smf.ols(regression_statement, data=smashed_df)
                    est2 = est.fit()
                    est2 = est2.get_robustcov_results() # make it robust
                
                
                
                summary.write(str(est2.summary()))
                summary.write('\n\n\n\n--------\n--------\n\n\n\n')
                
                results_summary = est2.summary()
                
                results_as_html = results_summary.tables[1].as_html()
                as_df = pd.read_html(results_as_html, header=0)[0]
                as_df['criterion'] = f
                as_df['covariates'] = '+'.join(controller)
                as_df['keep'] = keep_text
                as_df['interactions'] = interact
                as_df['description'] = des
                as_df['is_logit'] = is_logit
                
                for i, row in as_df.iterrows():
                    as_df.loc[i, 'P>|t|'] = est2.pvalues[i]
                
                as_df = as_df.rename(columns={'Unnamed: 0':'predictor'})
                
                #droppers = ['const', 'age', 'gender']
                #for d in droppers:
                #    as_df = as_df[as_df['predictor'] != d]
                
                
                as_df = as_df[as_df['predictor'].isin(preds_of_interest)]
                
                c_df = c_df.append(as_df, ignore_index=True)
                
                
            c_df['p_fdr_corrected'] = mtt.multipletests(c_df['P>|t|'], alpha=0.05, method='fdr_bh')[1]
            p_df = p_df.append(c_df, ignore_index=True)
            
            
            summary.close()
        p_df.to_excel(p_df_name)
            
            
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
    
    '''
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
    '''
    
    ##### scatter+bland-altman plots
    
    def bland_altman_plot(data1, data2, ax, left_loc=None, do_right=False, *args, **kwargs):
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
            if do_right:
                left_loc = max(mean)
        
        ax.annotate(f'Mean diff: {md:.2f}', (left_loc,md+6), path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.75)])
        ax.annotate(f'-SD 1.96: {md-1.96*sd:.2f}', (left_loc,md-1.96*sd+6), path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.75)])
        ax.annotate(f'+SD 1.96: {md+1.96*sd:.2f}', (left_loc,md+1.96*sd+6), path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.75)])
        
        #ax.text(0.5, 0.5, f'Mean diff: {round(md,2)}',
        #      size=20,
        #      color='white',
        #      path_effects=[pe.withStroke(linewidth=4, foreground="red")])
        
        ax.set_xlabel("Mean (mL)")
        ax.set_ylabel("Difference (mL)")
        
    
    
    out_of_spec = []
    
    vol_measures = ['total_vol', 'gm_vol', 'wm_vol']
    formal_measures = ['Agreement of total volumes', 'Agreement of gray matter volumes', 'Agreement of white matter volumes']
    lim_list =[[700,1400],[350,850],[250,600]]
    
    #lim_list_bland_ex = [[700,1300],[250,825],[250,825]]
    #lim_list_bland_why = [[-200,200],[-165,200],[-165,200]]
    lim_list_bland_ex = [[None,None],[None,None],[None,None]]
    lim_list_bland_why = [[None,None],[None,None],[None,None]]
    program_pairs = list(itertools.combinations(data_dicts.keys(), 2))
    letters = ['A', 'B', 'C']
    
    for lims, measure, f_measure, bl_x, bl_y in zip(lim_list, vol_measures, formal_measures, lim_list_bland_ex, lim_list_bland_why):
        fig, axs = plt.subplots(len(program_pairs), 2, figsize=(12,24))
        for (p1, p2), axrow, let in zip(program_pairs, axs, letters):
            
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
            
            inds1 = list(d1.index)
            inds2 = list(d2.index)
            
            inds = inds1.copy()
            inds.extend(inds2)
            
            inds = set(inds)
            
            indl = []
            for ind in inds:
                try:
                    m1 = d1[measure].loc[ind]
                    m2 = d2[measure].loc[ind]
                    
                    exes1.append(m1)
                    exes2.append(m2)
                    
                    indl.append(ind)
                except KeyError:
                    pass
                
            exes1 = np.array(exes1)
            exes2 = np.array(exes2)
            
            raters = []
            raters.extend([p1]*len(exes1))
            raters.extend([p2]*len(exes2))
            
            vols = []
            vols.extend(exes1)
            vols.extend(exes2)
            
            indlist = []
            indlist.extend(indl)
            indlist.extend(indl)
            
            icc_df = pd.DataFrame()
            icc_df['raters'] = raters
            icc_df['vols'] = vols
            icc_df['pts'] = indlist
            
            icc_df_vals = pg.intraclass_corr(data=icc_df, targets='pts', raters='raters',
                         ratings='vols').round(3)
            
            # we want the ICC3 (mixed effect single rater) because we are concerned only with the raters presented here, and are interested in the reliability of the rating of a single rater
            
            icc = icc_df_vals[icc_df_vals['Type'] == 'ICC3'].iloc[0]['ICC']
            icc_ci = icc_df_vals[icc_df_vals['Type'] == 'ICC3'].iloc[0]['CI95%']
            
            
            
            print(f'Lens: {len(exes1)},{len(exes2)}')
            
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
            axrow[0].set_xlabel(f'{progname1} (mL)')
            axrow[0].set_ylabel(f'{progname2} (mL)')
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
            
            icc_ci_low = f'{icc_ci[0]:.2f}'
            icc_ci_high = f'{icc_ci[1]:.2f}'
            #axrow[0].set_title(f'{progname2} vs. {progname1} ($r^2$ = {round(r2,2)})')
            axrow[0].set_title(f'{progname2} vs. {progname1}\n(ICC = {icc:.2f}, [{icc_ci_low},{icc_ci_high}])')#, {icc_ci})')
            
            if None not in bl_x:
                le_lo = bl_x[0]+10
            else:
                le_lo = None
            
            bland_altman_plot(exes1, exes2, ax=axrow[1], c='cornflowerblue', left_loc=le_lo, edgecolors='black', alpha=0.75)
            axrow[1].set_xlim(bl_x[0],bl_x[1])
            axrow[1].set_ylim(bl_y[0],bl_y[1])
            
            xlims = axrow[0].get_xlim()
            ylims = axrow[0].get_ylim()
            
            perc_adj_x = (xlims[1]-xlims[0])*0.1
            perc_adj_y = (ylims[1]-ylims[0])*0.05
            
            spot_x = xlims[0] - perc_adj_x
            spot_y = ylims[1] + perc_adj_y
            
            axrow[0].text(spot_x, spot_y, let, size=18, fontweight='bold')
            
            
        exes1 = []
        exes2 = []
        exes3 = []
        
        d1 = data_dicts['SPM']['data']
        d2 = data_dicts['FS']['data']
        d3 = data_dicts['SIENAX']['data']
        
        inds1 = list(d1.index)
        inds2 = list(d2.index)
        inds3 = list(d3.index)
        
        inds = inds1.copy()
        inds.extend(inds2)
        inds.extend(inds3)
        
        inds = set(inds)
        
        indl = []
        for ind in inds:
            try:
                m1 = d1[measure].loc[ind]
                m2 = d2[measure].loc[ind]
                m3 = d3[measure].loc[ind]
                
                exes1.append(m1)
                exes2.append(m2)
                exes3.append(m3)
                
                indl.append(ind)
            except KeyError:
                pass
            
        exes1 = np.array(exes1)
        exes2 = np.array(exes2)
        exes3 = np.array(exes3)
        
        raters = []
        raters.extend(['SPM']*len(exes1))
        raters.extend(['FS']*len(exes2))
        raters.extend(['SIENAX']*len(exes3))
        
        vols = []
        vols.extend(exes1)
        vols.extend(exes2)
        vols.extend(exes3)
        
        indlist = []
        indlist.extend(indl)
        indlist.extend(indl)
        indlist.extend(indl)
        
        icc_df_full = pd.DataFrame()
        icc_df_full['raters'] = raters
        icc_df_full['vols'] = vols
        icc_df_full['pts'] = indlist
        
        icc_df_vals = pg.intraclass_corr(data=icc_df_full, targets='pts', raters='raters',
                     ratings='vols').round(3)
        
        # we want the ICC3 (mixed effect single rater) because we are concerned only with the raters presented here, and are interested in the reliability of the rating of a single rater
        
        icc = icc_df_vals[icc_df_vals['Type'] == 'ICC3'].iloc[0]['ICC']
        icc_ci = icc_df_vals[icc_df_vals['Type'] == 'ICC3'].iloc[0]['CI95%']
        
        icc_ci_low = f'{icc_ci[0]:.2f}'
        icc_ci_high = f'{icc_ci[1]:.2f}'
        fig.suptitle(f'{f_measure}\n(full ICC = {icc:.2f}, [{icc_ci_low},{icc_ci_high}])', size=16)#, {icc_ci})')
        fig.tight_layout(rect=[0.01, 0.03, 1, 0.95])
        figname =  os.path.join(interrater_folder, f'agreement_{measure}.png')
        plt.savefig(figname, dpi=400)
        
    unique_out = set(out_of_spec)
    
    # hex projection
    
    figname =  os.path.join(interrater_folder, f'hexprojection.png')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
    
    vol_types = ['wm_vol', 'gm_vol']
    vol_names = ['White matter', 'Gray matter']
    
    for vt, vn, ax in zip(vol_types, vol_names, axs):      
        
        
        ax.axis('off')

        x = data_dicts['FS']['data'][vt]
        y = data_dicts['SIENAX']['data'][vt]
        z = data_dicts['SPM']['data'][vt]
        
        x_dir = np.array([np.cos(0), np.sin(0)])
        y_dir = np.array([np.cos(2*np.pi*1/3), np.sin(2*np.pi*1/3)])
        z_dir = np.array([np.cos(2*np.pi*2/3), np.sin(2*np.pi*2/3)])
        
        x_dir = x_dir / np.linalg.norm(x_dir)
        y_dir = y_dir / np.linalg.norm(y_dir)
        z_dir = z_dir / np.linalg.norm(z_dir)
        
        x_prop = [i*x_dir for i in x]
        y_prop = [i*y_dir for i in y]
        z_prop = [i*z_dir for i in z]
        
        verts = [np.array([tx, ty, tz]) for tx,ty,tz in zip(x_prop, y_prop, z_prop)]
        verts_norm = [v/np.mean([tx,ty,tz]) for v,tx,ty,tz in zip(verts,x,y,z)]
        means = [np.mean([tx,ty,tz]) for tx,ty,tz in zip(x,y,z)]
        
        centroids = [np.array([vert[:,0].mean(), vert[:,1].mean()]) for vert in verts_norm]
        
        cx = [c[0] for c in centroids]
        cy = [c[1] for c in centroids]
        
        #ax.scatter(cx, cy, color='gray', alpha=0.5, edgecolor='black')
        ax.scatter(cx, cy, c=means, cmap='magma', alpha=0.3, edgecolor='black')
        
        scaler = 0.18
        ax.plot([0,x_dir[0]*scaler],[0,x_dir[1]*scaler], color='red', alpha=0.6)#, label='FreeSurfer')
        ax.plot([0,y_dir[0]*scaler],[0,y_dir[1]*scaler], color='green', alpha=0.6)#, label='SIENAX')
        ax.plot([0,z_dir[0]*scaler],[0,z_dir[1]*scaler], color='blue', alpha=0.6)#, label='SPM')
        
        ax.plot([0,-x_dir[0]*scaler],[0,-x_dir[1]*scaler], color='red', alpha=0.6, ls='--')
        ax.plot([0,-y_dir[0]*scaler],[0,-y_dir[1]*scaler], color='green', alpha=0.6, ls='--')
        ax.plot([0,-z_dir[0]*scaler],[0,-z_dir[1]*scaler], color='blue', alpha=0.6, ls='--')
        
        ax.set_title(vn)
        ax.set_aspect('equal', 'box')
        plt.legend()  
    
    
    #ax.set_xlim(-0.25,0.25)
    #ax.set_ylim(-0.25,0.25)
            
        directions = [np.array([np.cos(2*np.pi*i/6), np.sin(2*np.pi*i/6)])*scaler*0.85 for i in np.arange(0,6,1)]
        txts = ['FreeSurfer\noverestimates', 'SPM\nunderestimates', 'SIENAX\noverestimates', 'FreeSurfer\nunderestimates', 'SPM\noverestimates', 'SIENAX\nunderestimates']
        
        upnudge = 0.005
        for di, tx in zip(directions, txts):
            # path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.75)
            #ax.text(di[0], di[1]+upnudge, tx)
            ax.annotate(tx, [di[0], di[1]+upnudge], path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.75)])
    
    plt.tight_layout()
    plt.show()
    plt.savefig(figname, dpi=400)
    
    
    
        
        
        
        
             

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
        
        
if other_plots:
    plt.style.use('dark_background')
    plt.rc('axes', labelsize=16)
    plt.rc('axes', titlesize=16)
    plt.rc('figure', titlesize=16)
    
    segfig_name = os.path.join(out_folder_orig, 'segmentations.png')
    
    mr_ids = ['SCD_C036', 'SCD_P002_01', 'SCD_P046']
    mr_labels = ['HC', 'SCD, no SCI', 'SCI with SCI']
    
    races = ['Black', 'Black', 'Black']
    ages = [25, 21, 21]
    genders = ['male', 'male', 'male']
    
    #im_labels = ['T$_1$', 'Segmentation', 'FLAIR']
    #im_labels = ['T$_1$', 'Overlay', 'Segmentation']
    im_labels = ['T$_1$', 'Overlay']
    
    
    figure, axs = plt.subplots(len(mr_ids), len(im_labels), figsize=(12, 12))
    
    #gm_codes = [3, 42, 11, 50]
    no_codes = [4, 43, 0]
    wm_codes = [41, 2, 46,7, 251, 252, 253, 254, 255]
    
    #not_wm_codes = gm_codes.copy()
    #not_wm_codes.extend(no_codes)
    not_gm_codes = wm_codes.copy()
    not_gm_codes.extend(no_codes)
    
    for i, (mr_id, mr_label, race, age, gender, row) in enumerate(zip(mr_ids, mr_labels, races, ages, genders, axs)):
        mr_folder = os.path.join(fs_folder, mr_id)
        
        seg = os.path.join(mr_folder, 'mri', 'aseg.mgz')
        brain = os.path.join(mr_folder, 'mri', 'orig.mgz')
        
        seg_load = nib.load(seg)
        brain_load = nib.load(brain)
        
        
        for j, (lab, ax) in enumerate(zip(im_labels, row)):
            
            braindat = brain_load.get_fdata()
            brainshape = braindat.shape
            braindat_slice = np.rot90(braindat[12:244, int(brainshape[1]/2)-10, 12:244])
            
            segdat = seg_load.get_fdata()
            segshape = segdat.shape
            segdat_slice = np.rot90(segdat[12:244, int(segshape[1]/2)-10, 12:244])
            
            if j < 2:
                cmap = 'gist_gray'
                ax.imshow(braindat_slice, cmap=cmap)
            if j > 0:
                
                gm_mask = ~np.isin(segdat_slice, not_gm_codes)
                wm_mask = np.isin(segdat_slice, wm_codes)
                
                gm_cmap = plt.get_cmap('Blues')
                wm_cmap = plt.get_cmap('gist_gray')
                
                gmma = np.ma.masked_array(gm_mask, ~gm_mask)
                wmma = np.ma.masked_array(wm_mask, ~wm_mask)
                
                
                gm_cmap.set_bad(alpha=0)
                wm_cmap.set_bad(alpha=0)
                
                ax.imshow(gmma, cmap=gm_cmap, alpha=0.6, vmin=0.75, vmax=1.25, interpolation='nearest')
                ax.imshow(wmma, cmap=wm_cmap, alpha=0.6, vmin=0.75, vmax=1, interpolation='nearest')
                
            
            
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            
            plt.subplots_adjust(
                    wspace=00, 
                    hspace=00
                    )
            
            if i == 0:
                ax.set_title(lab)
                
            if j == 0:
                ax.set_ylabel(f'{mr_label}\n{race} {gender}, {age} years old')
    
    plt.tight_layout()
    plt.savefig(segfig_name, dpi=500)
            

        