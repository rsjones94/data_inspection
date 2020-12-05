
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
import scipy

from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data

from parse_fs_stats import parse_freesurfer_stats

exclude_pts = ['SCD_K065', 'SCD_TRANSP_P001_01']

in_csv = '/Users/manusdonahue/Documents/Sky/stroke_status.csv'
out_folder = '/Users/manusdonahue/Documents/Sky/freesurfer_volume_visualization/'
parsed_folder = '/Users/manusdonahue/Documents/Sky/freesurfer_volume_visualization/parsed'

brain_vol_csv = '/Users/manusdonahue/Documents/Sky/normal_brain_vols.csv' # from Borzage, Equations to describe brain size across the continuum of human lifespan (2012)
# values originally reported as mass in g, converted to cc assuming rho = 1.04 g/cc

fs_folder = '/Volumes/DonahueDataDrive/freesurfer_subjects/'

parse = False
collate = False
visualize = True

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

in_table = pd.read_csv(in_csv)
in_table = in_table.dropna(subset=['mr1_mr_id_real'])
mr_ids = in_table['mr1_mr_id_real']

in_table_indexed = pd.read_csv(in_csv, index_col='mr1_mr_id_real')

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
                  'supratent_normal':None,
                  'age':None,
                  'stroke_silent':None,
                  'white_matter_injury':None,
                  'stroke_overt':None,
                  'sci':None,
                  'transf':None,
                  'scd':None,
                  'anemia':None,
                  'control':None,
                  'gender':None,
                  'exclude':0}
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
        working['age'] = in_table_indexed.loc[pt_name]['age']
        working['etiv'] = parsed_csv.loc['eTIV']['value'] / 1e3
        working['gm_normal'] = working['gm_vol'] / working['etiv']
        working['wm_normal'] = working['wm_vol'] / working['etiv']
        working['total_normal'] = working['total_vol'] / working['etiv']
        working['supratent'] = parsed_csv.loc['SupraTentorialVolNotVent']['value'] / 1e3
        working['supratent_normal'] = working['supratent'] / working['etiv']
        working['csf_vol'] = working['etiv'] - working['total_vol']
        
        if in_table_indexed.loc[pt_name]['mri1_wml_drp'] == 1:
            working['white_matter_injury'] = 1
        else:
            working['white_matter_injury'] = 0
        
        stroke_overt = in_table_indexed.loc[pt_name]['mh_rf_prior_stroke_overt']
        stroke_silent = in_table_indexed.loc[pt_name]['mh_rf_prior_stroke_silent']
        
        if stroke_overt == 1 or stroke_silent == 1:
            working['exclude'] = 1
        
        sci = in_table_indexed.loc[pt_name]['outcome_mri1_sci']
        transf = in_table_indexed.loc[pt_name]['enroll_sca_transfusion']
        
        #if transf == 1:
            #working['exclude'] = 1
        
        for val, name in zip([stroke_overt, stroke_silent, sci, transf],
                             ['stroke_overt', 'stroke_silent', 'sci', 'transf']):
            if val == 1:
                working[name] = 1
            else:
                working[name] = 0
                
        status = in_table_indexed.loc[pt_name]['case_control']
        
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
            
        working['gender']  = in_table_indexed.loc[pt_name]['gender']
        
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
    
        
                
        out_df = out_df.append(working, ignore_index=True)
    
    out_df = out_df[blank_dict.keys()]
    out_df.to_csv(out_csv, index=False)
        
if visualize:
    print('Visualizing')
    brain_vol_df = pd.read_csv(brain_vol_csv)
    
    collated_csv = os.path.join(out_folder, 'collated.csv')
    clean_table = pd.read_csv(collated_csv, index_col='mr_id')
    clean_table = clean_table[clean_table['exclude'] != 1]
    
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.06)

    y_pred = clf.fit_predict(clean_table)
    #y_pred_unsort = y_pred.copy()
    x_scores = clf.negative_outlier_factor_
    #x_scores_unsort = x_scores.copy()
    clean_table['outlier'] = y_pred
        
    clean_table['normal_control'] = [all([i, not j, not k]) for i,j,k in zip(clean_table['control'], clean_table['sci'], clean_table['white_matter_injury'])]        
    clean_table['sci_control'] = [all([i, j]) for i,j in zip(clean_table['control'], clean_table['sci'])]      
    clean_table['wmi_control'] = [all([i, j, not k]) for i,j,k in zip(clean_table['control'], clean_table['white_matter_injury'], clean_table['sci'])]      
    
    
    clean_table['normal_scd'] = [all([i, not j, not k]) for i,j,k in zip(clean_table['scd'], clean_table['sci'], clean_table['white_matter_injury'])]        
    clean_table['sci_scd'] = [all([i, j]) for i,j in zip(clean_table['scd'], clean_table['sci'])]        
    clean_table['wmi_scd'] = [all([i, j, not k]) for i,j,k in zip(clean_table['scd'], clean_table['white_matter_injury'], clean_table['sci'])]

    '''
    names = ['normal_control', 'sci_control', 'normal_scd', 'sci_scd']
    colors = ['green', 'blue', 'red', 'orange']
    
    fig, ax = plt.subplots(figsize=(10,10))
    x_multi = [clean_table[clean_table[col]]['age'] for col in names]
    bin_list = np.arange(0,60,5)
    ax.hist(x_multi, bins=bin_list, histtype='bar', color=colors, label=names)
    #ax.hist(x_multi, 20, histtype='step', stacked=True, fill=True, color=colors, label=names)
    ax.set_title('Age distribution in SCD study cohorts')
    plt.xticks(bin_list)
    plt.xlabel('Age (years)')
    plt.ylabel('Occurrences')
    hist_out = os.path.join(out_folder, 'age_hist.png')
    ax.legend(prop={'size': 10})
    plt.tight_layout()
    plt.savefig(hist_out)
    
    
    
    y_names = ['wm_vol', 'gm_vol', 'total_vol', 'total_vol_custom', 'mask_vol_custom', 'gm_normal', 'wm_normal', 'total_normal', 'etiv', 'supratent', 'supratent_normal']

    xlims = [0, 50]
    ylims = [0, 1800]
    
    genders = [[1,'male'], [0,'female'], [None,'all']]
    for g, g_name in genders:
        if g is not None:
            gender_table = clean_table[clean_table['gender'] == g]
        else:
            gender_table = clean_table
        for y_name in y_names:
            fig, ax = plt.subplots(figsize=(12,6))
            plot_name = os.path.join(out_folder, f'{y_name}_{g_name}.png')
            artists = []
            artist_descs = []
            for n, c in zip(names, colors):
                tab = gender_table[gender_table[n] == True]
                exes = tab['age']
                whys = tab[y_name]
                
                plt.scatter(exes, whys, color=c, alpha=0.5)
                
                slope, inter, res, p_val, stderr = stats.linregress(exes, whys)
                eq = f'y = {round(slope,1)}*x + {int(inter)}'
                
                mean_age = round(exes.mean(),1)
                stddev_age = round(exes.std(),1)
                artist_descs.append(f'{n}: {eq}\nmean age: {mean_age} (sd +/- {stddev_age})')
                artists.append(plt.Circle((0,0), color=c))
                
                #lin_ex = [clean_table['age'].min(), clean_table['age'].max()]
                lin_ex = xlims
                lin_y = [slope*i + inter for i in lin_ex]
                plt.plot(lin_ex, lin_y, color=c, alpha = 0.7)
                
            plt.xlabel('Patient age (years)')
            if 'normal' not in y_name:
                plt.ylabel('Volume (cc)')
            else:
                plt.ylabel('Normalized volume (tissue vol / eTIV)')
                   
            if y_name in ['total_vol', 'total_vol_custom', 'mask_vol_custom'] and g_name != 'all':
                plt.plot(brain_vol_df['age'], brain_vol_df[f'fit_{g_name}'], color='purple', alpha=0.25)
                plt.plot(brain_vol_df['age'], brain_vol_df[f'min_{g_name}'], color='purple', ls='dashed', alpha=0.25)
                plt.plot(brain_vol_df['age'], brain_vol_df[f'max_{g_name}'], color='purple', ls='dashed', alpha=0.25)
            
            
            sign = 1
            all_out = []
            for pt_name, score, ex, why, gen in zip(clean_table.index, clean_table['outlier'], clean_table['age'], clean_table[y_name], clean_table['gender']):
                if score == -1 and gen == g:
                    plt.scatter(ex,why,color='black', marker='x')
                    plt.annotate(pt_name, (ex-0.25,why-20), ha='right')
                    
                    all_out.append(pt_name)
                    
                    #the_img = '/Users/manusdonahue/Documents/Sky/small.png'
                    #with get_sample_data(the_img) as file:
                        #arr_img = plt.imread(file, format='png')
                    
                    if y_name in ['total_vol', 'total_vol_custom', 'mask_vol_custom']:
                        sign *= -1
                        the_img = os.path.join(fs_folder, pt_name, 'mri', 'brain.mgz')
                        #the_img = os.path.join(fs_folder, pt_name, 'mri', 'aseg.mgz')
                        raw = nib.load(the_img)
                        img = raw.get_fdata()
                        img = filter_zeroed_axial_slices(img, 0.95)
                        sli = int(img.shape[1]*0.4)
                        arr_img = img[:,sli,:]
                        
                        arr_img = np.rot90(arr_img, k=1)
                        
                        # filter out null border
                        # argwhere will give you the coordinates of every non-zero point
                        true_points = np.argwhere(arr_img)
                        # take the smallest points and use them as the top left of your crop
                        top_left = true_points.min(axis=0)
                        # take the largest points and use them as the bottom right of your crop
                        bottom_right = true_points.max(axis=0)
                        out = arr_img[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
                                  top_left[1]:bottom_right[1]+1]  # inclusive
                        
                        arr_img = out
                        
                        imagebox = OffsetImage(arr_img, zoom=0.2, cmap='gist_yarg')
                        imagebox.image.axes = ax
                        
                        ab = AnnotationBbox(imagebox, (ex,why),
                                            xybox=(125., sign*30.),
                                            xycoords='data',
                                            boxcoords="offset points",
                                            pad=0.5,
                                            arrowprops=dict(
                                                arrowstyle="->",
                                                connectionstyle="angle,angleA=0,angleB=90,rad=3")
                                            )
                        
                        ax.add_artist(ab)
            
            
            plt.title(f'{y_name} ({g_name})\noutliers: {all_out}')
            if 'normal' not in y_name:
                plt.ylim(ylims)
            else:
                plt.ylim((0,1))
                
            plt.xlim(xlims)
            
            plt.legend(artists, artist_descs)
            
            plt.tight_layout()
            plt.savefig(plot_name)
    ''' 
    #interest = ['total_vol', 'wm_vol', 'gm_vol', 'csv']
    """
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
          'supratent_normal':None,
          'age':None,
          'stroke_silent':None,
          'stroke_overt':None,
          'sci':None,
          'transf':None,
          'scd':None,
          'anemia':None,
          'control':None,
          'gender':None}
        """
          
    interest = ['wm_vol', 'gm_vol', 'supratent', 'wm_normal', 'gm_normal', 'supratent_normal']
    pt_types = ['normal_control', 'wmi_control', 'sci_control', 'normal_scd', 'wmi_scd', 'sci_scd']
    cols = ['red', 'blue', 'red', 'blue', 'red', 'blue']
    fig, axs = plt.subplots(len(pt_types), len(interest), figsize=(4*len(interest),4*len(pt_types)))
    for pt_type, color, axrow in zip(pt_types, cols, axs):
        
        print(f'Row: {pt_type}')
        
        expr = clean_table[pt_type] == 1
        subdf = clean_table[expr]
        
        subdf_young = subdf[subdf['age'] < 16]
        subdf_old = subdf[subdf['age'] >= 16]
        subds = [subdf_young, subdf_old]
        for col, ax in zip(interest, axrow):
            
            subcolors = ['red', 'blue']
            for subcolor, subd in zip(subcolors, subds):
                
                exes = subd['age']
                whys = subd[col]
                
                
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
                    ax.plot(x, y_model, "-", color="0.1", linewidth=1.5, alpha=0.5, label="Fit")  
                    
                    x2 = np.linspace(np.min(x), np.max(x), 100)
                    y2 = equation(p, x2)
                    
                    # Confidence Interval (select one)
                    plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)
                    #plot_ci_bootstrap(x, y, resid, ax=ax)
                    
                    # Prediction Interval
                    pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))  
                    ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
                    ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
                    ax.plot(x2, y2 + pi, "--", color="0.5")
                    
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
                
                ax.scatter(exes, whys, color=subcolor, alpha = 0.4, s=4)
                
            ax.set_title(f'{pt_type}: {col}')
            ax.set_xlabel('Age (years)')
            
            if 'norm' in col:
                ax.set_ylabel('Normalized volume')
            else:
                ax.set_ylabel('Volume (cc)')
            
            ax.set_xlim(0,50)
            
            if 'norm' in col:
                ax.set_ylim(-0.1,1.1)
            else:
                ax.set_ylim(0,1400)
       
    plt.tight_layout()
    nice_name = os.path.join(out_folder, 'tissue_vol_age_dependency.png')
    plt.savefig(nice_name)
    
    
    
    # bar plots
    interest = ['wm_vol', 'gm_vol', 'supratent', 'wm_normal', 'gm_normal', 'supratent_normal']
    #pt_types = [['normal_control', 'wmi_control'], ['normal_control'], ['wmi_control'], ['sci_control'], ['normal_scd', 'wmi_scd'], ['normal_scd'], ['wmi_scd'], ['sci_scd']]
    pt_types = ['normal_control', 'wmi_control', 'sci_control', 'normal_scd', 'wmi_scd', 'sci_scd']
    #colors = ['blanchedalmond', 'gold', 'orange', 'red', 'lightsteelblue', 'cornflowerblue', 'blue', 'darkblue']
    colors = ['blanchedalmond', 'gold', 'orange', 'lightsteelblue', 'cornflowerblue', 'blue']
    
    fig_old, axs_old = plt.subplots(1, len(interest), figsize=(8*len(pt_types), 8))
    fig_young, axs_young = plt.subplots(1, len(interest), figsize=(8*len(pt_types), 8))
    
    subdf = clean_table
    
    subdf_young = subdf[subdf['age'] < 16]
    subdf_old = subdf[subdf['age'] >= 16]
    
    for column, young_ax, old_ax in zip(interest, axs_young, axs_old):
        
        for subdf, age_id, ax in zip([subdf_young, subdf_old], ['u16', 'o16'], [young_ax, old_ax]):
        
            x_multi = [subdf[subdf[ptt]][column] for ptt in pt_types]
            
            
            if not 'norm' in column:
                ax.set_ylabel('Volume (cc)')
                ax.set_ylim([0,1200])
            else:
                ax.set_ylabel('Normalized volume')
                ax.set_ylim([0,1])
            
            bplot = ax.boxplot(x_multi, notch=True, patch_artist=True)
            
            ax.get_xaxis().set_ticks([])
            #ax.set_xticks([i+1 for i,c in enumerate(pt_types)], pt_types)
            ax.set_xticks([i+1 for i,c in enumerate(pt_types)])
            
            ns = [len(x) for x in x_multi]
            the_labels = [f'{ptt} ({n})' for ptt,n in zip(pt_types, ns)]
            
            ax.set_xticklabels(the_labels, rotation=45)
            
            ax.set_title(f'{column} ({age_id})')
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
       
            
            
        
        
    fig_old.tight_layout()
    fig_young.tight_layout()
    old_name = os.path.join(out_folder, 'bars_over16.png')
    young_name = os.path.join(out_folder, 'bars_under16.png')
    
    fig_old.savefig(old_name)
    fig_young.savefig(young_name)
        
    
    
    ######## statistical significance of slopes
          
    interest = ['wm_vol', 'gm_vol', 'supratent']#, 'wm_normal', 'gm_normal', 'supratent_normal']
    pt_type_pairs = [['normal_control', 'normal_scd'],['wmi_control','wmi_scd'],['sci_control','sci_scd']]
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
            for subcolor, subd, icolor, patient_type in zip(subcolors, subdfs, int_colors, pt_type):
                
                
                exes = subd['age']
                whys = subd[col]
                

                
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
                    ax.plot(x, y_model, "-", color=icolor, linewidth=1.5, alpha=0.5, label=patient_type)  
                    
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
                
                ax.scatter(exes, whys, color=subcolor, alpha = 0.4, s=4)
                ax.legend()
               
            z_stat = abs((bs[0] - bs[1]) / np.sqrt(ses[0]**2 + ses[1]**2))
            # Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). Applied multiple regression/correlation analysis for the behavioral sciences (3rd ed.)
            # Paternoster, R., Brame, R., Mazerolle, P., & Piquero, A. R. (1998). Using the Correct Statistical Test for the Equality of Regression Coefficients. Criminology, 36(4), 859–866.
            ax.set_title(f'{pt_type}: {col}\n(zstat = {round(z_stat,2)})')
            ax.set_xlabel('Age (years)')
            
            if 'norm' in col:
                ax.set_ylabel('Normalized volume')
            else:
                ax.set_ylabel('Volume (cc)')
            
            ax.set_xlim(0,50)
            
            if 'norm' in col:
                ax.set_ylim(-0.1,1.1)
            else:
                ax.set_ylim(0,1400)
       
    plt.tight_layout()
    nice_name = os.path.join(out_folder, 'ztesting.png')
    plt.savefig(nice_name)
    
        