
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
from parse_sienax_stats import parse_sienax_stats

exclude_pts = ['SCD_K065', 'SCD_TRANSP_P001_01']

in_csv = '/Users/manusdonahue/Documents/Sky/stroke_status.csv'
out_folder = '/Users/manusdonahue/Documents/Sky/freesurfer_volume_visualization/'
parsed_folder = '/Users/manusdonahue/Documents/Sky/freesurfer_volume_visualization/parsed'

brain_vol_csv = '/Users/manusdonahue/Documents/Sky/normal_brain_vols.csv' # from Borzage, Equations to describe brain size across the continuum of human lifespan (2012)
# values originally reported as mass in g, converted to cc assuming rho = 1.04 g/cc

fs_folder = '/Volumes/DonahueDataDrive/freesurfer_subjects_scd/'

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
        
        """
        if in_table_indexed.loc[pt_name]['mri1_wml_drp'] == 1:
            working['white_matter_injury'] = 1
        else:
            working['white_matter_injury'] = 0
        """
        
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
        
        
        for ind in parsed_csv.index:
            try: # if the index is an integer (or can be coerced to one) then the value is for a granular segmentation volume
                num = int(ind)
                keyname = parsed_csv.loc[ind]['long']
                val = parsed_csv.loc[ind]['value']
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
        
    clean_table['normal_control'] = [all([i, not j]) for i,j in zip(clean_table['control'], clean_table['sci'])]        
    clean_table['sci_control'] = [all([i, j]) for i,j in zip(clean_table['control'], clean_table['sci'])]   
    
    
    clean_table['normal_scd'] = [all([i, not j]) for i,j in zip(clean_table['scd'], clean_table['sci'])]        
    clean_table['sci_scd'] = [all([i, j]) for i,j in zip(clean_table['scd'], clean_table['sci'])]

    
    ######## statistical significance of slopes
    ''' 
    interest = ['gm_normal', 'wm_normal', 'supratent_normal', 'total_normal'] # 'gm_vol', 'wm_vol', 'supratent', 'total_vol'], 
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
    '''
    
    
    
    # multiple linear regression looking at brain vol vs (age, gender, scd status)
    
    factors = ['gm_vol', 'wm_vol', 'supratent', 'total_vol', 'gm_normal', 'wm_normal', 'supratent_normal', 'total_normal', 'etiv']
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
        
        
    print('QUAD')
    for f in factors:
        print(f'\n\n\nFACTOR: {f}\n')
        X = clean_table[['age','gender','scd', 'sci']]
        Y =  clean_table[f]
        
        X2 = sm.add_constant(X)
        est = sm.OLS(Y, X2)
        est2 = est.fit()
        print(est2.summary())
        
     
    to_explore = []
    
    explore_factors_raw = [f for f in clean_table if '-' in f]
    explore_factors = []
    
    for f in explore_factors_raw:
        fn = f+'_brainfrac'
        clean_table[fn] = (clean_table[f]/1e3) / clean_table['total_vol']
        explore_factors.append(f)
        explore_factors.append(fn)
    
    
    for f in explore_factors:
        print(f'\n\n\nEXPLORATORY FACTOR: {f}')
        X = clean_table[['age','gender','scd', 'sci']]
        Y =  clean_table[f]
        
        X2 = sm.add_constant(X)
        est = sm.OLS(Y, X2)
        est2 = est.fit()
        
        for key, val in est2.pvalues.items():
            if key not in ['scd']:
                continue
            if val <= 0.05:
                print(f'SIGNIFICANT. {key}: pval={round(val,2)}')
                to_explore.append((f,val))
            elif val <= 0.1:
                print(f'POTENTIAL. {key}: pval={round(val,2)}')
                to_explore.append((f,val))
                
    
    explore_folder = os.path.join(out_folder, 'explore')
    for key, pval in to_explore:
        figname = os.path.join(explore_folder, f'{key}.png')
        
        fig, ax = plt.subplots(1,1)
        for status, color in zip(['control', 'scd'], ['red','blue']):
            expr = clean_table[status] == 1
            subdf = clean_table[expr]
            exes = subdf['age']
            whys = subdf[key]
            ax.scatter(exes, whys, color=color, alpha = 0.4, s=4, label=status)
            ax.set_title(f'{key} (pval={round(pval,4)})')
            ax.set_xlabel('Age (years)')
            if '_brainfrac' not in key:
                ax.set_ylabel('Volume (cu mm)')
            else:
                ax.set_ylabel('Fraction of total brain volume')
            ax.legend()
            
            
                
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
                ax.plot(x, y_model, "-", color=color, linewidth=1.5, alpha=0.5) 
                
                x2 = np.linspace(np.min(x), np.max(x), 100)
                y2 = equation(p, x2)
                
                # Confidence Interval (select one)
                plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax, color=color)
                #plot_ci_bootstrap(x, y, resid, ax=ax)
                
                # Prediction Interval
                pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))  
                ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
                ax.plot(x2, y2 - pi, "--", color=color, alpha=0.2)#, label="95% Prediction Limits")
                ax.plot(x2, y2 + pi, "--", color=color, alpha=0.2)
            except np.linalg.LinAlgError:
                print('Linear algebra error, likely due to singular matrix')
                pass
                
            ax.scatter(exes, whys, color=color, alpha = 0.4, s=4)
            
            
            plt.savefig(figname)
        
            
            
            
            
            

        