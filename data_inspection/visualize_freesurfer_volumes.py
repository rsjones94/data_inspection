
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
                  'age':None,
                  'stroke_silent':None,
                  'stroke_overt':None,
                  'sci':None,
                  'transf':None,
                  'scd':None,
                  'anemia':None,
                  'control':None,
                  'gender':None}
    for i, csv in enumerate(parsed_csvs):
        pt_name = os.path.basename(os.path.normpath(csv))[:-4]
        print(f'Parsing {pt_name} ({i+1} of {len(parsed_csvs)})')
        
        working = pd.Series(blank_dict.copy())
        parsed_csv = pd.read_csv(csv, index_col='short')
        
        working['mr_id'] = pt_name
        if pt_name in exclude_pts:
            continue
        working['gm_vol'] = parsed_csv.loc['TotalGrayVol']['value'] / 1e3
        working['total_vol'] = parsed_csv.loc['BrainSegVolNotVent']['value'] / 1e3
        working['wm_vol'] = working['total_vol'] - working['gm_vol']
        working['age'] = in_table_indexed.loc[pt_name]['age']
        
        stroke_overt = in_table_indexed.loc[pt_name]['mh_rf_prior_stroke_overt']
        stroke_silent = in_table_indexed.loc[pt_name]['mh_rf_prior_stroke_silent']
        
        if stroke_overt == 1 or stroke_silent == 1:
            continue
        
        sci = in_table_indexed.loc[pt_name]['outcome_mri1_sci']
        transf = in_table_indexed.loc[pt_name]['enroll_sca_transfusion']
        
        if transf == 1:
            continue
        
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
    brain_vol_df = pd.read_csv(brain_vol_csv)
    
    collated_csv = os.path.join(out_folder, 'collated.csv')
    clean_table = pd.read_csv(collated_csv, index_col='mr_id')
    
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
    
    fig, ax = plt.subplots(figsize=(10,10))
    plt.ylabel('Age (years)')
    ax.get_xaxis().set_ticks([])
    bplot = plt.boxplot(x_multi, notch=True, patch_artist=True)
    plt.xticks([i+1 for i,c in enumerate(names)], names)
    plt.title('Age distribution in SCD study cohorts')
    box_out = os.path.join(out_folder, 'age_box.png')
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    plt.savefig(box_out)
    
    y_names = ['wm_vol', 'gm_vol', 'total_vol', 'total_vol_custom', 'mask_vol_custom']

    xlims = [0, 50]
    ylims = [0, 1800]
    
    genders = [[1,'male'], [0,'female']]
    for g, g_name in genders:
        gender_table = clean_table[clean_table['gender'] == g]
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
            plt.ylabel('Volume (cc)')
                   
            if y_name in ['total_vol', 'total_vol_custom', 'mask_vol_custom']:
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
            plt.xlim(xlims)
            plt.ylim(ylims)
            
            plt.legend(artists, artist_descs)
            
            plt.tight_layout()
            plt.savefig(plot_name)
            
        
        
        