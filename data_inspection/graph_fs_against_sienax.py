#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grabs brain volumes for Freesurfer and SIENAX segmentations with follow up
scans and plots them
"""

import os
from glob import glob
import re
import itertools

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import stats

sienax_master = '/Users/manusdonahue/Documents/Sky/brain_volumes/'
fast_master = '/Users/manusdonahue/Documents/Sky/volume_testing/'
freesurfer_master = os.environ['SUBJECTS_DIR']

out_dir = '/Users/manusdonahue/Documents/Sky/vol_comp_charts'



#####

bound_size = 0.05
text_size = 6

def bland_altman_plot(data1, data2, *args, **kwargs):
    """
    Based on Neal Fultz' answer on Stack Overflow
    """
    
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    
    plt.annotate(f'Mean diff: {round(md,2)}', (min(mean),md+5))
    plt.annotate(f'-SD 1.96: {round(md-1.96*sd,2)}', (min(mean),md-1.96*sd+5))
    plt.annotate(f'+SD 1.96: {round(md+1.96*sd,2)}', (min(mean),md+1.96*sd+5))

def get_fs_stats(f):
    stats_file = os.path.join(f, 'stats', 'aseg.stats')
    
    if not os.path.exists(stats_file):
        print(f'{f} is incomplete. skipping')
        return
    
    
    stats_report = open(stats_file)
    
    txt = stats_report.read()
    lines = txt.split('\n')
    
    wm_line = [i for i in lines if 'Total cerebral white matter volume' in i][0] # ...cerebral white matter volume????
    gm_line = [i for i in lines if 'Total gray matter volume' in i][0]
    icv_line = [i for i in lines if 'Estimated Total Intracranial Volume' in i][0]
    
    wm_val = float(wm_line.split(', ')[-2]) / 1e3
    gm_val = float(gm_line.split(', ')[-2]) / 1e3
    icv_val = float(icv_line.split(', ')[-2]) / 1e3
    
    trans_mat_file = os.path.join(f, 'mri', 'transforms', 'talairach.xfm') 
    trans_report = open(trans_mat_file)
    trans_txt = trans_report.read()
    trans_lines = trans_txt.split('\n')
    
    mat_as_text = trans_lines[-4:-1]
    mat = [[float(a) for a in re.split(';| ', i) if a != ''] for i in mat_as_text]
    mat.append([0, 0, 0, 1])
    mat = np.array(mat)
    
    det = np.linalg.det(mat)
    
    return wm_val, gm_val, icv_val, det


def get_sienax_stats(f):
    sienax_report = open(os.path.join(f, 'bin', 'axT1_raw_sienax', 'report.sienax'))
        
    txt = sienax_report.read()
    lines = txt.split('\n')
    
    greys = lines[-4]
    whites = lines[-3]
    brains = lines[-2]
    
    grey_vol_raw = float(greys.split(' ')[-1]) / 1e3
    
    white_vol_raw = float(whites.split(' ')[-1]) / 1e3
    
    brain_vol_raw = float(brains.split(' ')[-1]) / 1e3
    
    return white_vol_raw, grey_vol_raw


def get_fast_stats(f):
    
    fast_folder = os.path.join(f, 'comp', 'fast')
    fast_pve_path = os.path.join(fast_folder, 'fast_pveseg.nii.gz')

    
    try:
        raw = nib.load(fast_pve_path)
    except FileNotFoundError:
        print(f'{fast_pve_path} does not exist')
        return
    img = raw.get_fdata()
    
    header = raw.header
    voxel_dims = header['pixdim'][1:4]
    voxel_vol = np.product(voxel_dims)
    
    # 1 = csf, 2 = gm, 3 = wm
    # use partial voluems for calculation
    seg_types = {1: 'csf', 2: 'gm', 3:'wm'}
    
    fast_sub = {'gm': None,
                'wm': None,
                'csf': None}
    
    for num, matter_type in seg_types.items():
        
        subnum = num-1
        subseg_file = os.path.join(fast_folder, f'fast_pve_{subnum}.nii.gz')
        subraw = nib.load(subseg_file)
        subim = subraw.get_fdata()
        
        
        vol = float(subim.sum() * voxel_vol) / 1e3
        
        fast_sub[matter_type] = vol
        
    return fast_sub['wm'], fast_sub['gm']


# first scan, second scan
freesurfer_gms = [[],[]]
freesurfer_wms = [[],[]]
freesurfer_vols = [[],[]]

sienax_gms = [[],[]]
sienax_wms = [[],[]]
sienax_vols = [[],[]]

fast_gms = [[],[]]
fast_wms = [[],[]]
fast_vols = [[],[]]

freesurfer_folders = np.array(glob(os.path.join(freesurfer_master, '*/'))) # list of all possible subdirectories
sienax_folders = np.array(glob(os.path.join(sienax_master, '*/'))) # list of all possible subdirectories
fast_folders = np.array(glob(os.path.join(fast_master, '*/'))) # list of all possible subdirectories


### get freesurfer data

## first find pairs
freesurfer_pairs = []
for i, f in enumerate(freesurfer_folders):
    subject_name = os.path.basename(os.path.normpath(f))
    
    if subject_name[-3:] != '_01' or '-' in subject_name:
        continue
    else:
        basename = subject_name.replace('_01', '')
    
    follow_name = basename+'_02'
    follow_path = os.path.join(freesurfer_master, follow_name)
    if not os.path.exists(follow_path):
        continue

    try:
        wm1, gm1, icv, det1 = get_fs_stats(f)
        wm2, gm2, icv2, det2 = get_fs_stats(follow_path)
    except TypeError:
        continue
    
    vol1 = wm1 + gm1
    vol2 = wm2 + gm2
    
    freesurfer_wms[0].append(wm1)
    freesurfer_gms[0].append(gm1)
    freesurfer_vols[0].append(vol1)
    
    freesurfer_wms[1].append(wm2)
    freesurfer_gms[1].append(gm2)
    freesurfer_vols[1].append(vol2)
    
    
    freesurfer_pairs.append([subject_name, follow_name])
    
### get sienax data

## first find pairs
sienax_pairs = []
for i, f in enumerate(sienax_folders):
    subject_name = os.path.basename(os.path.normpath(f))
    
    if subject_name[-3:] != '_01' or '-' in subject_name:
        continue
    else:
        basename = subject_name.replace('_01', '')
    
    follow_name = basename+'_02'
    follow_path = os.path.join(sienax_master, follow_name)
    if not os.path.exists(follow_path):
        continue

    try:
        wm1, gm1 = get_sienax_stats(f)
        wm2, gm2 = get_sienax_stats(follow_path)
    except TypeError:
        continue
    
    vol1 = wm1 + gm1
    vol2 = wm2 + gm2
    
    sienax_wms[0].append(wm1)
    sienax_gms[0].append(gm1)
    sienax_vols[0].append(vol1)
    
    sienax_wms[1].append(wm2)
    sienax_gms[1].append(gm2)
    sienax_vols[1].append(vol2)
    
    
    sienax_pairs.append([subject_name, follow_name])
    

### get fast data

## first find pairs
fast_pairs = []
for i, f in enumerate(fast_folders):
    subject_name = os.path.basename(os.path.normpath(f))
    
    if subject_name[-3:] != '_01' or '-' in subject_name:
        continue
    else:
        basename = subject_name.replace('_01', '')
    
    follow_name = basename+'_02'
    follow_path = os.path.join(fast_master, follow_name)
    if not os.path.exists(follow_path):
        continue

    try:
        wm1, gm1 = get_fast_stats(f)
        wm2, gm2 = get_fast_stats(follow_path)
    except TypeError:
        continue
    
    vol1 = wm1 + gm1
    vol2 = wm2 + gm2
    
    fast_wms[0].append(wm1)
    fast_gms[0].append(gm1)
    fast_vols[0].append(vol1)
    
    fast_wms[1].append(wm2)
    fast_gms[1].append(gm2)
    fast_vols[1].append(vol2)
    
    
    fast_pairs.append([subject_name, follow_name])
  

    
sienax_artist = plt.Circle((0,0), color='green')
fs_artist = plt.Circle((0,0), color='blue')
fast_artist = plt.Circle((0,0), color='orange')


for sienax_list, freesurfer_list, fast_list, title, lims, outname, offset in zip((sienax_wms, sienax_gms, sienax_vols),
                                              (freesurfer_wms, freesurfer_gms, freesurfer_vols),
                                              (fast_wms, fast_gms, fast_vols),
                                              ('White matter volume', 'Grey matter volume', 'Total volume'),
                                              ([0,650],[0,900],[0,1600]),
                                              ('white','grey','total'),
                                              (5, 7, 14)):
    
    fig = plt.figure(figsize=(15,30))
    lims = np.array(lims)
    line_x = lims
    line_y = lims
    
    upper_y = line_y * (1+bound_size)  
    
    lower_y = line_y * (1-bound_size)
    
    plt.plot(line_x,line_y, color='black', alpha = 0.3)
    plt.plot(line_x,upper_y, color='grey', linestyle='dashed', alpha = 0.3)
    plt.plot(line_x,lower_y, color='grey', linestyle='dashed', alpha = 0.3)
    
    plt.scatter(freesurfer_list[0], freesurfer_list[1], color='blue', alpha=0.4)
    plt.scatter(sienax_list[0], sienax_list[1], color='green', alpha=0.4)
    plt.scatter(fast_list[0], fast_list[1], color='orange', alpha=0.4)
    plt.ylabel('Follow up volume (cc)')
    plt.xlabel('Initial scan volume (cc)')
    
    fs_slope, fs_intercept, fs_r, fs_p, fs_stderr = stats.linregress(freesurfer_list[0], freesurfer_list[1])
    sienax_slope, sienax_intercept, sienax_r, sienax_p, sienax_stderr = stats.linregress(sienax_list[0], sienax_list[1])
    fast_slope, fast_intercept, fast_r, fast_p, fast_stderr = stats.linregress(fast_list[0], fast_list[1])
    
    fs_why = [fs_slope*i+fs_intercept for i in line_x]
    sienax_why = [sienax_slope*i+sienax_intercept for i in line_x]
    fast_why = [fast_slope*i+fast_intercept for i in line_x]
    
    plt.plot(line_x,fs_why,color='blue', alpha=0.7)
    plt.plot(line_x,sienax_why,color='green', alpha=0.7)
    plt.plot(line_x,fast_why,color='orange', alpha=0.7)
    
    
    labels_of_interest = []
    freesurfer_in = len(freesurfer_pairs)
    for i, (label, x, y) in enumerate(zip(freesurfer_pairs, freesurfer_list[0], freesurfer_list[1])):
        change = y / x
        if change > (1+bound_size) or change < (1-bound_size):
            if change > (1+bound_size):
                hor_align = 'right'
                ver_align = 'bottom'
                realoffset = offset*-1
            else:
                hor_align = 'left'
                ver_align = 'top'
                realoffset = offset
                
            
            the_label = f'{label[0]} : {round(change,2)}'
            plt.scatter([x], [y], marker='_', color='red')
            plt.annotate(the_label, (x+realoffset, y-realoffset), size=text_size, color='blue', ha=hor_align, va=ver_align)
            freesurfer_in -= 1
            labels_of_interest.append(label)
            
    sienax_in = len(sienax_pairs)
    for i, (label, x, y) in enumerate(zip(sienax_pairs, sienax_list[0], sienax_list[1])):
        change = y / x
        if change > (1+bound_size) or change < (1-bound_size):
            if change > (1+bound_size):
                hor_align = 'right'
                ver_align = 'bottom'
                realoffset = offset*-1
            else:
                hor_align = 'left'
                ver_align = 'top'
                realoffset = offset
                
            the_label = f'{label[0]} : {round(change,2)}'
            plt.scatter([x], [y], marker='_', color='red')
            plt.annotate(the_label, (x+realoffset, y-realoffset), size=text_size, color='green', ha=hor_align, va=ver_align)
            sienax_in -= 1
            labels_of_interest.append(label)
            
    fast_in = len(fast_pairs)
    for i, (label, x, y) in enumerate(zip(fast_pairs, fast_list[0], fast_list[1])):
        change = y / x
        if change > (1+bound_size) or change < (1-bound_size):
            if change > (1+bound_size):
                hor_align = 'right'
                ver_align = 'bottom'
                realoffset = offset*-1
            else:
                hor_align = 'left'
                ver_align = 'top'
                realoffset = offset
                
            the_label = f'{label[0]} : {round(change,2)}'
            plt.scatter([x], [y], marker='_', color='red')
            plt.annotate(the_label, (x+realoffset, y-realoffset), size=text_size, color='orange', ha=hor_align, va=ver_align)
            fast_in -= 1
            labels_of_interest.append((label))
            
    unique_labels = []
    for i in labels_of_interest:
        if i not in unique_labels:
            unique_labels.append(i)
    
    for label in unique_labels:
        try:
            sienax_i = sienax_pairs.index(label)
        except ValueError:
            sienax_i = None
        try:
            fast_i = fast_pairs.index(label)
        except ValueError:
            fast_i = None
        try:
            freesurfer_i = freesurfer_pairs.index(label)
        except ValueError:
            freesurfer_i = None
        
        exwhys = []
        for index, li in zip((sienax_i, fast_i, freesurfer_i),(sienax_list, fast_list, freesurfer_list)):
            try:
                xy = [li[0][index], li[1][index]]
                exwhys.append(xy)
            except TypeError:
                print(f'Label {label} not found')
                
        indices = [i for i in range(len(exwhys))]
        combs = itertools.combinations(indices, 2)
        for i1, i2 in combs:
            the_ex = [exwhys[i1][0], exwhys[i2][0]]
            the_why = [exwhys[i1][1], exwhys[i2][1]]
            plt.plot(the_ex, the_why, color='darkred', alpha=0.5)
        
    
    
    
    plt.title(f'{title}\nFS:{freesurfer_in}/{len(freesurfer_pairs)}:{round(freesurfer_in/len(freesurfer_pairs), 2)}, SIENAX:{sienax_in}/{len(sienax_pairs)}:{round(sienax_in/len(sienax_pairs), 2)}, FAST:{fast_in}/{len(fast_pairs)}:{round(fast_in/len(fast_pairs), 2)}')
    
    plt.legend((sienax_artist, fs_artist, fast_artist),
               (f'SIENAX: y = {round(sienax_slope,2)}*x + {round(sienax_intercept,2)}', 
                f'Freesurfer: y = {round(fs_slope,2)}*x + {round(fs_intercept,2)}', 
                f'FAST: y = {round(fast_slope,2)}*x + {round(fast_intercept,2)}'))
    plt.xlim(lims)
    plt.ylim(lims)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    figname = os.path.join(out_dir, f'{outname}.png')
    plt.savefig(figname)
    
    for li, prog_name in zip((freesurfer_list, sienax_list, fast_list),
                         ('Freesurfer', 'SIENAX', 'FAST')):
        plt.figure()
        bland_altman_plot(li[0], li[1])
        plt.title(f'Bland-Altman: {prog_name}, {title}')
        alt_outname = f'{outname}_bland_{prog_name}'
        alt_figname = os.path.join(out_dir, f'{alt_outname}.png')
        plt.xlabel('Average of paired observations (cc)')
        plt.ylabel('Difference of paired observations (cc)')
        plt.savefig(alt_figname)



    
    
