#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grabs brain volumes for Freesurfer and SIENAX segmentations with follow up
scans and plots them
"""

import os
from glob import glob
import re

import numpy as np
import matplotlib.pyplot as plt

sienax_master = '/Users/manusdonahue/Documents/Sky/brain_volumes/'
freesurfer_master = os.environ['SUBJECTS_DIR']

out_dir = '/Users/manusdonahue/Documents/Sky/'



#####

text_size = 6

def get_fs_stats(f):
    stats_file = os.path.join(f, 'stats', 'aseg.stats')
    
    if not os.path.exists(stats_file):
        print(f'{f} is incomplete. skipping')
        return
    
    
    stats_report = open(stats_file)
    
    txt = stats_report.read()
    lines = txt.split('\n')
    
    wm_line = [i for i in lines if 'Total cerebral white matter volume' in i][0]
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


# first scan, second scan
freesurfer_gms = [[],[]]
freesurfer_wms = [[],[]]
freesurfer_vols = [[],[]]

sienax_gms = [[],[]]
sienax_wms = [[],[]]
sienax_vols = [[],[]]

freesurfer_folders = np.array(glob(os.path.join(freesurfer_master, '*/'))) # list of all possible subdirectories
sienax_folders = np.array(glob(os.path.join(sienax_master, '*/'))) # list of all possible subdirectories


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
  

    
sienax_artist = plt.Circle((0,0), color='green')
fs_artist = plt.Circle((0,0), color='blue')

for sienax_list, freesurfer_list, title, lims, outname, offset in zip((sienax_wms, sienax_gms, sienax_vols),
                                              (freesurfer_wms, freesurfer_gms, freesurfer_vols),
                                              ('White matter volume', 'Grey matter volume', 'Total volume'),
                                              ([0,650],[0,900],[0,1600]),
                                              ('white','grey','total'),
                                              (5, 7, 14)):
    
    fig = plt.figure(figsize=(15,30))
    lims = np.array(lims)
    line_x = lims
    line_y = lims
    
    upper_y = line_y * 1.05    
    
    lower_y = line_y * .95
    
    plt.plot(line_x,line_y, color='black')
    plt.plot(line_x,upper_y, color='grey', linestyle='dashed')
    plt.plot(line_x,lower_y, color='grey', linestyle='dashed')
    
    plt.scatter(freesurfer_list[0], freesurfer_list[1], color='blue')
    plt.scatter(sienax_list[0], sienax_list[1], color='green')
    plt.ylabel('Follow up volume (cc)')
    plt.xlabel('Initial scan volume (cc)')
    
    freesurfer_in = len(freesurfer_pairs)
    for label, x, y in zip(freesurfer_pairs, freesurfer_list[0], freesurfer_list[1]):
        change = y / x
        if change > 1.05 or change < 0.95:
            the_label = f'{label[0]} : {round(change,2)}'
            plt.scatter([x], [y], marker='_', color='red')
            plt.annotate(the_label, (x+offset, y-offset), size=text_size, color='blue')
            freesurfer_in -= 1
            
    sienax_in = len(sienax_pairs)
    for label, x, y in zip(sienax_pairs, sienax_list[0], sienax_list[1]):
        change = y / x
        if change > 1.05 or change < 0.95:
            the_label = f'{label[0]} : {round(change,2)}'
            plt.scatter([x], [y], marker='_', color='red')
            plt.annotate(the_label, (x+offset, y-offset), size=text_size, color='green')
            sienax_in -= 1
            
    
    
    
    plt.title(f'{title}\nFS:{freesurfer_in}/{len(freesurfer_pairs)}:{round(freesurfer_in/len(freesurfer_pairs), 2)}, SIENAX:{sienax_in}/{len(sienax_pairs)}:{round(sienax_in/len(sienax_pairs), 2)}')
    
    plt.legend((sienax_artist, fs_artist),
               ('SIENAX', 'Freesurfer'))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    figname = os.path.join(out_dir, f'{outname}.png')
    plt.savefig(figname)
    
    
    
    
    
    
