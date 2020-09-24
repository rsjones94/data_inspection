
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:34:40 2020

@author: skyjones
"""

import os

import pandas as pd
import shutil

stats_file = '/Users/manusdonahue/Documents/freesurfer_subjects/SCD_K001_02/stats/aseg.stats'
output_csv = '/Users/manusdonahue/Documents/Sky/parsing_testing.csv'

def parse_freesurfer_stats(stats_file, output_csv):
    pass

out_df = pd.DataFrame()
empty_ser = pd.Series({'short':None, 'long':None, 'value':None, 'units':None})

stats_report = open(stats_file)

txt = stats_report.read()
lines = txt.split('\n')

part_1_indices = range(14,34)
for i in part_1_indices:
    the_line = lines[i]
    parts = the_line.split(', ')
    
    ser = empty_ser.copy()
    
    short_name = parts[1]
    long_name = parts[2]
    value = float(parts[3])
    units = parts[4]

    ser['short'] = short_name
    ser['long'] = long_name
    ser['value'] = value
    ser['units'] = units
    
    out_df = out_df.append(ser, ignore_index=True)
    
out_df = out_df.append(empty_ser, ignore_index = True)
    
part_2_indices = range(79,124)
for i in part_2_indices:
    the_line = lines[i]
    parts = the_line.split(' ')
    parts = [i for i in parts if i != '']
    
    ser = empty_ser.copy()
    
    short_name = parts[1]
    long_name = parts[4]
    value = float(parts[3])
    units = 'mm^3'
    
    ser['short'] = short_name
    ser['long'] = long_name
    ser['value'] = value
    ser['units'] = units
    
    out_df = out_df.append(ser, ignore_index=True)
    
out_df = out_df[['short', 'long', 'value', 'units']]
out_df.to_csv(output_csv, index=False)
    

    


