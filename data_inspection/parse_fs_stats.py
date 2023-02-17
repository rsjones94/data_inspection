
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:34:40 2020

@author: skyjones
"""

import os

import pandas as pd
import shutil

def parse_freesurfer_stats(stats_file, output_csv=None):
    
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
    if output_csv:
        out_df.to_csv(output_csv, index=False)
        
    return out_df

def parse_freesurfer_stats_lobular(stats_file, output_csv=None):
    
    out_df = pd.DataFrame()
    empty_ser = pd.Series({'short':None, 'long':None, 'value':None, 'units':None})
    
    stats_report = open(stats_file)
    
    txt = stats_report.read()
    lines = txt.split('\n')
    
    checkline = '# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd'
    checker = [i for i,j in enumerate(lines) if j==checkline]
    start_ind = checker[-1]+1
    
    part_2_indices = range(start_ind,len(lines)-1)
    for i in part_2_indices:
        the_line = lines[i]
        parts = the_line.split(' ')
        parts = [i for i in parts if i != '']
        
        ser = empty_ser.copy()
        
        short_name = parts[0]
        long_name = None
        value = float(parts[3])
        units = 'mm^3'
        
        ser['short'] = short_name
        ser['long'] = long_name
        ser['value'] = value
        ser['units'] = units
        
        out_df = out_df.append(ser, ignore_index=True)
        
    out_df = out_df[['short', 'long', 'value', 'units']]
    if output_csv:
        out_df.to_csv(output_csv, index=False)
        
    return out_df
        

    


