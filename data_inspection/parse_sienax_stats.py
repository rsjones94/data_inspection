
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:34:40 2020

@author: skyjones
"""

import os

import pandas as pd
import shutil

stats_file = '/Users/manusdonahue/Documents/Sky/sienax_segmentations/K012/bin/axT1_raw_sienax/report.sienax'
output_csv = '/Users/manusdonahue/Documents/Sky/parsing_testing.csv'

def parse_sienax_stats(stats_file, output_csv):
    
    out_df = pd.DataFrame()
    empty_ser = pd.Series({'short':None, 'long':None, 'value':None, 'units':'mm^3'})
    
    stats_report = open(stats_file)
    
    txt = stats_report.read()
    lines = txt.split('\n')
    
    tissue_types = ['gm', 'wm', 'scaling']
    tissue_long = ['gray_matter_volume', 'white_matter_volume', 'vscaling_factor']
    tissue_lines = [-4, -3, -13]
    
    for tt, tl, line in zip(tissue_types, tissue_long, tissue_lines):
        
        ser = empty_ser.copy()
        
        ser['short'] = tt
        ser['long'] = tl
        
        try:
            the_line = lines[line]
            the_val = float(the_line.split(' ')[-1])
            ser['value'] = the_val
        except TypeError: # if the value of line is None
            pass
        
        
        out_df = out_df.append(ser, ignore_index=True)
        
    out_df = out_df[['short', 'long', 'value', 'units']]
    out_df.to_csv(output_csv, index=False)
        

    


