#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for data_inspection. This script reads in
tabular patient data and analyzes it for outliers. First, it inspects specified
columns for data integrity (missing values) and produces histograms if appropriate.

Then it analyzes specified 2d relationships, producing scatter plots and identifying
outliers.

Finally it runs the DBSCAN algorithm to flag any potential outliers.

Note that on my machine this uses the venv "tabular_analysis"
"""

import os
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from support import is_empty, numbery_string_to_number


data_path = r'/Users/manusdonahue/Documents/Sky/SCD_pt_data_labels_piped.csv'
out_folder = r'/Users/manusdonahue/Documents/Sky/data_inspection/analysis' # should not exist

# column that contains the unique deidentified patient ID
study_id_col = 'Study ID'

# columns we want to inspect for completeness and produce histograms/barplots for
# each key is a column name, and the value is True if there MUST be a value and
# False if there does not need to be a value. If there must be a value if and
# only if another column(s) is filled, then the value should be a list of those columns
single_cols = {
               'Age': True,
               'Race': True,
               'Hemoglobin genotype': True,
               'Gender': True,
               'BMI': True,
               'Specify total HU daily dosage (mg)': True,
               'HTN': True,
               'Diabetes': True,
               'Coronary artery disease': True,
               'High cholesterol': True,
               'Hgb': True,
               'Hct/PCV': True,
               'MRI 1 -  Pulse ox results': True, # note the extra space
               'MRI 2 - Pulse ox results': True,
               'MRI 3 - Pulse ox results': True,
               'MCV': True,
               'Receiving regular blood transfusions': True,
               r'Initial hemoglobin S% (pretransfusion if applicable)': True,
               r'Results': True, # this is posttransfusion HbS%, and it's amazing it's the only column with this name
               r'MRI 1 - SBP': True,
               r'MRI 1 - DBP': True,
               r'MRI 2 - SBP': True,
               r'MRI 2 - DBP': True,
               r'MRI 3 - SBP': True,
               r'MRI 3 - DBP': True,
               }

# 2d relationships we want to use to check for outliers. [independent, dependent]
# numeric data only pls
double_cols = [['Specify total HU daily dosage (mg)', 'MCV'],
               ['Specify total HU daily dosage (mg)', 'Initial hemoglobin S% (pretransfusion if applicable)'],
               ['Age', 'MRI 1 - SBP'],
               ['Age', 'MRI 1 - DBP'],
               ['Age', 'MRI 2 - SBP'],
               ['Age', 'MRI 2 - DBP'],
               ['Age', 'MRI 3 - SBP'],
               ['Age', 'MRI 3 - DBP']]

contam = 0.07 # estimated % of data that are outliers

np.random.seed(1)

#######################################

###### setup

mono_folder = os.path.join(out_folder, 'mono')
bi_folder = os.path.join(out_folder, 'bi')
# multi_folder = os.path.join(out_folder, 'multi')
custom_folder = os.path.join(out_folder, 'custom')

overview_report = os.path.join(out_folder, 'overview.txt')
missing_data_report = os.path.join(out_folder, 'missing_data.csv')
outliers_report = os.path.join(out_folder, 'outliers.csv')

try:
    os.mkdir(out_folder)
except FileExistsError:
    no_answer = True
    while no_answer:
        ans = input('The output directory exists. Overwrite? [y/n]\n')
        if ans == 'y':
            no_answer = False
            shutil.rmtree(out_folder)
            os.mkdir(out_folder)
        elif ans == 'n':
            raise FileExistsError('File exists. Process aborted')
        else:
            print('Response must be "y" or "n"')


log_file = os.path.join(out_folder, 'log.txt')
log = open(log_file, 'w')
    
os.mkdir(mono_folder)
os.mkdir(bi_folder)
# os.mkdir(multi_folder)
os.mkdir(custom_folder)

df = pd.read_csv(data_path, sep='|', low_memory=False, dtype={study_id_col:'object'})

problem_pts_cols = [study_id_col]
problem_pts_cols.extend(single_cols.keys())
problem_pts = pd.DataFrame(columns=problem_pts_cols)
problem_pts = problem_pts.set_index('Study ID') # this data will relate pt IDs to a list of columns for which data
# is missing, iff that missing data is marked as essential (by the variable single_cols)

outlier_pts = {} # this data will relate pt IDs to a list of columns for which
# the data seems to be an outlier

###### plot and inspect the monodimensional data
problem_patients_dict = {}
for col in single_cols:
    data = df[col]
    pts = df[study_id_col]
    plt.figure(figsize=(8,12))
    plt.title(col)
    
    print(f'Plotting: {col}. dtype is {data.dtype}')
    if data.dtype == 'object':
        counts = Counter(data)
        if np.nan in counts:
            counts['nan'] = counts[np.nan]
            del counts[np.nan]
        n_v = [(n,v) for n,v in counts.most_common()]
        names = [n for n,v in n_v]
        values = [v for n,v in n_v]
        plt.ylabel('Count')
        plt.bar(names, values)
    else:
        # plt.hist(data)
        data_drop = data.dropna()
        result = plt.boxplot(data_drop, notch=True)
        plt.ylabel('Value')
        points = result['fliers'][0].get_data()
        exes = points[0]+.01
        whys = points[1]
        for x,y in zip(exes,whys):
            matches = pts[data == y]
            label = ''
            for m in matches:
                label += f'{m} + '
            label = label[:-3]
            plt.annotate(label, (x,y), fontsize=8)
        # plt.xlabel('Value')
    
    scrub_col = col.replace('/', '-') # replace slashes with dashes to protect filepath
    fig_name = os.path.join(mono_folder, f'{scrub_col}.png')
    plt.savefig(fig_name)
    plt.close()

    print(f'Evaluating completeness')
    for i, row in df.iterrows():
        # explicit comparisons of bools needed because we are exploiting the ability to mix key datatypes
        if not is_empty(row[col]):
            has_data = True
            # print('Is not empty')
        elif single_cols[col] is False:
            has_data = True
            # print('Does not need data')
        elif single_cols[col] is True: # if data is required
            has_data = False
            # print('Does not have data and deffo needs it')
        else: # if we get here, need to see if the companion columns are filled
            # if all companion columns are filled, then data is required
            companions = [row[c] for c in single_cols[col]]
            has_required_companions = all([not is_empty(row[c]) for c in single_cols[col]])
            has_data = not has_required_companions
            
        if not has_data:
            pt_id = row[study_id_col]
            try:
                problem_patients_dict[pt_id].append(col)
            except KeyError:
                problem_patients_dict[pt_id] = [col]
            
            
# write the missing data report
for pt, cols in problem_patients_dict.items():
    insert = pd.Series({col:1 for col in cols}, name=pt)
    problem_pts = problem_pts.append(insert, ignore_index=False)
problem_pts = problem_pts.sort_index()
problem_pts.to_csv(missing_data_report)

print('\n')
###### do the 2d analyses
for ind_col, dep_col in double_cols:
    print(f'2d: {ind_col} and {dep_col}')
    fig_name = os.path.join(bi_folder, f'{dep_col}-v-{ind_col}.png')
    plt.figure()
    plt.title(f'{dep_col} vs. {ind_col}')
    
    x = df[ind_col]
    y = df[dep_col]
    pt_id = df[study_id_col]
    
    try:
        
        x = [numbery_string_to_number(i) for i in x]
        y = [numbery_string_to_number(i) for i in y]
        
        
        data = np.array( [np.array( [a,b] ) for a,b,c in zip(x,y,pt_id) if all([not np.isnan(a), not(np.isnan(b))]) ] )
        pts = [ c for a,b,c in zip(x,y,pt_id) if all([not np.isnan(a), not(np.isnan(b))]) ]
        clf = IsolationForest(max_samples='auto', random_state=1, contamination=contam)
        preds = clf.fit_predict(data)
        
        x = data[:,0]
        y = data[:,1]
        
        plt.scatter(x, y, c=preds)
        
        for pt, x, y, p in zip(pts, x, y, preds):
            if p == -1:
                plt.annotate(pt, (x,y))

        plt.xlabel(ind_col)
        plt.ylabel(dep_col)
        plt.savefig(fig_name)
        plt.close()
    except ValueError as e:
        print(f'Error analyzing -{ind_col}- against -{dep_col}-')
        log.write(f'Error analyzing -{ind_col}- against -{dep_col}-:\n\t{e}\n')
        plt.close()
        continue
    
###### multivariate outlier detection
        
# figure out which columns are numeric
numeric_cols = [c for c in df.columns if df[c].dtype != 'object']



###### custom analyses

# see whose HbS actually increased postransfusion

fig_name = os.path.join(custom_folder, f'anomalous_posttransfusion_HbS_increases.png')
plt.figure(figsize=(8,30))
plt.title(r'Post- vs. Pre-transfusion HbS %')
plt.xlabel(r'Transfusion status')
plt.ylabel(r'% HbS')

text_size = 7
for status, pre, post, pt in zip(df['Receiving regular blood transfusions'], df['Initial hemoglobin S% (pretransfusion if applicable)'], df['Results'], df[study_id_col]):
    if status == 'No':
        continue
    
    if pd.isnull(post) and not pd.isnull(pre):
        bad = -1
        col='orange'
        al = 1
    elif post >= pre:
        bad = 2
        col ='red'
        al = 1
    elif post >= pre*.9:
        bad = 1
        col = 'blue'
        al = 1
    else:
        bad = 0
        col = 'green'
        al = 0.2
    
    exes = [0,1]
    whys = [pre,post]
    
    plt.scatter(exes, whys, color=col, alpha=al)
    plt.plot(exes, whys, color=col, alpha=al)
    
    if bad >= 1:
        plt.annotate(pt, (exes[1]+0.02, whys[1]), size=text_size)
    if bad == -1:
        plt.annotate(pt, (exes[0]-0.05, whys[0]), size=text_size)
    
norm_artist = plt.Circle((0,0), color='green')
bad_artist = plt.Circle((0,0), color='blue')
vbad_artist = plt.Circle((0,0), color='red')
missing_artist = plt.Circle((0,0), color='orange')
plt.legend((norm_artist, bad_artist, vbad_artist, missing_artist),
           ('Normal', 'HbS reduction <10%', 'HbS constancy or increase', 'No post-transfusion value'))
plt.xlim(-0.2,1.2)
plt.ylim(0,100)
plt.savefig(fig_name)
plt.close()
    
log.close()
    
    
    
    
    
    
    
    