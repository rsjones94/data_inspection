#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Looks through the system's SUBJECTS_DIR folder (where Freesurfer processing is done) and deletes those that are incomplete (those lacking an aseg file).
Useful to run after killing a Freesurfer processing script so you don't have to hunt down the folder that was being processed and remove it manually
"""

import os
from glob import glob
import shutil

subjects_folder = os.environ['SUBJECTS_DIR']
rel_file_of_interest = os.path.join('stats', 'aseg.stats')


freesurfer_folders = glob(os.path.join(subjects_folder, '*/'))

for sub in freesurfer_folders:
    completion_file = os.path.join(sub, rel_file_of_interest) # if this file doesn't exist, assume the subject folder is incomplete
    
    if os.path.exists(completion_file):
        continue
    
    print(f'{sub} is incomplete')
    shutil.rmtree(sub)