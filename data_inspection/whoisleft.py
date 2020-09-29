#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Looks through a move_and_prepare folder and checks if the subjects have a Freesurfer folder already. If they don't, writes the subject ID to
a number of text files split between specified prefixes
"""

import os
from glob import glob

in_folder = '/Users/manusdonahue/Documents/Sky/volume_testing/'
subjects_folder = os.environ['SUBJECTS_DIR']
prefixes = ['/Users/skyjones/Documents/volume_testing', '/Users/manusdonahue/Documents/Sky/volume_testing']
split_point = 0.4

out_txts = ['/Users/manusdonahue/Desktop/subject_list_mine.txt', '/Users/manusdonahue/Desktop/subject_list_perom.txt']

def get_terminal(path):
    """
    Takes a filepath or directory tree and returns the last file or directory
    

    Parameters
    ----------
    path : path
        path in question.

    Returns
    -------
    str of only the final file or directory.

    """
    return os.path.basename(os.path.normpath(path))


pt_folders = glob(os.path.join(in_folder, '*/'))
freesurfer_folders = glob(os.path.join(subjects_folder, '*/'))

pt_ids = [get_terminal(i) for i in pt_folders]
freesurfer_ids = [get_terminal(i) for i in freesurfer_folders]

still_left = [i for i in pt_ids if i not in freesurfer_ids]

split_ind = int(len(still_left) * split_point)

lefts = []
lefts.append(still_left[:split_ind])
lefts.append(still_left[split_ind:])


for i in range(2):
    
    paths = [os.path.join(prefixes[i], j) for j in lefts[i]]
    with open(out_txts[i], 'w') as f:
        for item in paths:
            f.write("%s\n" % item)