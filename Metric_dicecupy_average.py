#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:36:28 2023

@author: m235103
"""

import numpy as np
import os
import math
import shutil
# Define folder path
folder_path = '/research/projects/Jaidip/Work_Data/Dr_Rule_project_KidneyPathology/Nephrectomy_donor_data/Nephrectomy_split/Train_Val_Test_Data/Data20x/Analysis/Output/Dice_Metric/Raw_noArm/'

check_images=folder_path+'checkfiles'+os.sep
if not os.path.exists(check_images):
    os.mkdir(check_images)           

# Get all files in folder
files = os.listdir(folder_path)

# Initialize variables for metrics
dice_total = 0
accuracy_total = 0
precision_total = 0
recall_total = 0
true_positive_rate_total = 0
false_negative_rate_total = 0
confusion_matrix_total = np.zeros((2, 2))

# Loop through all files in folder
for file in files:
    if 'npz' in file:
        # Load metrics from file
        metrics = np.load(folder_path + file)
        dice = metrics['dice']
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        true_positive_rate = metrics['true_positive_rate']
        false_negative_rate = metrics['false_negative_rate']
        confusion_matrix = metrics['confusion_matrix']
        
        if math.isnan(precision):
            print('precision is nana for', file)
        if (confusion_matrix[0][0])<(confusion_matrix[1][0]):
            print('check CM for', file)
        if dice<0.4:
            shutil.move(folder_path+file, check_images) #    + 'testmatrics/' 
            print()
        # Update variables for metrics
        dice_total += dice
        accuracy_total += accuracy
        precision_total += precision
        recall_total += recall
        true_positive_rate_total += true_positive_rate
        false_negative_rate_total += false_negative_rate
        confusion_matrix_total += confusion_matrix

# Calculate averages for metrics
num_files = len(files)
dice_avg = dice_total / num_files
accuracy_avg = accuracy_total / num_files
precision_avg = precision_total / num_files
recall_avg = recall_total / num_files
true_positive_rate_avg = true_positive_rate_total / num_files
false_negative_rate_avg = false_negative_rate_total / num_files
confusion_matrix_avg = confusion_matrix_total / num_files
confusion_matrix_normalized = confusion_matrix_avg.astype('float') / confusion_matrix_avg.sum(axis=1)[:, np.newaxis]

# Print averages for metrics
print('Dice average:', dice_avg)
print('Accuracy average:', accuracy_avg)
print('Precision average:', precision_avg)
print('Recall average:', recall_avg)
print('True positive rate average:', true_positive_rate_avg)
print('False negative rate average:', false_negative_rate_avg)
print('Normalized Confusion matrix average:\n', confusion_matrix_normalized)

