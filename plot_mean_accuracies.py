# -*- coding: utf-8 -*-
"""
Created on March 06 2019
plotting mean accuracies over number of frozen layers n
@author: jrbru
"""
from __future__ import absolute_import, division, print_function

import yaml
import numpy as np
import matplotlib.pyplot as plt
import statistics

# Parameters:
    # minimum number of frozen convolutional layers 
    # maximum number of frozen convolutional layers
def main(min_num_frozen,max_num_frozen): 
    # range of numbers of frozen convolutional layers 
    range_frozen = range(min_num_frozen,max_num_frozen+1,1)
    # number of epochs that the CNN was trained with 9folds of the PI dataset
    num_epochs = 10
    # transforming to zero based index
    j = num_epochs - 1
    # two lists that will be plotted over range_frozen
    trainings_means = []
    evaluations_means = []
    
    for num_frozen in range_frozen:
        try:
            # load two lists that contain validation accuracies to be compared
            key = 'trainings_V' + str(num_frozen)
            with open("Results/" + key + ".yaml") as stream:
                a = yaml.load(stream)
                A = np.array(a, dtype=float)
            key = 'evaluations_V' + str(num_frozen)
            with open("Results/" + key + ".yaml") as stream:
                b = yaml.load(stream)
                B = np.array(b, dtype=float)     
            # extract mean of specified epoch
            trainings_means.append(statistics.mean(A[:,j]))
            evaluations_means.append(statistics.mean(B[:,j]))
        except FileNotFoundError:
            print('File not found: trainings_V' + str(num_frozen))
    
    # FOR A MATRIX A WITH k ROWS AND l COLUMNS
    # corresponding to k folds and l epochs
    fig1, ax1 = plt.subplots()
    # ax1.set_title('Accuracies')
    ax1.set_xlabel('Number of frozen convolutional layers')
    ax1.set_ylabel('Mean top-1 accuracy')
    ax1.set_ylim(0,1.01)
    ax1.set_xlim(0,len(evaluations_means))
    ax1.plot(trainings_means, 'b+')
    ax1.plot(evaluations_means, 'ro')
    # ax1.legend(['Training','Testing'])
    # print(fig1.canvas.get_supported_filetypes())
    fig1.savefig('Plots/mean_accuracies_' + str(len(evaluations_means)) + '_layers_' + str(num_epochs) + '_epochs.jpg', transparent=False, dpi=500, bbox_inches="tight")