# -*- coding: utf-8 -*-
"""
Created on March 06 2019
computing statistics of classification results
@author: jrbru
"""
from __future__ import absolute_import, division, print_function

import yaml
import statistics
import numpy as np
from scipy.stats import t
import math

# Parameters
    # validation_version_2 is the version specified in the filenames of the prediction results of the first method
    # validation_version_1 is the version specified in the filenames of the prediction results of the second method
def main(validation_version_1,validation_version_2):
    # True if the accuracies describe the prediction of data used in training of the same classifier
    is_training_data=True
    if is_training_data is True:
        filename = 'trainings'
    else:
        filename = 'evaluations'
    
    # load two lists that contain validation accuracies to be compared
    key = filename + '_V' + str(validation_version_1)
    with open("Results/" + key + ".yaml") as stream:
        a = yaml.load(stream)
        A = np.array(a)
        print('Validation accuracies of version' + str(validation_version_1))
        print(A)
    key = filename + '_V' + str(validation_version_2)
    with open("Results/" + key + ".yaml") as stream:
        b = yaml.load(stream)
        B = np.array(b)
        print('Validation accuracies of version' + str(validation_version_2))
        print(B)
    
    # following bouckaert and frank 2004 Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms 3.2
    # ,but without repetitions and with epochs, we test whether the difference between accuracies of two algorithms is significant.
    # compute elementwise differences x between two matrices
    X = A - B
    X = X.astype(float)
    # FOR A MATRIX X WITH k ROWS AND l COLUMNS
    # corresponding to k folds and l epochs
    k = X.shape[0]
    l = X.shape[1]
    # define degrees of freedom
    df = (k) - 1
    # compute value based on the 0.975 percentil for a two-sided test with 5% cl
    threshold = t.ppf([0.975], 9)
    # two lists are created with means and variances for differences of each epoch
    means = []
    variances = []
    corrected_ts = []
    is_significant = []
    for j in range(l):
        # compute means p_i,j_bar
        
        mean = statistics.mean(X[:,j])
        means.append(mean)
        # compute variance s_pij
        variance = statistics.variance(X[:,j])
        variances.append(variance)
        # compute t-statistic for k-fold cv
        corrected_t = mean/math.sqrt(((1/k)+(1/(k-1)))*variance)
        corrected_ts.append(corrected_t)
        # The null hypothesis is that the two learning algorithms will have the same accuracy
        # We can reject the null hypothesis if |corrected_t| > threshold
        if abs(corrected_t) > threshold:
            is_significant.append(True)
        else:
            is_significant.append(False)
           
    for i in range(len(means)):
        if is_significant[i]==True:
            booleanstring=''
        elif is_significant[i]==False:
            booleanstring='not'
        else:
            booleanstring='ERROR'
        print('After '+i+' epochs, the method with '+str(validation_version_1)+' frozen layers leads to a mean accuracy that is '+str(means[i])+' higher than the one of method '+str(validation_version_2)+'. This is '+booleanstring+'statistically significant at the 5% confidence level.')
    
    statistics_dict = {'mean difference': means,
                             'variance of difference': variances,
                             'corrected t statistic': corrected_ts,
                             'Different at 5% CL': is_significant,}
    statistics_key= filename + '_statistics_' + str(validation_version_1) + 'vs' + str(validation_version_2)
    dictionary = {statistics_key: statistics_dict}
    # saving all lists by serialising to yaml
    for key, value in dictionary.items():
        stream = open('Results/statistics/' + key + '.yaml', 'w')
        yaml.dump(value, stream)
