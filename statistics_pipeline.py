# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:13:15 2020

@author: Johannes-Robert Bruch

This script helps you to analyse the data that the kfold_valdation file produces.
It needs the yaml, statistics, numpy, scipy, math and matplotlib packages.
"""
import compile_results
import plot_mean_accuracies
import corrected_t_test

def main():
    print('To skip a step, press enter without typing "yes".')
    answer=input('Enter yes to compile and plot the results from the k-fold validation.')
    if 'yes'==answer:
        min_num_frozen=input('Enter the lowest number of frozen layers that you tested: ')
        max_num_frozen=input('Enter the highest number of frozen layers that you tested: ')
        compile_results.main(min_num_frozen,max_num_frozen)
        plot_mean_accuracies.main(min_num_frozen,max_num_frozen)
    else:
        print('We will skip this and continue with the next step:')
        
    answer=input('Enter yes to check whether the difference between two methods is significant.')
    if 'yes'==answer:
        validation_version_1=input('Enter the number of frozen layers in the first method that you want to compare: ')
        validation_version_2=input('Enter the number of frozen layers in the second method that you want to compare: ')
        corrected_t_test.main(validation_version_1,validation_version_2)
    else:
        print('We will skip this and continue with the next step:')
        
        

        
