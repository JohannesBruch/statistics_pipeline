"""author J-R Bruch
compiling classification results
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml

# Parameters:
    # minimum number of frozen convolutional layers 
    # maximum number of frozen convolutional layers
def main(min_num_frozen,max_num_frozen):
    # number of folds
    k = 10
    # validation version is the version specified in the filenames of the prediction results
    for validation_version in range(min_num_frozen,max_num_frozen+1):
        
        # a list of dictionaries with evaluation results will be created
        evaluations = []
        # a list of main prediction results will be created
        trainings = []
        
        for i_test_fold in range(k):
            trainings_key = 'trainings_V' + str(validation_version) + 'F' + str(i_test_fold)
            with open("Results/fold files/" + trainings_key + ".yaml") as stream:
                # the .append is used to add an item to the end of an iterable
                trainings.append(yaml.load(stream))
            evaluations_key = 'evaluations_V' + str(validation_version) + 'F' + str(i_test_fold)
            with open("Results/fold files/" + evaluations_key + ".yaml") as stream:
                # the .append is used to add an item to the end of an iterable
                evaluations.append(yaml.load(stream))
        
        evaluations_key = 'evaluations_V' + str(validation_version)
        trainings_key = 'trainings_V' + str(validation_version) 
        dictionary = {trainings_key: trainings,
                      evaluations_key: evaluations}
        for key, value in dictionary.items():
            stream = open('Results/' + key + '.yaml', 'w')
            yaml.dump(value, stream)
