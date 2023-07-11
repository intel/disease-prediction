# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from os import path 

def data_preparation(config, root_folder):
    """
    Prepares the data for training and testing NLP and Vision models, as well as creating and saving training and testing data.
    
    Args:
    - config: a dictionary containing configuration parameters
    - root_folder: the root folder where data is stored
    
    Returns:
    - config: a dictionary containing updated configuration parameters
    """
    
    # get the path to training and testing data
    data_root = config["nlp"]["args"]["local_dataset"]["finetune_input"]
    training_data_path = path.join(root_folder, data_root)
    
    data_root = config["nlp"]["args"]["local_dataset"]["inference_input"]
    testing_data_path = path.join(root_folder, data_root)
    
    # if training or testing data does not exist, create them unless overwrite_training_testing_ids is False
    if not (path.exists(training_data_path) and path.exists(testing_data_path)):
        raise Exception("Training or Testing data does not exist") 
    
    # update the paths to the training and testing data
    config["nlp"]["args"]["local_dataset"]["finetune_input"] = training_data_path
    config["nlp"]["args"]["local_dataset"]["inference_input"] = testing_data_path

    # sort the labels in the configuration
    config["nlp"]["args"]["local_dataset"]["label_list"] = np.sort(
        config["nlp"]["args"]["local_dataset"]["label_list"]
    ).tolist()

    return config



