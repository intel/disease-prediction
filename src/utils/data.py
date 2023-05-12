# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import shutil
import numpy as np
from os import path, listdir
from pandas import DataFrame, read_csv
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from pathlib import Path
from .utils import get_subject_id

def create_train_and_test_nlp(df: DataFrame, test_size: float) -> Tuple[DataFrame, DataFrame]:
    """Split the dataset into training and testing sets for NLP.

    Args:
        df: Pandas DataFrame containing the data.
        test_size: Proportion of the dataset to include in the test split.

    Returns:
        Tuple of Pandas DataFrames for training and testing, respectively.
    """
    return train_test_split(df, test_size=test_size)


def copy_images(patient_ids: DataFrame, source_folder: str, target_folder: str) -> None:
    """Copy images of selected patients from the source folder to the target folder.

    Args:
        patient_ids: List of patient IDs for whom the images need to be copied.
        source_folder: Path to the source folder containing the images.
        target_folder: Path to the target folder where the images need to be copied.
    """
    
    for f in listdir(source_folder):
        if ("_CM_" in f) and (get_subject_id(f) in patient_ids.Patient_ID.to_list()):
            full_src_path = path.join(source_folder, f)
            shutil.copy(full_src_path, target_folder)

def create_train_and_test_vision_data(root_folder: str, config: dict, train_data: DataFrame, test_data: DataFrame) -> dict:
    """
    This function creates and organizes the train and test dataset for vision task.

    Parameters:
    root_folder (str): The root directory path.
    config (dict): A dictionary containing the configuration information.
    train_data (pd.DataFrame): A pandas DataFrame containing the training data.
    test_data (pd.DataFrame): A pandas DataFrame containing the testing data.

    Returns:
    dict: A dictionary containing the configuration information.
    """

    # Get the list of class labels and the column containing the class label
    label_list = config["nlp"]["args"]["local_dataset"]["label_list"]
    label_column = config["nlp"]["args"]["local_dataset"]["features"]["class_label"]

    # Create the target directory for the dataset
    target_folder = config["vision"]["args"]["dataset_dir"]
    if path.exists(target_folder):
        shutil.rmtree(target_folder)

    # Iterate over train and test data
    for cat in ["test", "train"]:
        # Get the data based on the current category
        df = test_data if cat == "test" else train_data
        
        # Iterate over each label
        for label in label_list:
            # Source folder for images
            source_folder = path.join(root_folder, config["vision"]["args"]["segmented_dir"], label)
            
            # Create target folder
            target_folder = path.join(config["vision"]["args"]["dataset_dir"], cat, label)
            Path(target_folder).mkdir(parents=True, exist_ok=True)

            # Copy images
            patient_ids = df[df[label_column] == label]
            copy_images(patient_ids, source_folder, target_folder)

    return config


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
    
    if config['test_size']:
        training_data_path = path.join(root_folder, "/".join(data_root.split("/")[:-1]), "training_data.csv" )
    else:
        training_data_path = path.join(root_folder, data_root) # "/".join(data_root.split("/")[:-1]), "training_data.csv" )


    data_root = config["nlp"]["args"]["local_dataset"]["inference_input"]
    if config['test_size']:
        testing_data_path = path.join(root_folder, "/".join(data_root.split("/")[:-1]), "testing_data.csv")
    else:
        testing_data_path = path.join(root_folder, data_root)
    
    # if training or testing data does not exist, create them unless overwrite_training_testing_ids is False
    if not (path.exists(training_data_path) and path.exists(testing_data_path)):
        if config["overwrite_training_testing_ids"]:
            print("Training or Testing data does not exist. Creating them.")
        else:
            raise Exception("Training or Testing data does not exist and overwrite_training_testing_ids is False.")

    # create training and testing data for NLP if overwrite_training_testing_ids is True
    if config["overwrite_training_testing_ids"] == True:
        # read the input data
        input_data = read_csv(path.join(root_folder, config["nlp"]["args"]["local_dataset"]["finetune_input"]))

        # create training and testing data
        training_data, testing_data = create_train_and_test_nlp(input_data, config["test_size"])

        # save training and testing data
        training_data.to_csv(training_data_path, index=False)
        testing_data.to_csv(testing_data_path, index=False)

        # create vision data
        create_train_and_test_vision_data(root_folder, config, training_data, testing_data)
    
    # update the paths to the training and testing data
    config["nlp"]["args"]["local_dataset"]["finetune_input"] = training_data_path
    config["nlp"]["args"]["local_dataset"]["inference_input"] = testing_data_path

    # sort the labels in the configuration
    config["nlp"]["args"]["local_dataset"]["label_list"] = np.sort(
        config["nlp"]["args"]["local_dataset"]["label_list"]
    ).tolist()

    return config


