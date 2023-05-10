# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from pandas import DataFrame 
from os import path
from pathlib import Path

def get_subject_id(image_name):
    """
    Extracts the patient ID from an image filename.

    Args:
    - image_name: string representing the filename of an image

    Returns:
    - patient_id: string representing the patient ID extracted from the image filename
    """

    # Split the filename by "/"
    image_name = image_name.split("/")[-1]

    # Extract the first two substrings separated by "_", remove the first character (which is "P"), and join them
    # together to form the patient ID
    patient_id = "".join(image_name.split("_")[:2])[1:]

    return patient_id


def create_confusion_matrix(df, true_labels, predictions):
    """
    Computes and returns a confusion matrix for the given DataFrame, true label column name, and predicted label column name.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the true and predicted labels.
    true_labels (str): Name of the column containing the true labels.
    predictions (str): Name of the column containing the predicted labels.

    Returns:
    pandas.DataFrame: Confusion matrix as a DataFrame.
    """

    # Get unique labels from true label column
    labels = df[true_labels].unique().tolist()

    # Compute classification report and round results to 3 decimal places
    cm_res = classification_report(
        df[true_labels].to_list(), df[predictions].to_list(), output_dict=True
    )
    df_cm_res = DataFrame(cm_res).transpose().round(3)

    # Compute confusion matrix and create DataFrame with labeled rows and columns
    cm = confusion_matrix(df[true_labels], df[predictions], labels=labels)
    df_cm = DataFrame(cm, columns=labels, index=labels)

    # Add precision and recall rows to the DataFrame
    df_cm["Recall"] = None
    labels = [str(i) for i in labels]
    df_cm.loc["Precision"] = df_cm_res.loc[labels + ["accuracy"], "precision"].to_list()
    df_cm["Recall"] = df_cm_res.loc[labels + ["accuracy"], "recall"].to_list()

    return df_cm


def report_the_results(df_results, true_labels, prediction_list):
    """
    Generate a confusion matrix for each prediction type and print them out.

    Args:
    - df_results: pandas DataFrame, the data on which to generate the confusion matrices
    - true_labels: str, the column name for the true labels
    - prediction_list: list of str, the column names for the different types of predictions
    """
    for p in prediction_list:
        # Print the name of the prediction type for which the confusion matrix is generated
        print("        Confusion Matrix for " + p)
        
        # Generate the confusion matrix for this prediction type
        cm = create_confusion_matrix(df_results, true_labels=true_labels, predictions=p)
        
        # Print the confusion matrix
        print(cm)
        print("")


# update config file
def update_config_file(config, root_folder):
    # Update the paths in the config dictionary to use absolute paths
    config["nlp"]["training_args"]["output_dir"] = path.join(
        root_folder, config["output_dir"], "nlp"
    )
    config["vision"]["training_args"]["output_dir"] = path.join(
        root_folder, config["output_dir"], "vision"
    )

    config["vision"]["args"]["dataset_dir"] = path.join(
        root_folder, config["vision"]["args"]["dataset_dir"]
    )
    config["vision"]["args"]["saved_model_dir"] = path.join(
        root_folder,
        config["vision"]["training_args"]["output_dir"],
        config["vision"]["args"]["saved_model_dir"],
    )

    # Create the directories if they don't exist
    Path(config["nlp"]["training_args"]["output_dir"]).mkdir(
        parents=True, exist_ok=True
    )
    Path(config["vision"]["training_args"]["output_dir"]).mkdir(
        parents=True, exist_ok=True
    )

    config["nlp"]["write"] = config["write"]
    config["vision"]["write"] = config["write"]
    
    return config
