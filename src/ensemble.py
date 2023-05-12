# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import yaml
import numpy as np
from pandas import read_csv, DataFrame, concat
from sklearn.metrics import classification_report


class Ensemble(object):
    def __init__(self):
        pass

    def ensemble(self, config):
        """
        Ensemble predictions from NLP and Vision models
        Args:
            config (dict): configuration for the ensemble predictions
        Returns:
            df_results (pandas.DataFrame): a DataFrame containing the ensemble predictions and the labels
        """
        
        # Get corrected prediction probabilities for NLP and Vision models
        nlp_corr_pred = corrected_prediction_prob(config["nlp"], "nlp")

        vision_corr_pred = corrected_prediction_prob(config["vision"], "vision")

        # Read YAML file for Vision model
        yaml_file = read_yaml_file(config["vision"]["args"]["inference_output"])

        # Sort predictions by ID and align IDs between NLP and Vision models
        nlp_corr_pred = nlp_corr_pred.sort_values(by="id").reset_index(drop=True)
        vision_corr_pred = vision_corr_pred.sort_values(by="id").reset_index(drop=True)

        if (len(nlp_corr_pred) != len(vision_corr_pred)) or ( all(nlp_corr_pred.id != vision_corr_pred.id)):
            nlp_corr_pred, vision_corr_pred = set_diff_ids(nlp_corr_pred, vision_corr_pred, config['test_size'])

        # Create ensemble scores and map predicted labels to their original label names
        pred, pred_score = create_ensemble_scores(
            nlp_corr_pred["predictions_probabilities"].to_list(),
            vision_corr_pred["predictions_probabilities"].to_list(),
        )
        pred_mapped = [yaml_file["label"][i] for i in pred]

        # Create DataFrame with predictions and labels
        df_results = DataFrame()
        df_results["id"] = nlp_corr_pred.id
        df_results["labels"] = vision_corr_pred["labels"]
        df_results["vision_predictions"] = vision_corr_pred["predictions_label"]
        df_results["nlp_predictions"] = nlp_corr_pred["predictions_label"]
        df_results["ensemble_predictions"] = pred_mapped
        df_results["prediction_scores"] = pred_score

        return df_results


def read_yaml_file(yaml_path):
    with open(yaml_path, "r") as file:
        yaml_dict = yaml.safe_load(file)

    return yaml_dict


def f1_score_per_class(yaml_dict):
    """
    Compute the F1-score per class based on the predictions in a YAML dictionary.

    Args:
    - yaml_dict (dict): A dictionary containing the predictions for a given task.

    Returns:
    - f1_scores (pandas.Series): A Series containing the F1-score per class.
    """
    # Extract ground truth labels and predicted labels
    truth = yaml_dict["label_id"]
    pred = yaml_dict["predictions_label"]
    
    # Compute the classification report and convert to DataFrame
    cm_res = classification_report(truth, pred, output_dict=True)
    df_cm_res = DataFrame(cm_res).transpose().round(3)
    
    # Extract the F1-score per class and return as a Series
    f1_scores = df_cm_res.loc[[str(i) for i in np.unique(truth)], "f1-score"]
    return f1_scores


def corrected_prediction_prob(config, domain=None):
    """
    Reads predictions_probabilities for finetune and inference, and corrects the inference scores based on
    the f1-scores of the training data. The corrected scores are then returned as a pandas DataFrame.

    Args:
        config (dict): A dictionary containing the configuration parameters.
        domain (str): A string specifying the domain for which the predictions are being made.
                      Supported values are 'vision' and 'nlp'. Defaults to None.

    Returns:
        pandas DataFrame: A DataFrame containing corrected predictions probabilities, labels and IDs.
    """
    # read predictions_probabilities for finetune
    yaml_dict_finetune = read_yaml_file(config["args"]["finetune_output"])

    # read predictions_probabilities for inference
    yaml_dict_inference = read_yaml_file(config["args"]["inference_output"])

    if domain == "vision":
        # preprocess vision predictions
        yaml_dict_finetune = vision_prediction_postprocess(yaml_dict_finetune)
        yaml_dict_inference = vision_prediction_postprocess(yaml_dict_inference)

    elif domain == "nlp":
        # preprocess NLP predictions
        yaml_dict_finetune = nlp_prediction_postprocess(
            yaml_dict_finetune, config, "finetune"
        )
        yaml_dict_inference = nlp_prediction_postprocess(
            yaml_dict_inference, config, "inference"
        )

    else:
        # Raise an error if the domain is not supported
        raise ValueError(
            "Unknown domain request. Supported domains are 'vision' an 'nlp'."
        )

    # get f1_scores from training data
    f1_scores = f1_score_per_class(yaml_dict_finetune)

    # corrected scores based on f1-scores
    corrected_scores = [
        list(f1_scores * i) for i in yaml_dict_inference["predictions_probabilities"]
    ]

    # create a pandas DataFrame with corrected predictions, labels and IDs
    df_corrected = DataFrame()
    df_corrected["labels"] = yaml_dict_inference["label_id"]
    df_corrected["predictions_label"] = yaml_dict_inference["predictions_label"]
    df_corrected["predictions_probabilities"] = corrected_scores
    df_corrected["id"] = yaml_dict_inference["id"]

    return df_corrected


def nlp_prediction_postprocess(yaml_dict, config, pipeline):
    """
    Postprocesses predictions obtained from an NLP pipeline.

    Args:
        yaml_dict: A dictionary containing the predictions.
        config: A dictionary containing configuration parameters.
        pipeline: A string indicating whether the predictions are from the finetune or inference pipeline.

    Returns:
        A dictionary containing the postprocessed predictions.
    """

    # Get the path to the CSV file containing patient IDs.
    path_csv = (
        config["args"]["local_dataset"]["finetune_input"]
        if pipeline == "finetune"
        else config["args"]["local_dataset"]["inference_input"]
    )

    # Add patient IDs to the YAML dictionary.
    yaml_dict["id"] = read_csv(path_csv)["Patient_ID"].to_list()

    return yaml_dict


def vision_prediction_postprocess(yaml_dict):
    """
    Postprocesses predictions obtained from a computer vision pipeline.

    Args:
        yaml_dict: A dictionary containing the predictions.

    Returns:
        A dictionary containing the postprocessed predictions.
    """

    # Get the predictions for each file.
    results_dict = yaml_dict["results"]

    # Create a DataFrame for each file.
    df_list = []
    for f in results_dict.keys():
        temp_dict = results_dict[f]
        temp_dict[0]["file"] = f
        temp_dict[0]["id"] = get_subject_id(f)
        df_list.append(DataFrame(temp_dict))

    # Concatenate the DataFrames and group by patient ID.
    df = concat(df_list)
    df_g = df.groupby("id")

    # For each patient ID, calculate the mean prediction per class.
    # Then, assign the class with the highest mean prediction as the predicted class.
    df_list = []
    for k in df_g.groups.keys():
        df_temp = df_g.get_group(k).reset_index(drop=True)
        mean_per_class = np.mean(np.array(df_temp.pred_prob.to_list()), axis=0)
        pred = yaml_dict["label"][np.argmax(mean_per_class)]

        df_temp.loc[0, "pred_prob"] = mean_per_class
        df_temp.loc[0, "pred"] = pred

        df_list.append(df_temp.loc[[0]])

    df_results = concat(df_list)

    # Create a dictionary with the postprocessed predictions.
    predictions_report = {}
    predictions_report["label_id"] = df_results.label.to_list()
    predictions_report["predictions_label"] = df_results.pred.to_list()
    predictions_report["predictions_probabilities"] = df_results.pred_prob.to_list()
    predictions_report["id"] = df_results.id.to_list()

    return predictions_report


def create_ensemble_scores(nlp_corr_pred, vision_corr_pred):
    """
    Create ensemble scores by adding the corrected prediction scores for both NLP and vision models element-wise.

    Args:
    nlp_corr_pred: List containing corrected prediction scores for NLP model.
    vision_corr_pred: List containing corrected prediction scores for vision model.

    Returns:
    List containing predicted labels based on ensemble scores.
    """
    corr_pred_scores = np.array(nlp_corr_pred) + np.array(vision_corr_pred)
    return [np.argmax(i) for i in corr_pred_scores], [ max(i/sum(i)) for i in corr_pred_scores]


def set_diff_ids(df1, df2, test_size ):
    """
    Concatenates dataframes and ensures both dataframes have the same unique IDs.

    Args:
    df1: First dataframe.
    df2: Second dataframe.

    Returns:
    Two dataframes with the same unique IDs.
    """

    if test_size:
        # find the IDs in df1 that are not in df2
        diff_ids_1 = list(set(df1.id) - set(df2.id))

        # concatenate df1 and the subset of df1 that contains the IDs not in df2
        if diff_ids_1:
            df2 = (
                concat([df2, df1[df1.id.isin(diff_ids_1)]])
                .sort_values(by="id")
                .reset_index(drop=True)
            )

        # find the IDs in df2 that are not in df1
        diff_ids_2 = list(set(df2.id) - set(df1.id))

        # concatenate df2 and the subset of df2 that contains the IDs not in df1
        if diff_ids_2:
            df1 = (
                concat([df1, df2[df2.id.isin(diff_ids_2)]])
                .sort_values(by="id")
                .reset_index(drop=True)
            )

    else:
        intersection_items =  list(set(df1.id).intersection(set(df2.id)))
        df1 = df1[df1.id.isin(intersection_items)].reset_index(drop=True)
        df2 = df2[df2.id.isin(intersection_items)].reset_index(drop=True)

    return df1, df2

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