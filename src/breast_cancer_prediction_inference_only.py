# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys

from os import path, walk
from pandas import read_csv
from transformers import TrainingArguments

# Set the root folder to the current file's directory
root_folder = path.dirname(path.abspath(__file__))

# Insert necessary paths for module imports
sys.path.insert(
    0, path.join(root_folder, "../hf_nlp/workflows/hf_finetuning_and_inference_nlp/src")
)
sys.path.insert(0, path.join(root_folder, "../vision_wf"))

from infer_itrex import ItrexInfer
from workflows.disease_prediction.src.vision_wl import (
    load_model,
    run_inference_per_patient,
)
from src.ensemble import (
    get_subject_id,
    vision_prediction_postprocess,
    read_yaml_file,
    f1_score_per_class,
    nlp_prediction_postprocess,
    create_ensemble_scores,
)


class Inference(object):
    """
    Class representing the inference process.

    Attributes:
        nlp_inference: NLP inference object.
        nlp_yaml_dict_inference_config: NLP inference configuration dictionary.
        vision_yaml_dict_inference_config: Vision inference configuration dictionary.
        vision_f1_scores: F1 scores for the vision domain.
        nlp_f1_scores: F1 scores for the NLP domain.
        vision_model: Vision model for inference.
        class_names: List of class names for NLP inference.
    """

    def __init__(self):
        """
        Initializes the Inference object by loading the necessary inference configurations and models.
        """
        (
            self.nlp_inference,
            self.nlp_yaml_dict_inference_config,
            self.vision_yaml_dict_inference_config,
            self.vision_f1_scores,
            self.nlp_f1_scores,
            self.vision_model,
            self.class_names,
        ) = load_finetune_configs()

    def multimodel_inference(self, pid_list):
        """
        Performs multimodel inference using the provided list of patient IDs.

        Args:
            pid_list (list): List of patient IDs for inference.

        Returns:
            dict: A dictionary mapping patient IDs to their corresponding predicted labels.
        """
        
        import time
        start_time = time.time()
        
        # Perform vision inference
        vision_pred = create_vis_pred(
            pid_list,
            self.vision_model,
            self.class_names,
            self.vision_yaml_dict_inference_config,
            self.vision_f1_scores,
        )
        
        print(">>> --- Load vision_pred Inference: %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        # Perform NLP inference
        nlp_pred = create_nlp_pred(
            pid_list,
            self.nlp_yaml_dict_inference_config,
            self.nlp_inference,
            self.nlp_f1_scores,
        )
        
        print(">>> --- Load NLP pred Inference: %s seconds ---" % (time.time() - start_time))

        # Create ensemble scores and map predicted labels to their original label names
        pred, pred_score = create_ensemble_scores(vision_pred, nlp_pred)

        # Mapping patient IDs to predicted labels
        pred_dict = {}
        for i, v in enumerate(pid_list):
            pred_dict[v] = self.class_names[pred[i]]

        return pred_dict


def create_nlp_pred(pid_list, nlp_yaml_dict_inference_config, nlp_inference, f1_scores):
    """
    Create NLP predictions for the given list of patient IDs.

    Args:
        pid_list (list): List of patient IDs for prediction.
        nlp_yaml_dict_inference_config (dict): NLP inference configuration dictionary.
        nlp_inference (object): NLP inference object.
        f1_scores (list): List of F1 scores.

    Returns:
        list: List of corrected prediction scores.
    """

    # Create csv file for inference
    inf_list = create_nlp_inference_data(pid_list, nlp_yaml_dict_inference_config)

    # Predict and save results
    for f in inf_list:
        nlp_inference.e2e_infer_only(f)

    nlp_yaml_dict_inference_config = read_yaml_file(
        path.join(root_folder, "../configs/nlp_inference.yaml")
    )
    nlp_dict_inference = read_yaml_file(
        nlp_yaml_dict_inference_config["args"]["inference_output"]
    )
    nlp_dict_inference["id"] = pid_list

    # Corrected scores based on f1-scores
    corrected_scores = [
        list(f1_scores * i) for i in nlp_dict_inference["predictions_probabilities"]
    ]

    return corrected_scores


def nlp_inference_setup(nlp_infr_config):
    """
    Sets up the NLP inference by initializing the necessary configurations and objects.

    Args:
        nlp_infr_config (dict): NLP inference configuration dictionary.

    Returns:
        ItrexInfer: NLP inference object.

    """

    training_args = TrainingArguments(output_dir=path.join(root_folder, "output_dir"))
    for item in nlp_infr_config["training_args"]:
        setattr(training_args, item, nlp_infr_config["training_args"][item])

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    for item in nlp_infr_config["args"]:
        setattr(args, item, nlp_infr_config["args"][item])

    kwargs = {"args": args, "training_args": training_args}
    nlp_inference = ItrexInfer(**kwargs)
    nlp_inference.e2e_infer_setup_only()

    return nlp_inference


def create_nlp_inference_data(pid_list, nlp_yaml_dict_inference_config):
    """
    Create NLP inference data for the given list of patient IDs.

    Args:
        pid_list (list): List of patient IDs for inference.
        nlp_yaml_dict_inference_config (dict): NLP inference configuration dictionary.

    Returns:
        list: List of file paths for NLP inference data.
    """

    test_file_loc = nlp_yaml_dict_inference_config["args"]["local_dataset"][
        "inference_input"
    ]
    df_test = read_csv(test_file_loc)
    id_columns = nlp_yaml_dict_inference_config["args"]["local_dataset"]["features"][
        "id"
    ]

    df_inference = df_test[df_test[id_columns].isin(pid_list)]

    inference_path = path.join(
        "/".join(test_file_loc.split("/")[:-1]), "_inference.csv"
    )

    df_inference.to_csv(inference_path, index=False)
    return [inference_path]


def read_config_yaml_files(finetune_yaml, inference_yaml, domain):
    """
    Reads and processes the configuration YAML files for finetuning and inference.

    Args:
        finetune_yaml (str): File name of the finetuning YAML configuration.
        inference_yaml (str): File name of the inference YAML configuration.
        domain (str): Domain name ("vision" or "nlp").

    Returns:
        tuple: A tuple containing the finetuning configuration dictionary, inference configuration dictionary,
               and the processed finetuning data dictionary.
    Raises:
        ValueError: If the domain is not supported.
    """

    yaml_dict_finetune_config = read_yaml_file(
        path.join(root_folder, "../configs", finetune_yaml)
    )
    # read predictions_probabilities for finetune
    yaml_dict_finetune = read_yaml_file(
        yaml_dict_finetune_config["args"]["finetune_output"]
    )

    if domain == "vision":
        yaml_dict_finetune = vision_prediction_postprocess(yaml_dict_finetune)
    elif domain == "nlp":
        yaml_dict_finetune = nlp_prediction_postprocess(
            yaml_dict_finetune, yaml_dict_finetune_config, "finetune"
        )

    else:
        # Raise an error if the domain is not supported
        raise ValueError(
            "Unknown domain request. Supported domains are 'vision' an 'nlp'."
        )

    yaml_dict_inference_path = path.join(root_folder, "../configs", inference_yaml)
    yaml_dict_inference_config = read_yaml_file(yaml_dict_inference_path)

    return yaml_dict_finetune_config, yaml_dict_inference_config, yaml_dict_finetune


def load_finetune_configs():
    """
    Loads the finetune configurations and models for vision and NLP domains.

    Returns:
        tuple: A tuple containing the following elements:
            - nlp_inference: NLP inference object.
            - nlp_yaml_dict_inference_config: NLP inference configuration dictionary.
            - vision_yaml_dict_inference_config: Vision inference configuration dictionary.
            - vision_f1_scores: F1 scores for the vision domain.
            - nlp_f1_scores: F1 scores for the NLP domain.
            - vision_model: Vision model for inference.
            - class_names: List of class names for NLP inference.
    """

    # Load vision finetune configurations and inference configurations
    (
        vision_yaml_dict_finetune_config,
        vision_yaml_dict_inference_config,
        vision_yaml_dict_finetune,
    ) = read_config_yaml_files(
        "vision_finetune.yaml", "vision_inference.yaml", "vision"
    )

    # Load vision model for inference
    vision_model = vision_inference(vision_yaml_dict_finetune_config)

    # Get F1 scores for the vision domain from training
    vision_f1_scores = f1_score_per_class(vision_yaml_dict_finetune)

    # Load NLP finetune configurations and inference configurations
    (
        nlp_yaml_dict_finetune_config,
        nlp_yaml_dict_inference_config,
        nlp_yaml_dict_finetune,
    ) = read_config_yaml_files("nlp_finetune.yaml", "nlp_inference.yaml", "nlp")

    # Get F1 scores for the NLP domain from training data
    nlp_f1_scores = f1_score_per_class(nlp_yaml_dict_finetune)

    # Get the list of class names for NLP inference
    class_names = nlp_yaml_dict_inference_config["args"]["local_dataset"]["label_list"]

    # Setup NLP inference object
    nlp_inference = nlp_inference_setup(nlp_yaml_dict_inference_config)

    # Return the tuple of inference-related elements
    return (
        nlp_inference,
        nlp_yaml_dict_inference_config,
        vision_yaml_dict_inference_config,
        vision_f1_scores,
        nlp_f1_scores,
        vision_model,
        class_names,
    )


def vision_inference(vision_yaml_dict_finetune_config):
    """
    Loads and returns the vision model for inference based on the provided finetune configuration.

    Args:
        vision_yaml_dict_finetune_config (dict): Vision finetune configuration dictionary.

    Returns:
        object: Vision model for inference.
    """
    vision_model = load_model(
        model_name=vision_yaml_dict_finetune_config["args"]["model"],
        saved_model_dir=vision_yaml_dict_finetune_config["args"]["saved_model_dir"],
    )
    return vision_model


def find_image_paths(test_paths, pid):
    """
    Finds and returns the file paths of images associated with a specific patient ID.

    Args:
        test_paths (str): Directory containing the image files.
        pid (str): Patient ID.

    Returns:
        list: List of file paths of images associated with the patient ID.
    """

    file_path_list = []
    for file_path, subdirs, files in walk(test_paths):
        for name in files:
            if pid == get_subject_id(name):
                file_path_list.append(path.join(file_path, name))

    return file_path_list


def vis_pred_post_process(vis_pred_items, class_names):
    """
    Performs post-processing on the vision predictions.

    Args:
        vis_pred_items (dict): Dictionary containing vision predictions.
        class_names (list): List of class names.

    Returns:
        dict: Processed vision predictions.
    """

    temp_dict = {}
    temp_dict["label"] = class_names
    temp_dict["results"] = {}

    for p_id in vis_pred_items.keys():
        for item in vis_pred_items[p_id].keys():
            temp_dict["results"][item] = vis_pred_items[p_id][item]

    vis_pred = vision_prediction_postprocess(temp_dict)
    return vis_pred


def create_vis_pred(
    pid_list, vis_model, class_names, vision_yaml_dict_inference_config, f1_scores
):
    """
    Creates vision predictions for a list of patient IDs.

    Args:
        pid_list (list): List of patient IDs.
        vis_model (object): Vision model for inference.
        class_names (list): List of class names.
        vision_yaml_dict_inference_config (dict): Vision inference configuration dictionary.
        f1_scores (list): F1 scores for the vision predictions.

    Returns:
        list: Corrected vision prediction scores based on F1 scores.
    """
    vis_dict = {}
    for pid in pid_list:
        image_paths = find_image_paths(
            vision_yaml_dict_inference_config["args"]["dataset_dir"], pid
        )
        vis_dict[pid] = image_paths

    vis_pred_items = run_inference_per_patient(vis_model, vis_dict, class_names)
    yaml_dict_inference = vis_pred_post_process(vis_pred_items, class_names)

    # Corrected scores based on f1-scores
    corrected_scores = [
        list(f1_scores * i) for i in yaml_dict_inference["predictions_probabilities"]
    ]

    return corrected_scores
