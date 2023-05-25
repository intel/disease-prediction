# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause


import yaml
import argparse
from copy import deepcopy
from os import path

# Import necessary functions from custom modules
from ensemble import Ensemble
from run_workflow import RunWorkflow
from utils.utils import get_subject_id, report_the_results, update_config_file
from utils.data import data_preparation


def read_config():
    """Function to read the configuration parameters from the input file"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    
    parser.add_argument(
        "--finetune",
        # type=str,
        help="Execute the finetune",
        default=None
    )
    
    parser.add_argument(
        "--inference",
        # type=str,
        help="Execute the inference",
        default=None
    )
        
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
        
    return config, args.finetune, args.inference

def main():
    # Set the root folder to the current directory of the script
    root_folder = path.dirname(path.abspath(__file__))

    # read config parameters
    config, finetune, inference = read_config()
    
    # update config params
    config = update_config_file(config, finetune, inference, root_folder)

    # data preparation
    config = data_preparation(config, root_folder)

    # load the class to run workflows
    run_wf = RunWorkflow(root_folder)

    # set nlp-finetune
    config = run_wf.set_config_and_run(
        config, "nlp", "finetune", "nlp_finetune.yaml", "hf_nlp/workflows/hf_finetuning_and_inference_nlp/src/run.py"
    )

    # run nlp-inference
    config = run_wf.set_config_and_run(
        config, "nlp", "inference", "nlp_inference.yaml", "hf_nlp/workflows/hf_finetuning_and_inference_nlp/src/run.py"
    )

    # run vision fine-tuning
    temp_inference = deepcopy(config["vision"]["args"]["inference"])
    config["vision"]["args"]["inference"] = False
    config = run_wf.set_config_and_run(
        config,
        "vision",
        "finetune",
        "vision_finetune.yaml",
        "vision_wf/workflows/disease_prediction/src/run.py",
    )
    config["vision"]["args"]["inference"] = temp_inference
    
    # vision inference
    temp_inference = deepcopy(config["vision"]["args"]["finetune"])
    config["vision"]["args"]["finetune"] = False
    config = run_wf.set_config_and_run(
        config,
        "vision",
        "inference",
        "vision_inference.yaml",
        "vision_wf/workflows/disease_prediction/src/run.py",
    )
    config["vision"]["args"]["finetune"] = temp_inference

       
    if config["nlp"]["args"]["inference"] or config["vision"]["args"]["inference"]:
        # ensemble
        df_results = Ensemble().ensemble(config)

        # # Report the results
        prediction_list = ["vision_predictions", "nlp_predictions", "ensemble_predictions"]
        report_the_results(df_results, "labels", prediction_list, path.join(root_folder, config['output_dir']  ) )

    print("===== Multi-Modal Disease Prediction is completed =====")


if __name__ == "__main__":
    main()
