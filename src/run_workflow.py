# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import yaml
import subprocess

from os import path
from pathlib import Path


class RunWorkflow(object):
    def __init__(self, root_folder):
        """
        Initialize ModelRunner with root folder where the models and configs are stored.
        """

        self.root_folder = root_folder
        pass

    def set_config_and_run(self, config, wf, pipeline, yaml_file, run_file):

        """
        Set the configuration and run the pipeline.

        Args:
            config (dict): the configuration to set.
            wf (str): the workflow to run.
            pipeline (str): the pipeline to run, either "finetune" or "inference".
            yaml_file (str): the name of the yaml file to create.
            run_file (str): the name of the run file to execute.

        Returns:
            The updated configuration.
        """

        pipeline_func = self.fine_tuning if pipeline == "finetune" else self.inference

        config[wf]["args"]["pipeline"] = pipeline
        config[wf]["args"][pipeline + "_output"] = path.join(
            self.root_folder,
            config["output_dir"],
            wf,
            config[wf]["args"][pipeline + "_output"],
        )

        # running finetuning, config, name of the yaml file, location of the run file
        if config[wf]["args"][pipeline]:
            pipeline_func(config[wf], yaml_file, run_file)

        return config

    def run_system_call(self, config, yaml_file, run_file):
        """
        Run a system call with the given configuration and run file.

        Args:
            config (dict): the configuration to use.
            yaml_file (str): the name of the yaml file to create.
            run_file (str): the name of the run file to execute.
        """

        yaml_path = create_yaml_file(
            config, path.join(self.root_folder, "../configs"), yaml_file
        )

        cmd = (
            "python "
            + path.join(self.root_folder, "../", run_file)
            + " --config_file "
            + yaml_path
        ).split(' ')
        
        process = subprocess.Popen(cmd) 
        process.wait()

    # fine-tuning
    def fine_tuning(self, config, yaml_file, run_file):
        """
        Run the finetuning pipeline with the given configuration and run file.

        Args:
            config (dict): the configuration to use.
            yaml_file (str): the name of the yaml file to create.
            run_file (str): the name of the run file to execute.
        """
        self.run_system_call(config, yaml_file, run_file)

    # nlp inference
    def inference(self, config, yaml_file, run_file):
        """
        Run the inference pipeline with the given configuration and run file.

        Args:
            config (dict): the configuration to use.
            yaml_file (str): the name of the yaml file to create.
            run_file (str): the name of the run file to execute.
        """
        output_dir = config["training_args"]["output_dir"]
        config["args"]["model_name_or_path"] = output_dir
        config["args"]["tokenizer_name"] = output_dir

        self.run_system_call(config, yaml_file, run_file)


# create yaml file
def create_yaml_file(config, output_dir, yaml_file):
    """
    Create a yaml file with the given configuration.

    Args:
        config (dict): the configuration to write to the file.
        output_dir (str) : location of the output folder
        yaml_file (str): the name of the yaml file to create.

    Returns:
        The path to the created yaml file.
    """

    # Create directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set the path to the YAML file
    yaml_path = path.join(output_dir, yaml_file)

    # Write the configuration to the YAML file
    if config["write"]:
        with open(yaml_path, "w") as file:
            _ = yaml.dump(config, file)
    return yaml_path
