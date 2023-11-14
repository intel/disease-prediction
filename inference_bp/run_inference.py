"""
Copyright [2022-23] [Intel Corporation]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#!/usr/bin/env python
# coding: utf-8
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--input_path",
            required=False,
            type=str,
            help="specify the input source file path for running the workflow", default="/workspace/src/breast_cancer_prediction.py")
    parser.add_argument(
            "--config_path",
            required=False,
            type=str,
            default="disease_prediction_container.yaml",
            help="specify the config file name")
    
    args, _ = parser.parse_known_args()
    print("PATH",os.path.dirname(os.path.abspath(__file__)))
    src_file = args.input_path
    config_path = args.config_path
    cmd_line = f"python {src_file} --config_file {config_path}"
    print(cmd_line)
    os.system(cmd_line)