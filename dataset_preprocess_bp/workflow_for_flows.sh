#!/bin/bash

if [ $# -eq 0 ]; then
    >&2 echo "No arguments provided"
    exit 1
fi

while [[ "$#" -gt "0" ]]; do
    case "$1" in
         --preprocess)
          echo "Preprocess"
          echo "Downloading dataset from S3 bucket"
          # Takes around 2 hours
          bash CDD-CESM_downloader.sh
          echo "Download completed"
          mv /cnvrg/PKG_CDD_CESM/ /cnvrg/data
          mv /cnvrg/data/CDD_CESM/ /cnvrg/data/CDD-CESM/
          mv /cnvrg/data/CDD-CESM/Low_energy_images_of_CDD_CESM /cnvrg/data/CDD-CESM/Low\ energy\ images\ of\ CDD-CESM
          mv /cnvrg/data/CDD-CESM/Subtracted_images_of_CDD_CESM /cnvrg/data/CDD-CESM/Subtracted\ images\ of\ CDD-CESM
          yes | python -m dataset_librarian.dataset -n brca --download --preprocess -d /cnvrg/data/ --split_ratio 0.1
          shift
          ;;
        --nlp)
            echo "NLP"
            mkdir /cnvrg/output 
           # cp -r $INPUT_VISION/output/ /cnvrg/
            rm -rf /workspace/output
            ln -s /cnvrg/output /workspace/
            ln -s /cnvrg/data /workspace/data
            cd /workspace/
            python src/run.py --config_file /cnvrg/configs/nlp_finetune.yaml
            shift
            ;;
        --vision)
            echo "Vision"
            #cp -r $INPUT_NLP/output/ /cnvrg/
			mkdir /cnvrg/output
			rm -rf /workspace/output
			ln -s /cnvrg/output /workspace/
			ln -s /cnvrg/data /workspace/data
			cd /workspace
			python src/run.py --config_file /cnvrg/configs/vision_finetune.yaml
            shift
            ;;
        --inference)
            echo "Inference"
            cp -r $INPUT_NLP/output/ /cnvrg/
            cp -r $INPUT_VISION/output/ /cnvrg/
            rm -rf /workspace/output
            ln -s /cnvrg/output /workspace/
            ln -s /cnvrg/data/ /workspace/data
            cd /workspace/
            python src/breast_cancer_prediction.py --config_file /cnvrg/configs/disease_prediction_container.yaml
            shift
            ;;
        **)
            echo "Wrong argument passed"
            exit 1
    esac
done