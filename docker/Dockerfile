# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

FROM intel/ai-workflows:transfer-learning-0.4.0

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    wget

COPY . /workspace

WORKDIR /workspace

RUN python -m pip install --no-cache-dir -r /workspace/intel-models/datasets/dataset_api/requirements.txt

RUN python -m pip install --no-cache-dir -r /workspace/vision_wf/workflows/disease_prediction/requirements.txt

RUN python -m pip install --no-cache-dir -r /workspace/hf_nlp/workflows/hf_finetuning_and_inference_nlp/requirements.txt
