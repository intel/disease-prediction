# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

FROM intel/ai-workflows:pa-vision-tlt-disease-prediction

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    git \
    libgl1-mesa-glx \
    python3-dev \
    wget

COPY . /workspace

WORKDIR /workspace

RUN python -m pip install --no-cache-dir dataset-librarian

RUN python -m pip install --no-cache-dir -r /workspace/hf_nlp/workflows/hf_finetuning_and_inference_nlp/requirements.txt
