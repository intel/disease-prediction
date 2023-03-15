# Copyright (C) 2022 Intel Corporation
#  
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#  
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#  
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  
#  
# SPDX-License-Identifier: BSD-3-Clause

FROM ubuntu:22.04 as build

ENV LANG C.UTF-8
ARG PYTHON=python3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    libgl1 \
    libglib2.0-0 \
    make \
    pandoc \
    ${PYTHON} \
    ${PYTHON}-dev \
    ${PYTHON}-distutils && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/bin/python

RUN curl -fSsL https://bootstrap.pypa.io/get-pip.py | python

COPY ../nlp /disease-prediction/nlp
COPY ../vision /disease-prediction/vision
COPY ../hls_wrapper.py /disease-prediction/wrapper.py

WORKDIR /disease-prediction

RUN cd ${PWD}/vision/tlt_toolkit && \
    ${PYTHON} -m pip install --no-cache-dir --editable .[tensorflow,pytorch] && \
    ${PYTHON} -m pip install --no-cache-dir tensorflow-text==2.10.0

RUN ${PYTHON} -m pip install --no-cache-dir -r ./nlp/requirements.txt

ENV PYTHONPATH=${PWD}/vision/tlt_toolkit
ENV DATASET_DIR=data
ENV OUTPUT_DIR=output
