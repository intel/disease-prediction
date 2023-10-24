#!/usr/bin/env bash

#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#

# Note: For furthur details please visit:
# https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/260?passcode=5335d2514638afdaf03237780dcdfec29edf4238#

set +xe

AWS_BUCKET_URL="https://intelai-datasets.s3.cn-north-1.amazonaws.com.cn"

declare -a REMOTE_DIRS
declare -a LOCAL_DIRS

REMOTE_DIRS+=( "PKG+-+CDD-CESM/CDD-CESM/Low+energy+images+of+CDD-CESM" "PKG+-+CDD-CESM/CDD-CESM/Subtracted+images+of+CDD-CESM" )

for REMOTE_DIR in ${REMOTE_DIRS[@]}; do
  LOCAL_DIR=$(echo ${REMOTE_DIR} | sed -r 's/[+-]+/_/g')
  if [ ! -d ${LOCAL_DIR} ]; then
    mkdir -p ${LOCAL_DIR};
    fi
  LOCAL_DIRS+=( ${LOCAL_DIR} )
done

read -r -d '' IMAGES << EOM
P100_L_DM_CC.jpg
P100_L_DM_MLO.jpg
P100_R_DM_CC.jpg
P100_R_DM_MLO.jpg
P101_L_DM_CC.jpg
P101_L_DM_MLO.jpg
P101_R_DM_CC.jpg
P101_R_DM_MLO.jpg
P102_L_DM_CC.jpg
P102_L_DM_MLO.jpg
P102_R_DM_CC.jpg
P102_R_DM_MLO.jpg
P103_L_DM_CC.jpg
P103_L_DM_MLO.jpg
P103_R_DM_CC.jpg
P103_R_DM_MLO.jpg
P104_L_DM_MLO.jpg
P104_R_DM_CC.jpg
P105_L_DM_CC.jpg
P105_L_DM_MLO.jpg
P105_R_DM_CC.jpg
P105_R_DM_MLO.jpg
P106_L_DM_MLO.jpg
P106_R_DM_CC.jpg
P107_L_DM_CC.jpg
P107_R_DM_CC.jpg
P107_R_DM_MLO.jpg
P108_L_DM_CC.jpg
P108_R_DM_CC.jpg
P109_L_DM_CC.jpg
P10_L_DM_CC.jpg
P10_L_DM_MLO.jpg
P10_R_DM_CC.jpg
P10_R_DM_MLO.jpg
P110_L_DM_MLO.jpg
P110_R_DM_CC.jpg
P110_R_DM_MLO.jpg
P111_L_DM_CC.jpg
P111_L_DM_MLO.jpg
P111_R_DM_CC.jpg
P111_R_DM_MLO.jpg
P112_L_DM_CC.jpg
P113_L_DM_CC.jpg
P113_L_DM_MLO.jpg
P113_R_DM_MLO.jpg
P114_L_DM_CC.jpg
P114_L_DM_MLO.jpg
P114_R_DM_CC.jpg
P114_R_DM_MLO.jpg
P115_L_DM_CC.jpg
P115_L_DM_MLO.jpg
P115_R_DM_CC.jpg
P115_R_DM_MLO.jpg
P116_L_DM_MLO.jpg
P116_R_DM_CC.jpg
P116_R_DM_MLO.jpg
P117_R_DM_CC.jpg
P118_L_DM_CC.jpg
P118_R_DM_CC.jpg
P118_R_DM_MLO.jpg
P119_L_DM_CC.jpg
P119_L_DM_MLO.jpg
P119_R_DM_CC.jpg
P119_R_DM_MLO.jpg
P11_R_DM_CC.jpg
P11_R_DM_MLO.jpg
P120_L_DM_MLO.jpg
P121_L_DM_CC.jpg
P121_R_DM_MLO.jpg
P122_L_DM_CC.jpg
P122_L_DM_MLO.jpg
P122_R_DM_CC.jpg
P122_R_DM_MLO.jpg
P123_L_DM_CC.jpg
P123_L_DM_MLO.jpg
P123_R_DM_CC.jpg
P123_R_DM_MLO.jpg
P124_L_DM_CC.jpg
P124_L_DM_MLO.jpg
P124_R_DM_CC.jpg
P124_R_DM_MLO.jpg
P125_L_DM_CC.jpg
P125_L_DM_MLO.jpg
P125_R_DM_CC.jpg
P125_R_DM_MLO.jpg
P126_L_DM_CC.jpg
P126_L_DM_MLO.jpg
P126_R_DM_CC.jpg
P126_R_DM_MLO.jpg
P127_L_DM_MLO.jpg
P128_L_DM_CC.jpg
P128_L_DM_MLO.jpg
P128_R_DM_CC.jpg
P128_R_DM_MLO.jpg
P129_L_DM_CC.jpg
P129_L_DM_MLO.jpg
P129_R_DM_CC.jpg
P129_R_DM_MLO.jpg
P12_L_DM_CC.jpg
P12_L_DM_MLO.jpg
P12_R_DM_CC.jpg
P12_R_DM_MLO.jpg
P130_L_DM_CC.jpg
P131_L_DM_MLO.jpg
P132_L_DM_CC.jpg
P132_L_DM_MLO.jpg
P132_R_DM_CC.jpg
P132_R_DM_MLO.jpg
P133_R_DM_CC.jpg
P134_R_DM_MLO.jpg
P135_L_DM_MLO.jpg
P136_L_DM_MLO.jpg
P136_R_DM_CC.jpg
P136_R_DM_MLO.jpg
P137_L_DM_MLO.jpg
P138_L_DM_CC.jpg
P138_L_DM_MLO.jpg
P138_R_DM_CC.jpg
P138_R_DM_MLO.jpg
P139_R_DM_CC.jpg
P139_R_DM_MLO.jpg
P13_R_DM_CC.jpg
P140_L_DM_CC.jpg
P140_L_DM_MLO.jpg
P140_R_DM_CC.jpg
P140_R_DM_MLO.jpg
P141_L_DM_CC.jpg
P141_L_DM_MLO.jpg
P141_R_DM_CC.jpg
P141_R_DM_MLO.jpg
P142_R_DM_CC.jpg
P142_R_DM_MLO.jpg
P143_L_DM_CC.jpg
P143_L_DM_MLO.jpg
P143_R_DM_CC.jpg
P143_R_DM_MLO.jpg
P144_L_DM_CC.jpg
P144_L_DM_MLO.jpg
P144_R_DM_CC.jpg
P144_R_DM_MLO.jpg
P145_L_DM_CC.jpg
P145_L_DM_MLO.jpg
P145_R_DM_CC.jpg
P145_R_DM_MLO.jpg
P146_L_DM_CC.jpg
P146_L_DM_MLO.jpg
P146_R_DM_CC.jpg
P146_R_DM_MLO.jpg
P147_R_DM_CC.jpg
P147_R_DM_MLO.jpg
P148_L_DM_CC.jpg
P148_L_DM_MLO.jpg
P148_R_DM_CC.jpg
P148_R_DM_MLO.jpg
P149_L_DM_CC.jpg
P149_L_DM_MLO.jpg
P149_R_DM_CC.jpg
P149_R_DM_MLO.jpg
P14_L_DM_CC.jpg
P14_L_DM_MLO.jpg
P14_R_DM_CC.jpg
P14_R_DM_MLO.jpg
P150_L_DM_CC.jpg
P150_L_DM_MLO.jpg
P150_R_DM_CC.jpg
P150_R_DM_MLO.jpg
P151_L_DM_CC.jpg
P152_L_DM_CC.jpg
P152_L_DM_MLO.jpg
P152_R_DM_CC.jpg
P153_L_DM_CC.jpg
P153_L_DM_MLO.jpg
P153_R_DM_MLO.jpg
P154_L_DM_CC.jpg
P154_L_DM_MLO.jpg
P154_R_DM_CC.jpg
P154_R_DM_MLO.jpg
P155_L_DM_CC.jpg
P155_L_DM_MLO.jpg
P155_R_DM_CC.jpg
P155_R_DM_MLO.jpg
P156_L_DM_CC.jpg
P156_L_DM_MLO.jpg
P156_R_DM_CC.jpg
P156_R_DM_MLO.jpg
P157_R_DM_CC.jpg
P158_L_DM_MLO.jpg
P159_L_DM_CC.jpg
P15_L_DM_MLO.jpg
P15_R_DM_MLO.jpg
P160_L_DM_CC.jpg
P160_R_DM_CC.jpg
P161_L_DM_CC.jpg
P161_L_DM_MLO.jpg
P161_R_DM_CC.jpg
P161_R_DM_MLO.jpg
P162_L_DM_CC.jpg
P162_L_DM_MLO.jpg
P162_R_DM_CC.jpg
P162_R_DM_MLO.jpg
P163_R_DM_CC.jpg
P164_L_DM_CC.jpg
P165_L_DM_CC.jpg
P165_L_DM_MLO.jpg
P165_R_DM_CC.jpg
P165_R_DM_MLO.jpg
P166_L_DM_CC.jpg
P167_L_DM_MLO.jpg
P168_L_DM_MLO.jpg
P169_L_DM_CC.jpg
P169_L_DM_MLO.jpg
P169_R_DM_CC.jpg
P16_L_DM_CC.jpg
P16_L_DM_MLO.jpg
P16_R_DM_CC.jpg
P16_R_DM_MLO.jpg
P170_L_DM_CC.jpg
P170_L_DM_MLO.jpg
P170_R_DM_CC.jpg
P170_R_DM_MLO.jpg
P171_L_DM_MLO.jpg
P172_L_DM_CC.jpg
P172_L_DM_MLO.jpg
P172_R_DM_CC.jpg
P172_R_DM_MLO.jpg
P173_L_DM_CC.jpg
P173_L_DM_MLO.jpg
P173_R_DM_CC.jpg
P173_R_DM_MLO.jpg
P174_L_DM_CC.jpg
P174_L_DM_MLO.jpg
P174_R_DM_CC.jpg
P174_R_DM_MLO.jpg
P175_R_DM_CC.jpg
P175_R_DM_MLO.jpg
P176_L_DM_CC.jpg
P176_L_DM_MLO.jpg
P176_R_DM_CC.jpg
P176_R_DM_MLO.jpg
P177_L_DM_CC.jpg
P177_L_DM_MLO.jpg
P177_R_DM_CC.jpg
P177_R_DM_MLO.jpg
P178_R_DM_MLO.jpg
P179_L_DM_CC.jpg
P179_L_DM_MLO.jpg
P17_R_DM_MLO.jpg
P180_L_DM_CC.jpg
P180_L_DM_MLO.jpg
P180_R_DM_CC.jpg
P180_R_DM_MLO.jpg
P181_L_DM_CC.jpg
P181_L_DM_MLO.jpg
P181_R_DM_CC.jpg
P181_R_DM_MLO.jpg
P182_R_DM_MLO.jpg
P183_L_DM_CC.jpg
P183_L_DM_MLO.jpg
P183_R_DM_CC.jpg
P183_R_DM_MLO.jpg
P184_L_DM_CC.jpg
P184_L_DM_MLO.jpg
P184_R_DM_CC.jpg
P184_R_DM_MLO.jpg
P185_L_DM_CC.jpg
P185_L_DM_MLO.jpg
P185_R_DM_CC.jpg
P185_R_DM_MLO.jpg
P186_L_DM_CC.jpg
P186_L_DM_MLO.jpg
P186_R_DM_CC.jpg
P187_L_DM_CC.jpg
P187_R_DM_CC.jpg
P188_L_DM_CC.jpg
P188_L_DM_MLO.jpg
P188_R_DM_CC.jpg
P188_R_DM_MLO.jpg
P189_L_DM_CC.jpg
P189_R_DM_MLO.jpg
P18_L_DM_CC.jpg
P18_L_DM_MLO.jpg
P18_R_DM_CC.jpg
P18_R_DM_MLO.jpg
P190_L_DM_CC.jpg
P190_L_DM_MLO.jpg
P190_R_DM_MLO.jpg
P191_L_DM_CC.jpg
P191_L_DM_MLO.jpg
P191_R_DM_CC.jpg
P191_R_DM_MLO.jpg
P192_L_DM_CC.jpg
P192_L_DM_MLO.jpg
P192_R_DM_CC.jpg
P192_R_DM_MLO.jpg
P193_L_DM_MLO.jpg
P194_L_DM_CC.jpg
P194_L_DM_MLO.jpg
P194_R_DM_CC.jpg
P194_R_DM_MLO.jpg
P195_L_DM_CC.jpg
P195_L_DM_MLO.jpg
P195_R_DM_CC.jpg
P195_R_DM_MLO.jpg
P196_L_DM_CC.jpg
P196_L_DM_MLO.jpg
P196_R_DM_MLO.jpg
P197_L_DM_CC.jpg
P197_L_DM_MLO.jpg
P197_R_DM_CC.jpg
P197_R_DM_MLO.jpg
P198_L_DM_CC.jpg
P198_L_DM_MLO.jpg
P199_R_DM_CC.jpg
P19_L_DM_CC.jpg
P19_L_DM_MLO.jpg
P19_R_DM_MLO.jpg
P1_L_DM_MLO.jpg
P200_L_DM_CC.jpg
P200_L_DM_MLO.jpg
P200_R_DM_CC.jpg
P200_R_DM_MLO.jpg
P201_R_DM_MLO.jpg
P202_L_DM_CC.jpg
P202_L_DM_MLO.jpg
P202_R_DM_CC.jpg
P202_R_DM_MLO.jpg
P203_R_DM_CC.jpg
P203_R_DM_MLO.jpg
P204_L_DM_CC.jpg
P205_L_DM_MLO.jpg
P206_L_DM_CC.jpg
P206_L_DM_MLO.jpg
P206_R_DM_CC.jpg
P206_R_DM_MLO.jpg
P207_L_DM_CC.jpg
P207_L_DM_MLO.jpg
P207_R_DM_CC.jpg
P207_R_DM_MLO.jpg
P208_R_DM_CC.jpg
P208_R_DM_MLO.jpg
P209_L_DM_CC.jpg
P209_L_DM_MLO.jpg
P209_R_DM_CC.jpg
P209_R_DM_MLO.jpg
P20_R_DM_CC.jpg
P20_R_DM_MLO.jpg
P210_L_DM_CC.jpg
P210_L_DM_MLO.jpg
P211_L_DM_CC.jpg
P211_L_DM_MLO.jpg
P211_R_DM_CC.jpg
P211_R_DM_MLO.jpg
P212_L_DM_CC.jpg
P212_L_DM_MLO.jpg
P212_R_DM_CC.jpg
P212_R_DM_MLO.jpg
P213_L_DM_CC.jpg
P213_L_DM_MLO.jpg
P213_R_DM_CC.jpg
P213_R_DM_MLO.jpg
P214_L_DM_CC.jpg
P214_L_DM_MLO.jpg
P214_R_DM_CC.jpg
P214_R_DM_MLO.jpg
P215_L_DM_CC.jpg
P215_R_DM_CC.jpg
P215_R_DM_MLO.jpg
P216_L_DM_CC.jpg
P216_L_DM_MLO.jpg
P216_R_DM_CC.jpg
P216_R_DM_MLO.jpg
P217_L_DM_CC.jpg
P217_L_DM_MLO.jpg
P218_L_DM_MLO.jpg
P218_R_DM_CC.jpg
P219_L_DM_MLO.jpg
P21_L_DM_CC.jpg
P21_L_DM_MLO.jpg
P21_R_DM_CC.jpg
P21_R_DM_MLO.jpg
P220_L_DM_CC.jpg
P220_L_DM_MLO.jpg
P220_R_DM_CC.jpg
P220_R_DM_MLO.jpg
P221_L_DM_CC.jpg
P222_L_DM_CC.jpg
P223_L_DM_CC.jpg
P223_L_DM_MLO.jpg
P223_R_DM_MLO.jpg
P224_L_DM_CC.jpg
P225_L_DM_CC.jpg
P225_L_DM_MLO.jpg
P225_R_DM_CC.jpg
P225_R_DM_MLO.jpg
P226_L_DM_CC.jpg
P226_L_DM_MLO.jpg
P226_R_DM_CC.jpg
P226_R_DM_MLO.jpg
P227_L_DM_CC.jpg
P227_L_DM_MLO.jpg
P227_R_DM_CC.jpg
P227_R_DM_MLO.jpg
P228_L_DM_CC.jpg
P228_L_DM_MLO.jpg
P228_R_DM_CC.jpg
P228_R_DM_MLO.jpg
P229_R_DM_MLO.jpg
P22_L_DM_CC.jpg
P230_L_DM_CC.jpg
P230_L_DM_MLO.jpg
P230_R_DM_CC.jpg
P230_R_DM_MLO.jpg
P231_R_DM_MLO.jpg
P232_L_DM_CC.jpg
P232_L_DM_MLO.jpg
P232_R_DM_CC.jpg
P232_R_DM_MLO.jpg
P233_L_DM_CC.jpg
P233_L_DM_MLO.jpg
P233_R_DM_CC.jpg
P233_R_DM_MLO.jpg
P234_L_DM_CC.jpg
P235_L_DM_CC.jpg
P235_L_DM_MLO.jpg
P235_R_DM_CC.jpg
P235_R_DM_MLO.jpg
P236_L_DM_CC.jpg
P236_L_DM_MLO.jpg
P236_R_DM_CC.jpg
P236_R_DM_MLO.jpg
P237_L_DM_CC.jpg
P237_L_DM_MLO.jpg
P237_R_DM_CC.jpg
P237_R_DM_MLO.jpg
P238_L_DM_CC.jpg
P238_L_DM_MLO.jpg
P238_R_DM_CC.jpg
P238_R_DM_MLO.jpg
P239_L_DM_CC.jpg
P239_L_DM_MLO.jpg
P239_R_DM_CC.jpg
P239_R_DM_MLO.jpg
P23_R_DM_CC.jpg
P240_R_DM_MLO.jpg
P241_L_DM_CC.jpg
P241_R_DM_CC.jpg
P242_L_DM_CC.jpg
P242_R_DM_MLO.jpg
P243_L_DM_CC.jpg
P243_L_DM_MLO.jpg
P243_R_DM_CC.jpg
P243_R_DM_MLO.jpg
P244_L_DM_CC.jpg
P244_L_DM_MLO.jpg
P244_R_DM_CC.jpg
P244_R_DM_MLO.jpg
P245_L_DM_CC.jpg
P245_R_DM_CC.jpg
P246_R_DM_MLO.jpg
P247_L_DM_MLO.jpg
P248_L_DM_CC.jpg
P248_L_DM_MLO.jpg
P248_R_DM_MLO.jpg
P249_L_DM_CC.jpg
P249_L_DM_MLO.jpg
P249_R_DM_CC.jpg
P249_R_DM_MLO.jpg
P24_L_DM_CC.jpg
P24_L_DM_MLO.jpg
P24_R_DM_CC.jpg
P24_R_DM_MLO.jpg
P250_L_DM_CC.jpg
P251_L_DM_CC.jpg
P251_L_DM_MLO.jpg
P251_R_DM_CC.jpg
P251_R_DM_MLO.jpg
P252_L_DM_CC.jpg
P252_L_DM_MLO.jpg
P252_R_DM_CC.jpg
P252_R_DM_MLO.jpg
P253_L_DM_CC.jpg
P253_L_DM_MLO.jpg
P253_R_DM_CC.jpg
P253_R_DM_MLO.jpg
P254_L_DM_CC.jpg
P254_L_DM_MLO.jpg
P254_R_DM_CC.jpg
P254_R_DM_MLO.jpg
P255_L_DM_CC.jpg
P255_L_DM_MLO.jpg
P255_R_DM_CC.jpg
P255_R_DM_MLO.jpg
P256_L_DM_CC.jpg
P256_L_DM_MLO.jpg
P257_L_DM_CC.jpg
P257_L_DM_MLO.jpg
P257_R_DM_CC.jpg
P257_R_DM_MLO.jpg
P258_L_DM_CC.jpg
P258_L_DM_MLO.jpg
P258_R_DM_CC.jpg
P258_R_DM_MLO.jpg
P259_L_DM_CC.jpg
P259_L_DM_MLO.jpg
P259_R_DM_CC.jpg
P259_R_DM_MLO.jpg
P25_L_DM_MLO.jpg
P25_R_DM_CC.jpg
P260_L_DM_CC.jpg
P260_L_DM_MLO.jpg
P260_R_DM_CC.jpg
P260_R_DM_MLO.jpg
P261_L_DM_CC.jpg
P261_L_DM_MLO.jpg
P261_R_DM_CC.jpg
P261_R_DM_MLO.jpg
P262_L_DM_CC.jpg
P262_L_DM_MLO.jpg
P262_R_DM_CC.jpg
P262_R_DM_MLO.jpg
P263_L_DM_CC.jpg
P263_L_DM_MLO.jpg
P263_R_DM_CC.jpg
P263_R_DM_MLO.jpg
P264_L_DM_CC.jpg
P264_L_DM_MLO.jpg
P264_R_DM_CC.jpg
P264_R_DM_MLO.jpg
P265_L_DM_CC.jpg
P265_L_DM_MLO.jpg
P265_R_DM_CC.jpg
P265_R_DM_MLO.jpg
P266_L_DM_CC.jpg
P266_L_DM_MLO.jpg
P266_R_DM_CC.jpg
P266_R_DM_MLO.jpg
P267_L_DM_CC.jpg
P267_L_DM_MLO.jpg
P267_R_DM_CC.jpg
P267_R_DM_MLO.jpg
P268_L_DM_CC.jpg
P268_L_DM_MLO.jpg
P268_R_DM_CC.jpg
P268_R_DM_MLO.jpg
P269_L_DM_CC.jpg
P269_L_DM_MLO.jpg
P269_R_DM_CC.jpg
P269_R_DM_MLO.jpg
P26_L_DM_CC.jpg
P26_L_DM_MLO.jpg
P270_L_DM_CC.jpg
P270_L_DM_MLO.jpg
P270_R_DM_CC.jpg
P270_R_DM_MLO.jpg
P271_L_DM_CC.jpg
P271_L_DM_MLO.jpg
P271_R_DM_CC.jpg
P271_R_DM_MLO.jpg
P272_L_DM_CC.jpg
P272_L_DM_MLO.jpg
P272_R_DM_CC.jpg
P272_R_DM_MLO.jpg
P273_L_DM_CC.jpg
P273_L_DM_MLO.jpg
P273_R_DM_CC.jpg
P273_R_DM_MLO.jpg
P274_L_DM_CC.jpg
P274_L_DM_MLO.jpg
P274_R_DM_CC.jpg
P274_R_DM_MLO.jpg
P275_L_DM_CC.jpg
P275_L_DM_MLO.jpg
P275_R_DM_CC.jpg
P275_R_DM_MLO.jpg
P276_L_DM_CC.jpg
P276_L_DM_MLO.jpg
P276_R_DM_CC.jpg
P276_R_DM_MLO.jpg
P277_L_DM_CC.jpg
P277_L_DM_MLO.jpg
P277_R_DM_CC.jpg
P277_R_DM_MLO.jpg
P278_L_DM_CC.jpg
P278_L_DM_MLO.jpg
P278_R_DM_CC.jpg
P278_R_DM_MLO.jpg
P279_L_DM_CC.jpg
P279_L_DM_MLO.jpg
P279_R_DM_CC.jpg
P279_R_DM_MLO.jpg
P27_L_DM_CC.jpg
P27_L_DM_MLO.jpg
P280_L_DM_CC.jpg
P280_L_DM_MLO.jpg
P280_R_DM_CC.jpg
P280_R_DM_MLO.jpg
P281_L_DM_CC.jpg
P281_L_DM_MLO.jpg
P281_L_DM_MLO2.jpg
P281_R_DM_CC.jpg
P281_R_DM_MLO.jpg
P282_L_DM_CC.jpg
P282_L_DM_MLO.jpg
P282_R_DM_CC.jpg
P282_R_DM_MLO.jpg
P283_L_DM_CC.jpg
P283_L_DM_MLO.jpg
P283_R_DM_CC.jpg
P283_R_DM_MLO.jpg
P284_L_DM_CC.jpg
P284_L_DM_MLO.jpg
P284_R_DM_CC.jpg
P284_R_DM_MLO.jpg
P285_L_DM_CC.jpg
P285_L_DM_MLO.jpg
P285_R_DM_CC.jpg
P285_R_DM_MLO.jpg
P286_L_DM_CC.jpg
P286_L_DM_MLO.jpg
P286_R_DM_CC.jpg
P286_R_DM_MLO.jpg
P287_L_DM_CC.jpg
P287_L_DM_MLO.jpg
P287_R_DM_CC.jpg
P287_R_DM_MLO.jpg
P288_L_DM_CC.jpg
P288_L_DM_MLO.jpg
P288_R_DM_CC.jpg
P288_R_DM_MLO.jpg
P289_L_DM_CC.jpg
P289_L_DM_MLO.jpg
P289_R_DM_CC.jpg
P289_R_DM_MLO.jpg
P28_L_DM_MLO.jpg
P28_R_DM_CC.jpg
P28_R_DM_MLO.jpg
P290_L_DM_CC.jpg
P290_L_DM_MLO.jpg
P290_R_DM_CC.jpg
P290_R_DM_MLO.jpg
P291_L_DM_CC.jpg
P291_L_DM_MLO.jpg
P291_R_DM_CC.jpg
P291_R_DM_MLO.jpg
P292_L_DM_CC.jpg
P292_L_DM_MLO.jpg
P292_R_DM_CC.jpg
P292_R_DM_MLO.jpg
P293_L_DM_CC.jpg
P293_L_DM_MLO.jpg
P293_R_DM_CC.jpg
P293_R_DM_MLO.jpg
P294_R_DM_CC.jpg
P294_R_DM_MLO.jpg
P295_L_DM_CC.jpg
P295_L_DM_MLO.jpg
P295_R_DM_CC.jpg
P295_R_DM_MLO.jpg
P296_L_DM_CC.jpg
P296_L_DM_MLO.jpg
P296_R_DM_CC.jpg
P296_R_DM_MLO.jpg
P297_L_DM_CC.jpg
P297_L_DM_MLO.jpg
P297_R_DM_CC.jpg
P297_R_DM_MLO.jpg
P298_L_DM_CC.jpg
P298_L_DM_MLO.jpg
P298_R_DM_CC.jpg
P298_R_DM_MLO.jpg
P299_L_DM_CC.jpg
P299_L_DM_MLO.jpg
P299_R_DM_CC.jpg
P299_R_DM_MLO.jpg
P29_L_DM_CC.jpg
P29_R_DM_CC.jpg
P29_R_DM_MLO.jpg
P2_L_DM_CC.jpg
P2_L_DM_MLO.jpg
P2_R_DM_CC.jpg
P2_R_DM_MLO.jpg
P300_L_DM_CC.jpg
P300_L_DM_MLO.jpg
P300_R_DM_CC.jpg
P300_R_DM_MLO.jpg
P301_L_DM_CC.jpg
P301_L_DM_MLO.jpg
P301_R_DM_CC.jpg
P301_R_DM_MLO.jpg
P302_L_DM_CC.jpg
P302_L_DM_MLO.jpg
P302_R_DM_CC.jpg
P302_R_DM_MLO.jpg
P303_L_DM_CC.jpg
P303_L_DM_MLO.jpg
P303_R_DM_CC.jpg
P303_R_DM_MLO.jpg
P304_L_DM_CC.jpg
P304_L_DM_MLO.jpg
P304_R_DM_CC.jpg
P304_R_DM_MLO.jpg
P305_L_DM_CC.jpg
P305_L_DM_MLO.jpg
P305_R_DM_CC.jpg
P305_R_DM_MLO.jpg
P306_L_DM_CC.jpg
P306_L_DM_MLO.jpg
P306_R_DM_CC.jpg
P306_R_DM_MLO.jpg
P307_L_DM_CC.jpg
P307_L_DM_MLO.jpg
P307_R_DM_CC.jpg
P307_R_DM_MLO.jpg
P308_L_DM_CC.jpg
P308_L_DM_MLO.jpg
P308_R_DM_CC.jpg
P308_R_DM_MLO.jpg
P309_L_DM_CC.jpg
P309_L_DM_MLO.jpg
P309_R_DM_CC.jpg
P309_R_DM_MLO.jpg
P30_L_DM_CC.jpg
P30_L_DM_MLO.jpg
P30_R_DM_CC.jpg
P30_R_DM_MLO.jpg
P310_L_DM_CC.jpg
P310_L_DM_MLO.jpg
P310_R_DM_CC.jpg
P310_R_DM_MLO.jpg
P311_L_DM_CC.jpg
P311_L_DM_MLO.jpg
P311_R_DM_CC.jpg
P311_R_DM_MLO.jpg
P312_L_DM_CC.jpg
P312_L_DM_MLO.jpg
P312_R_DM_CC.jpg
P312_R_DM_MLO.jpg
P313_L_DM_CC.jpg
P313_L_DM_MLO.jpg
P313_R_DM_CC.jpg
P313_R_DM_MLO.jpg
P314_L_DM_CC.jpg
P314_L_DM_MLO.jpg
P314_R_DM_CC.jpg
P314_R_DM_MLO.jpg
P315_L_DM_CC.jpg
P315_L_DM_MLO.jpg
P315_R_DM_CC.jpg
P315_R_DM_MLO.jpg
P316_L_DM_CC.jpg
P316_L_DM_MLO.jpg
P316_R_DM_CC.jpg
P316_R_DM_MLO.jpg
P317_L_DM_CC.jpg
P317_L_DM_MLO.jpg
P317_R_DM_CC.jpg
P317_R_DM_MLO.jpg
P318_L_DM_CC.jpg
P318_L_DM_MLO.jpg
P318_R_DM_CC.jpg
P318_R_DM_MLO.jpg
P319_L_DM_CC.jpg
P319_L_DM_MLO.jpg
P319_R_DM_CC.jpg
P319_R_DM_MLO.jpg
P31_L_DM_CC.jpg
P31_L_DM_MLO.jpg
P31_R_DM_CC.jpg
P31_R_DM_MLO.jpg
P320_L_DM_CC.jpg
P320_L_DM_MLO.jpg
P320_R_DM_CC.jpg
P320_R_DM_MLO.jpg
P321_R_DM_CC.jpg
P321_R_DM_MLO.jpg
P322_L_DM_CC.jpg
P322_L_DM_MLO.jpg
P322_R_DM_CC.jpg
P322_R_DM_MLO.jpg
P323_L_DM_CC.jpg
P323_L_DM_MLO.jpg
P323_R_DM_CC.jpg
P323_R_DM_MLO.jpg
P324_L_DM_CC.jpg
P324_L_DM_MLO.jpg
P324_R_DM_MLO.jpg
P325_L_DM_CC.jpg
P325_L_DM_MLO.jpg
P325_R_DM_CC.jpg
P325_R_DM_MLO.jpg
P326_L_DM_CC.jpg
P326_L_DM_MLO.jpg
P326_R_DM_CC.jpg
P326_R_DM_MLO.jpg
P32_L_DM_CC.jpg
P32_L_DM_MLO.jpg
P32_R_DM_CC.jpg
P32_R_DM_MLO.jpg
P33_L_DM_CC.jpg
P33_L_DM_MLO.jpg
P34_L_DM_CC.jpg
P34_R_DM_MLO.jpg
P35_L_DM_CC.jpg
P35_L_DM_MLO.jpg
P35_R_DM_MLO.jpg
P36_R_DM_MLO.jpg
P37_L_DM_CC.jpg
P37_L_DM_MLO.jpg
P37_R_DM_CC.jpg
P37_R_DM_MLO.jpg
P38_R_DM_CC.jpg
P38_R_DM_MLO.jpg
P39_R_DM_CC.jpg
P39_R_DM_MLO.jpg
P3_L_DM_CC.jpg
P3_L_DM_MLO.jpg
P3_R_DM_CC.jpg
P3_R_DM_MLO.jpg
P40_L_DM_CC.jpg
P40_L_DM_MLO.jpg
P40_R_DM_CC.jpg
P40_R_DM_MLO.jpg
P41_L_DM_CC.jpg
P42_R_DM_MLO.jpg
P43_R_DM_CC.jpg
P43_R_DM_MLO.jpg
P44_L_DM_CC.jpg
P44_L_DM_MLO.jpg
P44_R_DM_CC.jpg
P45_L_DM_CC.jpg
P45_L_DM_MLO.jpg
P45_R_DM_CC.jpg
P45_R_DM_MLO.jpg
P46_L_DM_CC.jpg
P46_L_DM_MLO.jpg
P46_R_DM_CC.jpg
P46_R_DM_MLO.jpg
P47_R_DM_CC.jpg
P48_L_DM_CC.jpg
P49_L_DM_CC.jpg
P49_L_DM_MLO.jpg
P49_R_DM_CC.jpg
P49_R_DM_MLO.jpg
P4_L_DM_MLO.jpg
P4_R_DM_CC.jpg
P50_L_DM_CC.jpg
P50_R_DM_MLO.jpg
P51_L_DM_CC.jpg
P51_L_DM_MLO.jpg
P51_R_DM_CC.jpg
P51_R_DM_MLO.jpg
P52_L_DM_CC.jpg
P52_L_DM_MLO.jpg
P52_R_DM_CC.jpg
P52_R_DM_MLO.jpg
P53_R_DM_CC.jpg
P54_L_DM_CC.jpg
P54_L_DM_MLO.jpg
P54_R_DM_CC.jpg
P54_R_DM_MLO.jpg
P55_L_DM_CC.jpg
P55_L_DM_MLO.jpg
P55_R_DM_CC.jpg
P55_R_DM_MLO.jpg
P56_L_DM_MLO.jpg
P56_R_DM_CC.jpg
P56_R_DM_MLO.jpg
P57_R_DM_CC.jpg
P58_L_DM_MLO.jpg
P59_L_DM_CC.jpg
P59_L_DM_MLO.jpg
P59_R_DM_CC.jpg
P59_R_DM_MLO.jpg
P5_L_DM_CC.jpg
P5_L_DM_MLO.jpg
P5_R_DM_CC.jpg
P5_R_DM_MLO.jpg
P60_L_DM_MLO.jpg
P61_L_DM_CC.jpg
P61_L_DM_MLO.jpg
P61_R_DM_CC.jpg
P61_R_DM_MLO.jpg
P62_L_DM_CC.jpg
P62_L_DM_MLO.jpg
P62_R_DM_CC.jpg
P62_R_DM_MLO.jpg
P63_R_DM_CC.jpg
P64_L_DM_CC.jpg
P64_L_DM_MLO.jpg
P64_R_DM_CC.jpg
P65_L_DM_CC.jpg
P65_L_DM_MLO.jpg
P65_R_DM_CC.jpg
P65_R_DM_MLO.jpg
P66_L_DM_CC.jpg
P66_L_DM_MLO.jpg
P66_R_DM_CC.jpg
P66_R_DM_MLO.jpg
P67_L_DM_CC.jpg
P67_R_DM_MLO.jpg
P68_L_DM_CC.jpg
P68_L_DM_MLO.jpg
P69_L_DM_CC.jpg
P69_L_DM_MLO.jpg
P6_L_DM_MLO.jpg
P6_R_DM_CC.jpg
P70_L_DM_CC.jpg
P71_L_DM_MLO.jpg
P72_L_DM_CC.jpg
P72_R_DM_CC.jpg
P73_L_DM_CC.jpg
P73_L_DM_MLO.jpg
P73_R_DM_CC.jpg
P73_R_DM_MLO.jpg
P74_L_DM_CC.jpg
P74_L_DM_MLO.jpg
P74_R_DM_CC.jpg
P74_R_DM_MLO.jpg
P75_R_DM_MLO.jpg
P76_L_DM_CC.jpg
P76_L_DM_MLO.jpg
P76_R_DM_CC.jpg
P76_R_DM_MLO.jpg
P77_L_DM_MLO.jpg
P77_R_DM_CC.jpg
P78_L_DM_CC.jpg
P79_L_DM_CC.jpg
P79_L_DM_MLO.jpg
P7_L_DM_MLO.jpg
P7_R_DM_CC.jpg
P7_R_DM_MLO.jpg
P80_L_DM_CC.jpg
P80_L_DM_MLO.jpg
P80_R_DM_MLO.jpg
P81_L_DM_CC.jpg
P81_L_DM_MLO.jpg
P81_R_DM_CC.jpg
P81_R_DM_MLO.jpg
P82_L_DM_CC.jpg
P82_L_DM_MLO.jpg
P82_R_DM_CC.jpg
P82_R_DM_MLO.jpg
P83_L_DM_CC.jpg
P83_L_DM_MLO.jpg
P83_R_DM_CC.jpg
P83_R_DM_MLO.jpg
P84_L_DM_CC.jpg
P84_L_DM_MLO.jpg
P84_R_DM_CC.jpg
P84_R_DM_MLO.jpg
P85_R_DM_MLO.jpg
P86_L_DM_CC.jpg
P86_R_DM_MLO.jpg
P87_L_DM_CC.jpg
P87_L_DM_MLO.jpg
P87_R_DM_CC.jpg
P87_R_DM_MLO.jpg
P88_L_DM_CC.jpg
P88_L_DM_MLO.jpg
P88_R_DM_CC.jpg
P88_R_DM_MLO.jpg
P89_L_DM_CC.jpg
P89_L_DM_MLO.jpg
P89_R_DM_CC.jpg
P89_R_DM_MLO.jpg
P8_L_DM_MLO.jpg
P90_L_DM_CC.jpg
P90_L_DM_MLO.jpg
P90_R_DM_CC.jpg
P90_R_DM_MLO.jpg
P91_L_DM_MLO.jpg
P92_L_DM_CC.jpg
P92_L_DM_MLO.jpg
P92_R_DM_CC.jpg
P92_R_DM_MLO.jpg
P93_L_DM_CC.jpg
P93_L_DM_MLO.jpg
P93_R_DM_CC.jpg
P93_R_DM_MLO.jpg
P94_L_DM_CC.jpg
P94_L_DM_MLO.jpg
P94_R_DM_CC.jpg
P94_R_DM_MLO.jpg
P95_L_DM_CC.jpg
P95_R_DM_MLO.jpg
P96_L_DM_CC.jpg
P96_L_DM_MLO.jpg
P96_R_DM_CC.jpg
P96_R_DM_MLO.jpg
P97_L_DM_CC.jpg
P97_L_DM_MLO.jpg
P97_R_DM_CC.jpg
P97_R_DM_MLO.jpg
P98_L_DM_CC.jpg
P98_L_DM_MLO.jpg
P98_R_DM_CC.jpg
P98_R_DM_MLO.jpg
P99_L_DM_CC.jpg
P99_L_DM_MLO.jpg
P99_L_DM_MLO_2.jpg
P99_R_DM_CC.jpg
P99_R_DM_MLO.jpg
P9_L_DM_MLO.jpg
EOM

for IMAGE in ${IMAGES}; do
  if [ ! -f ${LOCAL_DIRS[0]}/${IMAGE} ]; then
    wget ${AWS_BUCKET_URL}/${REMOTE_DIRS[0]}/${IMAGE} -O ${IMAGE}
    mv ${IMAGE} ${LOCAL_DIRS[0]}/${IMAGE}
  fi
done

read -r -d '' IMAGES << EOM
P100_L_CM_CC.jpg
P100_L_CM_MLO.jpg
P100_R_CM_CC.jpg
P100_R_CM_MLO.jpg
P101_L_CM_CC.jpg
P101_L_CM_MLO.jpg
P101_R_CM_CC.jpg
P101_R_CM_MLO.jpg
P102_L_CM_CC.jpg
P102_L_CM_MLO.jpg
P102_R_CM_CC.jpg
P102_R_CM_MLO.jpg
P103_L_CM_CC.jpg
P103_L_CM_MLO.jpg
P103_R_CM_CC.jpg
P103_R_CM_MLO.jpg
P104_L_CM_MLO.jpg
P104_R_CM_CC.jpg
P105_L_CM_CC.jpg
P105_L_CM_MLO.jpg
P105_R_CM_CC.jpg
P105_R_CM_MLO.jpg
P106_L_CM_MLO.jpg
P106_R_CM_CC.jpg
P107_L_CM_CC.jpg
P107_R_CM_CC.jpg
P107_R_CM_MLO.jpg
P108_L_CM_CC.jpg
P108_R_CM_CC.jpg
P109_L_CM_CC.jpg
P10_L_CM_CC.jpg
P10_L_CM_MLO.jpg
P10_R_CM_CC.jpg
P10_R_CM_MLO.jpg
P110_L_CM_MLO.jpg
P110_R_CM_CC.jpg
P110_R_CM_MLO.jpg
P111_L_CM_CC.jpg
P111_L_CM_MLO.jpg
P111_R_CM_CC.jpg
P111_R_CM_MLO.jpg
P112_L_CM_CC.jpg
P113_L_CM_CC.jpg
P113_L_CM_MLO.jpg
P113_R_CM_MLO.jpg
P114_L_CM_CC.jpg
P114_L_CM_MLO.jpg
P114_R_CM_CC.jpg
P114_R_CM_MLO.jpg
P115_L_CM_CC.jpg
P115_L_CM_MLO.jpg
P115_R_CM_CC.jpg
P115_R_CM_MLO.jpg
P116_L_CM_MLO.jpg
P116_R_CM_CC.jpg
P116_R_CM_MLO.jpg
P117_R_CM_CC.jpg
P118_L_CM_CC.jpg
P118_R_CM_CC.jpg
P118_R_CM_MLO.jpg
P119_L_CM_CC.jpg
P119_L_CM_MLO.jpg
P119_R_CM_CC.jpg
P119_R_CM_MLO.jpg
P11_R_CM_CC.jpg
P11_R_CM_MLO.jpg
P120_L_CM_MLO.jpg
P121_L_CM_CC.jpg
P121_R_CM_MLO.jpg
P122_L_CM_CC.jpg
P122_L_CM_MLO.jpg
P122_R_CM_CC.jpg
P122_R_CM_MLO.jpg
P123_L_CM_CC.jpg
P123_L_CM_MLO.jpg
P123_R_CM_CC.jpg
P123_R_CM_MLO.jpg
P124_L_CM_CC.jpg
P124_L_CM_MLO.jpg
P124_R_CM_CC.jpg
P124_R_CM_MLO.jpg
P125_L_CM_CC.jpg
P125_L_CM_MLO.jpg
P125_R_CM_CC.jpg
P125_R_CM_MLO.jpg
P126_L_CM_CC.jpg
P126_L_CM_MLO.jpg
P126_R_CM_CC.jpg
P126_R_CM_MLO.jpg
P127_L_CM_MLO.jpg
P128_L_CM_CC.jpg
P128_L_CM_MLO.jpg
P128_R_CM_CC.jpg
P128_R_CM_MLO.jpg
P129_L_CM_CC.jpg
P129_L_CM_MLO.jpg
P129_R_CM_CC.jpg
P129_R_CM_MLO.jpg
P12_L_CM_CC.jpg
P12_L_CM_MLO.jpg
P12_R_CM_CC.jpg
P12_R_CM_MLO.jpg
P130_L_CM_CC.jpg
P131_L_CM_MLO.jpg
P132_L_CM_CC.jpg
P132_L_CM_MLO.jpg
P132_R_CM_CC.jpg
P132_R_CM_MLO.jpg
P133_R_CM_CC.jpg
P134_R_CM_MLO.jpg
P135_L_CM_MLO.jpg
P136_L_CM_MLO.jpg
P136_R_CM_CC.jpg
P136_R_CM_MLO.jpg
P137_L_CM_MLO.jpg
P138_L_CM_CC.jpg
P138_L_CM_MLO.jpg
P138_R_CM_CC.jpg
P138_R_CM_MLO.jpg
P139_R_CM_CC.jpg
P139_R_CM_MLO.jpg
P13_R_CM_CC.jpg
P140_L_CM_CC.jpg
P140_L_CM_MLO.jpg
P140_R_CM_CC.jpg
P140_R_CM_MLO.jpg
P141_L_CM_CC.jpg
P141_L_CM_MLO.jpg
P141_R_CM_CC.jpg
P141_R_CM_MLO.jpg
P142_R_CM_CC.jpg
P142_R_CM_MLO.jpg
P143_L_CM_CC.jpg
P143_L_CM_MLO.jpg
P143_R_CM_CC.jpg
P143_R_CM_MLO.jpg
P144_L_CM_CC.jpg
P144_L_CM_MLO.jpg
P144_R_CM_CC.jpg
P144_R_CM_MLO.jpg
P145_L_CM_CC.jpg
P145_L_CM_MLO.jpg
P145_R_CM_CC.jpg
P145_R_CM_MLO.jpg
P146_L_CM_CC.jpg
P146_L_CM_MLO.jpg
P146_R_CM_CC.jpg
P146_R_CM_MLO.jpg
P147_R_CM_CC.jpg
P147_R_CM_MLO.jpg
P148_L_CM_CC.jpg
P148_L_CM_MLO.jpg
P148_R_CM_CC.jpg
P148_R_CM_MLO.jpg
P149_L_CM_CC.jpg
P149_L_CM_MLO.jpg
P149_R_CM_CC.jpg
P149_R_CM_MLO.jpg
P14_L_CM_CC.jpg
P14_L_CM_MLO.jpg
P14_R_CM_CC.jpg
P14_R_CM_MLO.jpg
P150_L_CM_CC.jpg
P150_L_CM_MLO.jpg
P150_R_CM_CC.jpg
P150_R_CM_MLO.jpg
P151_L_CM_CC.jpg
P152_L_CM_CC.jpg
P152_L_CM_MLO.jpg
P152_R_CM_CC.jpg
P153_L_CM_CC.jpg
P153_L_CM_MLO.jpg
P153_R_CM_MLO.jpg
P154_L_CM_CC.jpg
P154_L_CM_MLO.jpg
P154_R_CM_CC.jpg
P154_R_CM_MLO.jpg
P155_L_CM_CC.jpg
P155_L_CM_MLO.jpg
P155_R_CM_CC.jpg
P155_R_CM_MLO.jpg
P156_L_CM_CC.jpg
P156_L_CM_MLO.jpg
P156_R_CM_CC.jpg
P156_R_CM_MLO.jpg
P157_R_CM_CC.jpg
P158_L_CM_MLO.jpg
P159_L_CM_CC.jpg
P15_L_CM_MLO.jpg
P15_R_CM_MLO.jpg
P160_L_CM_CC.jpg
P160_R_CM_CC.jpg
P161_L_CM_CC.jpg
P161_L_CM_MLO.jpg
P161_R_CM_CC.jpg
P161_R_CM_MLO.jpg
P162_L_CM_CC.jpg
P162_L_CM_MLO.jpg
P162_R_CM_CC.jpg
P162_R_CM_MLO.jpg
P163_R_CM_CC.jpg
P164_L_CM_CC.jpg
P165_L_CM_CC.jpg
P165_L_CM_MLO.jpg
P165_R_CM_CC.jpg
P165_R_CM_MLO.jpg
P166_L_CM_CC.jpg
P167_L_CM_MLO.jpg
P168_L_CM_MLO.jpg
P169_L_CM_CC.jpg
P169_L_CM_MLO.jpg
P169_R_CM_CC.jpg
P16_L_CM_CC.jpg
P16_L_CM_MLO.jpg
P16_R_CM_CC.jpg
P16_R_CM_MLO.jpg
P170_L_CM_CC.jpg
P170_L_CM_MLO.jpg
P170_R_CM_CC.jpg
P170_R_CM_MLO.jpg
P171_L_CM_MLO.jpg
P172_L_CM_CC.jpg
P172_L_CM_MLO.jpg
P172_R_CM_CC.jpg
P172_R_CM_MLO.jpg
P173_L_CM_CC.jpg
P173_L_CM_MLO.jpg
P173_R_CM_CC.jpg
P173_R_CM_MLO.jpg
P174_L_CM_CC.jpg
P174_L_CM_MLO.jpg
P174_R_CM_CC.jpg
P174_R_CM_MLO.jpg
P175_R_CM_CC.jpg
P175_R_CM_MLO.jpg
P176_L_CM_CC.jpg
P176_L_CM_MLO.jpg
P176_R_CM_CC.jpg
P176_R_CM_MLO.jpg
P177_L_CM_CC.jpg
P177_L_CM_MLO.jpg
P177_R_CM_CC.jpg
P177_R_CM_MLO.jpg
P178_R_CM_MLO.jpg
P179_L_CM_CC.jpg
P179_L_CM_MLO.jpg
P17_R_CM_MLO.jpg
P180_L_CM_CC.jpg
P180_L_CM_MLO.jpg
P180_R_CM_CC.jpg
P180_R_CM_MLO.jpg
P181_L_CM_CC.jpg
P181_L_CM_MLO.jpg
P181_R_CM_CC.jpg
P181_R_CM_MLO.jpg
P182_R_CM_MLO.jpg
P183_L_CM_CC.jpg
P183_L_CM_MLO.jpg
P183_R_CM_CC.jpg
P183_R_CM_MLO.jpg
P184_L_CM_CC.jpg
P184_L_CM_MLO.jpg
P184_R_CM_CC.jpg
P184_R_CM_MLO.jpg
P185_L_CM_CC.jpg
P185_L_CM_MLO.jpg
P185_R_CM_CC.jpg
P185_R_CM_MLO.jpg
P186_L_CM_CC.jpg
P186_L_CM_MLO.jpg
P186_R_CM_CC.jpg
P187_L_CM_CC.jpg
P187_R_CM_CC.jpg
P188_L_CM_CC.jpg
P188_L_CM_MLO.jpg
P188_R_CM_CC.jpg
P188_R_CM_MLO.jpg
P189_L_CM_CC.jpg
P189_R_CM_MLO.jpg
P18_L_CM_CC.jpg
P18_L_CM_MLO.jpg
P18_R_CM_CC.jpg
P18_R_CM_MLO.jpg
P190_L_CM_CC.jpg
P190_L_CM_MLO.jpg
P190_R_CM_MLO.jpg
P191_L_CM_CC.jpg
P191_L_CM_MLO.jpg
P191_R_CM_CC.jpg
P191_R_CM_MLO.jpg
P192_L_CM_CC.jpg
P192_L_CM_MLO.jpg
P192_R_CM_CC.jpg
P192_R_CM_MLO.jpg
P193_L_CM_MLO.jpg
P194_L_CM_CC.jpg
P194_L_CM_MLO.jpg
P194_R_CM_CC.jpg
P194_R_CM_MLO.jpg
P195_L_CM_CC.jpg
P195_L_CM_MLO.jpg
P195_R_CM_CC.jpg
P195_R_CM_MLO.jpg
P196_L_CM_CC.jpg
P196_L_CM_MLO.jpg
P196_R_CM_MLO.jpg
P197_L_CM_CC.jpg
P197_L_CM_MLO.jpg
P197_R_CM_CC.jpg
P197_R_CM_MLO.jpg
P198_L_CM_CC.jpg
P198_L_CM_MLO.jpg
P199_R_CM_CC.jpg
P19_L_CM_CC.jpg
P19_L_CM_MLO.jpg
P19_R_CM_MLO.jpg
P1_L_CM_MLO.jpg
P200_L_CM_CC.jpg
P200_L_CM_MLO.jpg
P200_R_CM_CC.jpg
P200_R_CM_MLO.jpg
P201_R_CM_MLO.jpg
P202_L_CM_CC.jpg
P202_L_CM_MLO.jpg
P202_R_CM_CC.jpg
P202_R_CM_MLO.jpg
P203_R_CM_CC.jpg
P203_R_CM_MLO.jpg
P204_L_CM_CC.jpg
P205_L_CM_MLO.jpg
P206_L_CM_CC.jpg
P206_L_CM_MLO.jpg
P206_R_CM_CC.jpg
P206_R_CM_MLO.jpg
P207_L_CM_CC.jpg
P207_L_CM_MLO.jpg
P207_R_CM_CC.jpg
P207_R_CM_MLO.jpg
P208_R_CM_CC.jpg
P208_R_CM_MLO.jpg
P209_L_CM_CC.jpg
P209_L_CM_MLO.jpg
P209_R_CM_CC.jpg
P209_R_CM_MLO.jpg
P20_R_CM_CC.jpg
P20_R_CM_MLO.jpg
P210_L_CM_CC.jpg
P210_L_CM_MLO.jpg
P211_L_CM_CC.jpg
P211_L_CM_MLO.jpg
P211_R_CM_CC.jpg
P211_R_CM_MLO.jpg
P212_L_CM_CC.jpg
P212_L_CM_MLO.jpg
P212_R_CM_CC.jpg
P212_R_CM_MLO.jpg
P213_L_CM_CC.jpg
P213_L_CM_MLO.jpg
P213_R_CM_CC.jpg
P213_R_CM_MLO.jpg
P214_L_CM_CC.jpg
P214_L_CM_MLO.jpg
P214_R_CM_CC.jpg
P214_R_CM_MLO.jpg
P215_L_CM_CC.jpg
P215_R_CM_CC.jpg
P215_R_CM_MLO.jpg
P216_L_CM_CC.jpg
P216_L_CM_MLO.jpg
P216_R_CM_CC.jpg
P216_R_CM_MLO.jpg
P217_L_CM_CC.jpg
P217_L_CM_MLO.jpg
P218_L_CM_MLO.jpg
P218_R_CM_CC.jpg
P219_L_CM_MLO.jpg
P21_L_CM_CC.jpg
P21_L_CM_MLO.jpg
P21_R_CM_CC.jpg
P21_R_CM_MLO.jpg
P220_L_CM_CC.jpg
P220_L_CM_MLO.jpg
P220_R_CM_CC.jpg
P220_R_CM_MLO.jpg
P221_L_CM_CC.jpg
P222_L_CM_CC.jpg
P223_L_CM_CC.jpg
P223_L_CM_MLO.jpg
P223_R_CM_MLO.jpg
P224_L_CM_CC.jpg
P225_L_CM_CC.jpg
P225_L_CM_MLO.jpg
P225_R_CM_CC.jpg
P225_R_CM_MLO.jpg
P226_L_CM_CC.jpg
P226_L_CM_MLO.jpg
P226_R_CM_CC.jpg
P226_R_CM_MLO.jpg
P227_L_CM_CC.jpg
P227_L_CM_MLO.jpg
P227_R_CM_CC.jpg
P227_R_CM_MLO.jpg
P228_L_CM_CC.jpg
P228_L_CM_MLO.jpg
P228_R_CM_CC.jpg
P228_R_CM_MLO.jpg
P229_R_CM_MLO.jpg
P22_L_CM_CC.jpg
P230_L_CM_CC.jpg
P230_L_CM_MLO.jpg
P230_R_CM_CC.jpg
P230_R_CM_MLO.jpg
P231_R_CM_MLO.jpg
P232_L_CM_CC.jpg
P232_L_CM_MLO.jpg
P232_R_CM_CC.jpg
P232_R_CM_MLO.jpg
P233_L_CM_CC.jpg
P233_L_CM_MLO.jpg
P233_R_CM_CC.jpg
P233_R_CM_MLO.jpg
P234_L_CM_CC.jpg
P235_L_CM_CC.jpg
P235_L_CM_MLO.jpg
P235_R_CM_CC.jpg
P235_R_CM_MLO.jpg
P236_L_CM_CC.jpg
P236_L_CM_MLO.jpg
P236_R_CM_CC.jpg
P236_R_CM_MLO.jpg
P237_L_CM_CC.jpg
P237_L_CM_MLO.jpg
P237_R_CM_CC.jpg
P237_R_CM_MLO.jpg
P238_L_CM_CC.jpg
P238_L_CM_MLO.jpg
P238_R_CM_CC.jpg
P238_R_CM_MLO.jpg
P239_L_CM_CC.jpg
P239_L_CM_MLO.jpg
P239_R_CM_CC.jpg
P239_R_CM_MLO.jpg
P23_R_CM_CC.jpg
P240_R_CM_MLO.jpg
P241_L_CM_CC.jpg
P241_R_CM_CC.jpg
P242_L_CM_CC.jpg
P242_R_CM_MLO.jpg
P243_L_CM_CC.jpg
P243_L_CM_MLO.jpg
P243_R_CM_CC.jpg
P243_R_CM_MLO.jpg
P244_L_CM_CC.jpg
P244_L_CM_MLO.jpg
P244_R_CM_CC.jpg
P244_R_CM_MLO.jpg
P245_L_CM_CC.jpg
P245_R_CM_CC.jpg
P246_R_CM_MLO.jpg
P247_L_CM_MLO.jpg
P248_L_CM_CC.jpg
P248_L_CM_MLO.jpg
P248_R_CM_MLO.jpg
P249_L_CM_CC.jpg
P249_L_CM_MLO.jpg
P249_R_CM_CC.jpg
P249_R_CM_MLO.jpg
P24_L_CM_CC.jpg
P24_L_CM_MLO.jpg
P24_R_CM_CC.jpg
P24_R_CM_MLO.jpg
P250_L_CM_CC.jpg
P251_L_CM_CC.jpg
P251_L_CM_MLO.jpg
P251_R_CM_CC.jpg
P251_R_CM_MLO.jpg
P252_L_CM_CC.jpg
P252_L_CM_MLO.jpg
P252_R_CM_CC.jpg
P252_R_CM_MLO.jpg
P253_L_CM_CC.jpg
P253_L_CM_MLO.jpg
P253_R_CM_CC.jpg
P253_R_CM_MLO.jpg
P254_L_CM_CC.jpg
P254_L_CM_MLO.jpg
P254_R_CM_CC.jpg
P254_R_CM_MLO.jpg
P255_L_CM_CC.jpg
P255_L_CM_MLO.jpg
P255_R_CM_CC.jpg
P255_R_CM_MLO.jpg
P256_L_CM_CC.jpg
P256_L_CM_MLO.jpg
P257_L_CM_CC.jpg
P257_L_CM_MLO.jpg
P257_R_CM_CC.jpg
P257_R_CM_MLO.jpg
P258_L_CM_CC.jpg
P258_L_CM_MLO.jpg
P258_R_CM_CC.jpg
P258_R_CM_MLO.jpg
P259_L_CM_CC.jpg
P259_L_CM_MLO.jpg
P259_R_CM_CC.jpg
P259_R_CM_MLO.jpg
P25_L_CM_MLO.jpg
P25_R_CM_CC.jpg
P260_L_CM_CC.jpg
P260_L_CM_MLO.jpg
P260_R_CM_CC.jpg
P260_R_CM_MLO.jpg
P261_L_CM_CC.jpg
P261_L_CM_MLO.jpg
P261_R_CM_CC.jpg
P261_R_CM_MLO.jpg
P262_L_CM_CC.jpg
P262_L_CM_MLO.jpg
P262_R_CM_CC.jpg
P262_R_CM_MLO.jpg
P263_L_CM_CC.jpg
P263_L_CM_MLO.jpg
P263_R_CM_CC.jpg
P263_R_CM_MLO.jpg
P264_L_CM_CC.jpg
P264_L_CM_MLO.jpg
P264_R_CM_CC.jpg
P264_R_CM_MLO.jpg
P265_L_CM_CC.jpg
P265_L_CM_MLO.jpg
P265_R_CM_CC.jpg
P265_R_CM_MLO.jpg
P266_L_CM_CC.jpg
P266_L_CM_MLO.jpg
P266_R_CM_CC.jpg
P266_R_CM_MLO.jpg
P267_L_CM_CC.jpg
P267_L_CM_MLO.jpg
P267_R_CM_CC.jpg
P267_R_CM_MLO.jpg
P268_L_CM_CC.jpg
P268_L_CM_MLO.jpg
P268_R_CM_CC.jpg
P268_R_CM_MLO.jpg
P269_L_CM_CC.jpg
P269_L_CM_MLO.jpg
P269_R_CM_CC.jpg
P269_R_CM_MLO.jpg
P26_L_CM_CC.jpg
P26_L_CM_MLO.jpg
P270_L_CM_CC.jpg
P270_L_CM_MLO.jpg
P270_R_CM_CC.jpg
P270_R_CM_MLO.jpg
P271_L_CM_CC.jpg
P271_L_CM_MLO.jpg
P271_R_CM_CC.jpg
P271_R_CM_MLO.jpg
P272_L_CM_CC.jpg
P272_L_CM_MLO.jpg
P272_R_CM_CC.jpg
P272_R_CM_MLO.jpg
P273_L_CM_CC.jpg
P273_L_CM_MLO.jpg
P273_R_CM_CC.jpg
P273_R_CM_MLO.jpg
P274_L_CM_CC.jpg
P274_L_CM_MLO.jpg
P274_R_CM_CC.jpg
P274_R_CM_MLO.jpg
P275_L_CM_CC.jpg
P275_L_CM_MLO.jpg
P275_R_CM_CC.jpg
P275_R_CM_MLO.jpg
P276_L_CM_CC.jpg
P276_L_CM_MLO.jpg
P276_R_CM_CC.jpg
P276_R_CM_MLO.jpg
P277_L_CM_CC.jpg
P277_L_CM_MLO.jpg
P277_R_CM_CC.jpg
P277_R_CM_MLO.jpg
P278_L_CM_CC.jpg
P278_L_CM_MLO.jpg
P278_R_CM_CC.jpg
P278_R_CM_MLO.jpg
P279_L_CM_CC.jpg
P279_L_CM_MLO.jpg
P279_R_CM_CC.jpg
P279_R_CM_MLO.jpg
P27_L_CM_CC.jpg
P27_L_CM_MLO.jpg
P280_L_CM_CC.jpg
P280_L_CM_MLO.jpg
P280_R_CM_CC.jpg
P280_R_CM_MLO.jpg
P281_L_CM_CC.jpg
P281_L_CM_MLO.jpg
P281_L_CM_MLO2.jpg
P281_R_CM_CC.jpg
P281_R_CM_MLO.jpg
P282_L_CM_CC.jpg
P282_L_CM_MLO.jpg
P282_R_CM_CC.jpg
P282_R_CM_MLO.jpg
P283_L_CM_CC.jpg
P283_L_CM_MLO.jpg
P283_R_CM_CC.jpg
P283_R_CM_MLO.jpg
P284_L_CM_CC.jpg
P284_L_CM_MLO.jpg
P284_R_CM_CC.jpg
P284_R_CM_MLO.jpg
P285_L_CM_CC.jpg
P285_L_CM_MLO.jpg
P285_R_CM_CC.jpg
P285_R_CM_MLO.jpg
P286_L_CM_CC.jpg
P286_L_CM_MLO.jpg
P286_R_CM_CC.jpg
P286_R_CM_MLO.jpg
P287_L_CM_CC.jpg
P287_L_CM_MLO.jpg
P287_R_CM_CC.jpg
P287_R_CM_MLO.jpg
P288_L_CM_CC.jpg
P288_L_CM_MLO.jpg
P288_R_CM_CC.jpg
P288_R_CM_MLO.jpg
P289_L_CM_CC.jpg
P289_L_CM_MLO.jpg
P289_R_CM_CC.jpg
P289_R_CM_MLO.jpg
P28_L_CM_MLO.jpg
P28_R_CM_CC.jpg
P28_R_CM_MLO.jpg
P290_L_CM_CC.jpg
P290_L_CM_MLO.jpg
P290_R_CM_CC.jpg
P290_R_CM_MLO.jpg
P291_L_CM_CC.jpg
P291_L_CM_MLO.jpg
P291_R_CM_CC.jpg
P291_R_CM_MLO.jpg
P292_L_CM_CC.jpg
P292_L_CM_MLO.jpg
P292_R_CM_CC.jpg
P292_R_CM_MLO.jpg
P293_L_CM_CC.jpg
P293_L_CM_MLO.jpg
P293_R_CM_CC.jpg
P293_R_CM_MLO.jpg
P294_R_CM_CC.jpg
P294_R_CM_MLO.jpg
P295_L_CM_CC.jpg
P295_L_CM_MLO.jpg
P295_R_CM_CC.jpg
P295_R_CM_MLO.jpg
P296_L_CM_CC.jpg
P296_L_CM_MLO.jpg
P296_R_CM_CC.jpg
P296_R_CM_MLO.jpg
P297_L_CM_CC.jpg
P297_L_CM_MLO.jpg
P297_R_CM_CC.jpg
P297_R_CM_MLO.jpg
P298_L_CM_CC.jpg
P298_L_CM_MLO.jpg
P298_R_CM_CC.jpg
P298_R_CM_MLO.jpg
P299_L_CM_CC.jpg
P299_L_CM_MLO.jpg
P299_R_CM_CC.jpg
P299_R_CM_MLO.jpg
P29_L_CM_CC.jpg
P29_R_CM_CC.jpg
P29_R_CM_MLO.jpg
P2_L_CM_CC.jpg
P2_L_CM_MLO.jpg
P2_R_CM_CC.jpg
P2_R_CM_MLO.jpg
P300_L_CM_CC.jpg
P300_L_CM_MLO.jpg
P300_R_CM_CC.jpg
P300_R_CM_MLO.jpg
P301_L_CM_CC.jpg
P301_L_CM_MLO.jpg
P301_R_CM_CC.jpg
P301_R_CM_MLO.jpg
P302_L_CM_CC.jpg
P302_L_CM_MLO.jpg
P302_R_CM_CC.jpg
P302_R_CM_MLO.jpg
P303_L_CM_CC.jpg
P303_L_CM_MLO.jpg
P303_R_CM_CC.jpg
P303_R_CM_MLO.jpg
P304_L_CM_CC.jpg
P304_L_CM_MLO.jpg
P304_R_CM_CC.jpg
P304_R_CM_MLO.jpg
P305_L_CM_CC.jpg
P305_L_CM_MLO.jpg
P305_R_CM_CC.jpg
P305_R_CM_MLO.jpg
P306_L_CM_CC.jpg
P306_L_CM_MLO.jpg
P306_R_CM_CC.jpg
P306_R_CM_MLO.jpg
P307_L_CM_CC.jpg
P307_L_CM_MLO.jpg
P307_R_CM_CC.jpg
P307_R_CM_MLO.jpg
P308_L_CM_CC.jpg
P308_L_CM_MLO.jpg
P308_R_CM_CC.jpg
P308_R_CM_MLO.jpg
P309_L_CM_CC.jpg
P309_L_CM_MLO.jpg
P309_R_CM_CC.jpg
P309_R_CM_MLO.jpg
P30_L_CM_CC.jpg
P30_L_CM_MLO.jpg
P30_R_CM_CC.jpg
P30_R_CM_MLO.jpg
P310_L_CM_CC.jpg
P310_L_CM_MLO.jpg
P310_R_CM_CC.jpg
P310_R_CM_MLO.jpg
P311_L_CM_CC.jpg
P311_L_CM_MLO.jpg
P311_R_CM_CC.jpg
P311_R_CM_MLO.jpg
P312_L_CM_CC.jpg
P312_L_CM_MLO.jpg
P312_R_CM_CC.jpg
P312_R_CM_MLO.jpg
P313_L_CM_CC.jpg
P313_L_CM_MLO.jpg
P313_R_CM_CC.jpg
P313_R_CM_MLO.jpg
P314_L_CM_CC.jpg
P314_L_CM_MLO.jpg
P314_R_CM_CC.jpg
P314_R_CM_MLO.jpg
P315_L_CM_CC.jpg
P315_L_CM_MLO.jpg
P315_R_CM_CC.jpg
P315_R_CM_MLO.jpg
P316_L_CM_CC.jpg
P316_L_CM_MLO.jpg
P316_R_CM_CC.jpg
P316_R_CM_MLO.jpg
P317_L_CM_CC.jpg
P317_L_CM_MLO.jpg
P317_R_CM_CC.jpg
P317_R_CM_MLO.jpg
P318_L_CM_CC.jpg
P318_L_CM_MLO.jpg
P318_R_CM_CC.jpg
P318_R_CM_MLO.jpg
P319_L_CM_CC.jpg
P319_L_CM_MLO.jpg
P319_R_CM_CC.jpg
P319_R_CM_MLO.jpg
P31_L_CM_CC.jpg
P31_L_CM_MLO.jpg
P31_R_CM_CC.jpg
P31_R_CM_MLO.jpg
P320_L_CM_CC.jpg
P320_L_CM_MLO.jpg
P320_R_CM_CC.jpg
P320_R_CM_MLO.jpg
P321_R_CM_CC.jpg
P321_R_CM_MLO.jpg
P322_L_CM_CC.jpg
P322_L_CM_MLO.jpg
P322_R_CM_CC.jpg
P322_R_CM_MLO.jpg
P323_L_CM_CC.jpg
P323_L_CM_MLO.jpg
P323_R_CM_CC.jpg
P323_R_CM_MLO.jpg
P324_L_CM_CC.jpg
P324_L_CM_MLO.jpg
P324_R_CM_MLO.jpg
P325_L_CM_CC.jpg
P325_L_CM_MLO.jpg
P325_R_CM_CC.jpg
P325_R_CM_MLO.jpg
P326_L_CM_CC.jpg
P326_L_CM_MLO.jpg
P326_R_CM_CC.jpg
P326_R_CM_MLO.jpg
P32_L_CM_CC.jpg
P32_L_CM_MLO.jpg
P32_R_CM_CC.jpg
P32_R_CM_MLO.jpg
P33_L_CM_CC.jpg
P33_L_CM_MLO.jpg
P34_L_CM_CC.jpg
P34_R_CM_MLO.jpg
P35_L_CM_CC.jpg
P35_L_CM_MLO.jpg
P35_R_CM_MLO.jpg
P36_R_CM_MLO.jpg
P37_L_CM_CC.jpg
P37_L_CM_MLO.jpg
P37_R_CM_CC.jpg
P37_R_CM_MLO.jpg
P38_R_CM_CC.jpg
P38_R_CM_MLO.jpg
P39_R_CM_CC.jpg
P39_R_CM_MLO.jpg
P3_L_CM_CC.jpg
P3_L_CM_MLO.jpg
P3_R_CM_CC.jpg
P3_R_CM_MLO.jpg
P40_L_CM_CC.jpg
P40_L_CM_MLO.jpg
P40_R_CM_CC.jpg
P40_R_CM_MLO.jpg
P41_L_CM_CC.jpg
P42_R_CM_MLO.jpg
P43_R_CM_CC.jpg
P43_R_CM_MLO.jpg
P44_L_CM_CC.jpg
P44_L_CM_MLO.jpg
P44_R_CM_CC.jpg
P45_L_CM_CC.jpg
P45_L_CM_MLO.jpg
P45_R_CM_CC.jpg
P45_R_CM_MLO.jpg
P46_L_CM_CC.jpg
P46_L_CM_MLO.jpg
P46_R_CM_CC.jpg
P46_R_CM_MLO.jpg
P47_R_CM_CC.jpg
P48_L_CM_CC.jpg
P49_L_CM_CC.jpg
P49_L_CM_MLO.jpg
P49_R_CM_CC.jpg
P49_R_CM_MLO.jpg
P4_L_CM_MLO.jpg
P4_R_CM_CC.jpg
P50_L_CM_CC.jpg
P50_R_CM_MLO.jpg
P51_L_CM_CC.jpg
P51_L_CM_MLO.jpg
P51_R_CM_CC.jpg
P51_R_CM_MLO.jpg
P52_L_CM_CC.jpg
P52_L_CM_MLO.jpg
P52_R_CM_CC.jpg
P52_R_CM_MLO.jpg
P53_R_CM_CC.jpg
P54_L_CM_CC.jpg
P54_L_CM_MLO.jpg
P54_R_CM_CC.jpg
P54_R_CM_MLO.jpg
P55_L_CM_CC.jpg
P55_L_CM_MLO.jpg
P55_R_CM_CC.jpg
P55_R_CM_MLO.jpg
P56_L_CM_MLO.jpg
P56_R_CM_CC.jpg
P56_R_CM_MLO.jpg
P57_R_CM_CC.jpg
P58_L_CM_MLO.jpg
P59_L_CM_CC.jpg
P59_L_CM_MLO.jpg
P59_R_CM_CC.jpg
P59_R_CM_MLO.jpg
P5_L_CM_CC.jpg
P5_L_CM_MLO.jpg
P5_R_CM_CC.jpg
P5_R_CM_MLO.jpg
P60_L_CM_MLO.jpg
P61_L_CM_CC.jpg
P61_L_CM_MLO.jpg
P61_R_CM_CC.jpg
P61_R_CM_MLO.jpg
P62_L_CM_CC.jpg
P62_L_CM_MLO.jpg
P62_R_CM_CC.jpg
P62_R_CM_MLO.jpg
P63_R_CM_CC.jpg
P64_L_CM_CC.jpg
P64_L_CM_MLO.jpg
P64_R_CM_CC.jpg
P65_L_CM_CC.jpg
P65_L_CM_MLO.jpg
P65_R_CM_CC.jpg
P65_R_CM_MLO.jpg
P66_L_CM_CC.jpg
P66_L_CM_MLO.jpg
P66_R_CM_CC.jpg
P66_R_CM_MLO.jpg
P67_L_CM_CC.jpg
P67_R_CM_MLO.jpg
P68_L_CM_CC.jpg
P68_L_CM_MLO.jpg
P69_L_CM_CC.jpg
P69_L_CM_MLO.jpg
P6_L_CM_MLO.jpg
P6_R_CM_CC.jpg
P70_L_CM_CC.jpg
P71_L_CM_MLO.jpg
P72_L_CM_CC.jpg
P72_R_CM_CC.jpg
P73_L_CM_CC.jpg
P73_L_CM_MLO.jpg
P73_R_CM_CC.jpg
P73_R_CM_MLO.jpg
P74_L_CM_CC.jpg
P74_L_CM_MLO.jpg
P74_R_CM_CC.jpg
P74_R_CM_MLO.jpg
P75_R_CM_MLO.jpg
P76_L_CM_CC.jpg
P76_L_CM_MLO.jpg
P76_R_CM_CC.jpg
P76_R_CM_MLO.jpg
P77_L_CM_MLO.jpg
P77_R_CM_CC.jpg
P78_L_CM_CC.jpg
P79_L_CM_CC.jpg
P79_L_CM_MLO.jpg
P7_L_CM_MLO.jpg
P7_R_CM_CC.jpg
P7_R_CM_MLO.jpg
P80_L_CM_CC.jpg
P80_L_CM_MLO.jpg
P80_R_CM_MLO.jpg
P81_L_CM_CC.jpg
P81_L_CM_MLO.jpg
P81_R_CM_CC.jpg
P81_R_CM_MLO.jpg
P82_L_CM_CC.jpg
P82_L_CM_MLO.jpg
P82_R_CM_CC.jpg
P82_R_CM_MLO.jpg
P83_L_CM_CC.jpg
P83_L_CM_MLO.jpg
P83_R_CM_CC.jpg
P83_R_CM_MLO.jpg
P84_L_CM_CC.jpg
P84_L_CM_MLO.jpg
P84_R_CM_CC.jpg
P84_R_CM_MLO.jpg
P85_R_CM_MLO.jpg
P86_L_CM_CC.jpg
P86_R_CM_MLO.jpg
P87_L_CM_CC.jpg
P87_L_CM_MLO.jpg
P87_R_CM_CC.jpg
P87_R_CM_MLO.jpg
P88_L_CM_CC.jpg
P88_L_CM_MLO.jpg
P88_R_CM_CC.jpg
P88_R_CM_MLO.jpg
P89_L_CM_CC.jpg
P89_L_CM_MLO.jpg
P89_R_CM_CC.jpg
P89_R_CM_MLO.jpg
P8_L_CM_MLO.jpg
P90_L_CM_CC.jpg
P90_L_CM_MLO.jpg
P90_R_CM_CC.jpg
P90_R_CM_MLO.jpg
P91_L_CM_MLO.jpg
P92_L_CM_CC.jpg
P92_L_CM_MLO.jpg
P92_R_CM_CC.jpg
P92_R_CM_MLO.jpg
P93_L_CM_CC.jpg
P93_L_CM_MLO.jpg
P93_R_CM_CC.jpg
P93_R_CM_MLO.jpg
P94_L_CM_CC.jpg
P94_L_CM_MLO.jpg
P94_R_CM_CC.jpg
P94_R_CM_MLO.jpg
P95_L_CM_CC.jpg
P95_R_CM_MLO.jpg
P96_L_CM_CC.jpg
P96_L_CM_MLO.jpg
P96_R_CM_CC.jpg
P96_R_CM_MLO.jpg
P97_L_CM_CC.jpg
P97_L_CM_MLO.jpg
P97_R_CM_CC.jpg
P97_R_CM_MLO.jpg
P98_L_CM_CC.jpg
P98_L_CM_MLO.jpg
P98_R_CM_CC.jpg
P98_R_CM_MLO.jpg
P99_L_CM_CC.jpg
P99_L_CM_MLO.jpg
P99_L_CM_MLO_2.jpg
P99_R_CM_CC.jpg
P99_R_CM_MLO.jpg
P9_L_CM_MLO.jpg
EOM

for IMAGE in ${IMAGES}; do
  if [ ! -f ${LOCAL_DIRS[1]}/${IMAGE} ]; then
    wget ${AWS_BUCKET_URL}/${REMOTE_DIRS[1]}/${IMAGE} -O ${IMAGE}
    mv ${IMAGE} ${LOCAL_DIRS[1]}/${IMAGE}
  fi
done

set -xe
