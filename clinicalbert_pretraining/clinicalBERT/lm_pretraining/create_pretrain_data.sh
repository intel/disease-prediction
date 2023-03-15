#!/bin/bash

#modify this to bert or biobert folder containing a vocab.txt file
echo "Script executed from: ${PWD}"

BERT_BASE_DIR=${PWD}/../../pretrained_bert_tf/biobert_pretrain_output_all_notes_150000

#modify this to be the path to the tokenized data
DATA_DIR=${PWD}/../../outputs/tokenized_notes

# modify this to be your output directory path
OUTPUT_DIR=${PWD}/../../outputs/pretraining_data

#modify this to be the note type that you want to create pretraining data for - e.g. ecg, echo, radiology, physician, nursing, etc. 
# Note that you can also specify multiple input files & output files below
DATA_FILE=formatted_output

# Note that create_pretraining_data.py is unmodified from the script in the original BERT repo. 
# Refer to the BERT repo for the most up to date version of this code.

python create_pretraining_data.py \
  --input_file=$DATA_DIR/$DATA_FILE.txt \
  --output_file=$OUTPUT_DIR/$DATA_FILE.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=False \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

