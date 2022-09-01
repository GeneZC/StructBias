#!/bin/bash

python run.py \
    --hidden_size 1024
    --model_name mug \
    --mode train \
    --train_data_name lasted \
    --pretrained_model_name_or_path plms/roberta-large-hfl \
    --suffix hfl_bert_large \
    --batch_size 16
