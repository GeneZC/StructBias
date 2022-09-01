#!/bin/bash

python run.py \
    --hidden_size 1024 \
    --model_name mug \
    --mode evaluate \
    --train_data_name lasted \
    --test_data_name lasted \
    --pretrained_model_name_or_path plms/roberta-large-hfl \
    --suffix hfl_bert_large_adapter \
    --use_adapter \
    --batch_size 10