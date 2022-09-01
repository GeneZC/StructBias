#!/bin/bash

python run.py \
    --model_name mug \
    --mode evaluate \
    --train_data_name lasted \
    --test_data_name lasted \
    --pretrained_model_name_or_path plms/bert-base-chinese \
    --suffix google_bert_adapter \
    --use_adapter \
    --batch_size 12
