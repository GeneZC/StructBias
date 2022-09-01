#!/bin/bash

python run.py \
    --hidden_size 1024 \
    --model_name mug \
    --mode train \
    --train_data_name lasted \
    --pretrained_model_name_or_path plms/roberta-large-hfl \
    --suffix hfl_bert_large_adapter \
    --use_adapter \
    --batch_size 10 \
    --embed_learning_rate 1e-5 \
    --learning_rate 1e-4
