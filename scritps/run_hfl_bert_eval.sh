#!/bin/bash

python run.py \
    --model_name mug \
    --mode evaluate \
    --train_data_name lasted \
    --test_data_name lasted \
    --pretrained_model_name_or_path plms/roberta-base-hfl \
    --suffix hfl_bert \
    --batch_size 16
