# -*- coding: utf-8 -*-

import os
import random
import argparse

import numpy as np

import torch

from data_utils import build_data, build_tokenizer, build_embedding

from models import build_model_class

from trainer import Trainer
from evaluator import Evaluator
from predictor import Predictor

def set_dir(args):
    args.train_data_dir = os.path.join(args.data_dir, args.train_data_name)
    args.test_data_dir = os.path.join(args.data_dir, args.test_data_name)
    args.output_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(args.model_name, args.train_data_name, args.suffix))
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

def set_device(args):
    args.num_gpus = torch.cuda.device_count()
    args.device = torch.device('cuda' if args.num_gpus > 0 else 'cpu')

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def set_model(args):
    args.model_class, args.input_fields = build_model_class(args.model_name)

def print_args(args):
    print('>> training arguments:')
    for arg in vars(args):
        print('>> {}: {}'.format(arg, getattr(args, arg)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--suffix', default='none', type=str)
    parser.add_argument('--model_name', default='mug', type=str)
    parser.add_argument('--pretrained_model_name_or_path', default='plms/bert-base-chinese', type=str)
    parser.add_argument('--train_data_name', default='lasted', type=str)
    parser.add_argument('--test_data_name', default='lasted', type=str)
    parser.add_argument('--data_dir', default='datasets', type=str)
    parser.add_argument('--output_dir', default='outputs', type=str)
    parser.add_argument('--cache_dir', default='caches', type=str)
    parser.add_argument('--embed_learning_rate', default=5e-5, type=float)
    parser.add_argument('--embed_weight_decay', default=0.0, type=float)
    parser.add_argument('--learning_rate', default=5e-4, type=float) # 1e-3, 1e-4, 1e-5
    parser.add_argument('--weight_decay', default=0.0, type=float) # 0, 1e-2
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--sentiment_size', default=4, type=int)
    parser.add_argument('--tag_size', default=3, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_length', default=150, type=int)
    parser.add_argument('--num_repeats', default=1, type=int)
    parser.add_argument('--num_train_epochs', default=20, type=int)
    parser.add_argument('--num_patience_epochs', default=5, type=int)
    parser.add_argument('--warmup_ratio', default=0.1, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--log_interval', default=50, type=int)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--use_adapter', action='store_true')
    args = parser.parse_args()

    set_dir(args)
    set_device(args)
    set_seed(args)
    set_model(args)
    print_args(args)

    if args.mode == 'train':
        tokenizer = build_tokenizer(args.pretrained_model_name_or_path, cache_dir=args.cache_dir, use_fast=True)
        embedding = build_embedding(args.pretrained_model_name_or_path, cache_dir=args.cache_dir, use_adapter=args.use_adapter)
        data_dict = build_data(args.train_data_dir, tokenizer, max_length=args.max_length)

        trainer = Trainer(args)
        trainer.run(args, embedding, data_dict['train'], data_dict['dev'])
    elif args.mode == 'evaluate':
        tokenizer = build_tokenizer(args.pretrained_model_name_or_path, cache_dir=args.cache_dir, use_fast=True)
        embedding = build_embedding(args.pretrained_model_name_or_path, cache_dir=args.cache_dir, use_adapter=args.use_adapter)
        data_dict = build_data(args.test_data_dir, tokenizer, max_length=args.max_length)

        evaluator = Evaluator(args)
        evaluator.run(args, embedding, data_dict['test'])
    elif args.mode == 'predict':
        tokenizer = build_tokenizer(args.pretrained_model_name_or_path, cache_dir=args.cache_dir, use_fast=True)
        embedding = build_embedding(args.pretrained_model_name_or_path, cache_dir=args.cache_dir, use_adapter=args.use_adapter)
            
        predictor = Predictor(args)
        predictor.run(args, embedding, tokenizer)
    else:
        raise ValueError('unknown mode.')
