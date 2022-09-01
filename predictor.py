# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from data_utils import idx2sentiment

class Predictor:
    def __init__(self, args):
        pass

    def run(self, args, embedding, tokenizer):
        print('{:*^70}'.format('prediction'))
        
        model = args.model_class(args, embedding).to(args.device)
        
        temp_best_path = os.path.join(args.output_dir, 'best_ckpt_1.pt')
        #if 'bert' in args.model_name:
        state_dict = torch.load(temp_best_path)
        #else:
        #    state_dict = torch.load(temp_best_path)
        #    state_dict.pop('embed.weight')
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        while True:
            with torch.no_grad():
                print('{:#^70}'.format('input your review'))
                review = input()
                text_tokens = [tokenizer.cls_token] + tokenizer.tokenize(review) + [tokenizer.sep_token]
                text_indices = tokenizer.convert_tokens_to_ids(text_tokens)
                text_mask = [1] * len(text_indices)
                target_indices = [0] * len(text_indices)
                opinion_indices = [0] * len(text_indices)
                sentiment_indices = np.zeros((len(text_indices), len(text_indices)), dtype=np.int64)
                instance = {
                    'text_indices': torch.tensor([text_indices]),
                    'text_mask': torch.tensor([text_mask]),
                    'target_indices': torch.tensor([target_indices]),
                    'opinion_indices': torch.tensor([opinion_indices]),
                    'sentiment_indices': torch.tensor([sentiment_indices]),
                }

                inputs = [instance[field].to(args.device) for field in args.input_fields]

                _, triplets = model.predict(inputs)

                print('{:#^70}'.format('preprocessed tokens'))
                print(''.join(text_tokens))
                print('{:#^70}'.format('predicted triplets'))
                for triplet in triplets[0]:
                    a_beg, a_end, o_beg, o_end, sent_idx = map(int, triplet.split('-'))
                    print('[{}, {}, {}]'.format(''.join(text_tokens[a_beg: a_end+1]), ''.join(text_tokens[o_beg: o_end+1]), idx2sentiment[sent_idx]))
