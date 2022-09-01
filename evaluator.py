# -*- coding: utf-8 -*-

import os
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import DataLoader, DataCollator
from metric import exact_match_f1, fuzzy_match_f1
from logger import Logger

class Evaluator:
    def __init__(self, args):
        self.logger = Logger(os.path.join(args.output_dir, '{}_eval.log'.format(args.test_data_name)))

    @staticmethod
    def _evaluate(args, model, data_loader):
        model.eval()
        loss_all, target_all, output_all = [], [], []
        with torch.no_grad():
            for batch in data_loader:
                inputs = [batch[field] for field in args.input_fields]
                loss, output = model.predict(inputs)
                target = batch['eval_triplets']
                loss_all.append(loss)
                target_all.extend(target)
                output_all.extend(output)

        loss = np.mean(loss_all)
        precison, recall, f1 = exact_match_f1(output_all, target_all)
        
        return loss, precison, recall, f1

    def run(self, args, embedding, test_data):
        self.logger('{:*^70}'.format('evaluation on ' + args.test_data_name))
        
        result_dict = {'p': [], 'r': [], 'f1': []}
        for i in range(args.num_repeats):
            self.logger('{:#^70}'.format('repeat ' + str(i + 1)))

            data_collator = DataCollator(args.input_fields, args.device)

            test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

            model = args.model_class(args, copy.deepcopy(embedding)).to(args.device)
            
            temp_best_path = os.path.join(args.output_dir, 'best_ckpt_{}.pt'.format(i + 1))
            #if 'bert' in args.model_name:
            state_dict = torch.load(temp_best_path)
            #else:
            #    state_dict = torch.load(temp_best_path)
            #    state_dict.pop('embed.weight')
            model.load_state_dict(state_dict, strict=False)

            _, test_p, test_r, test_f1 = self._evaluate(args, model, test_data_loader)
            self.logger('test p: {:.4f}, test r: {:.4f}, test f1: {:.4f}'.format(test_p, test_r, test_f1))
            result_dict['p'].append(test_p)
            result_dict['r'].append(test_r)
            result_dict['f1'].append(test_f1)
        self.logger('{:*^70}'.format('evaluation result'))
        self.logger('p: {:.4f}, r: {:.4f}, f1: {:.4f}'.format(
            np.mean(result_dict['p']), np.mean(result_dict['r']), np.mean(result_dict['f1'])))
