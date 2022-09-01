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

from transformers import AdamW, get_linear_schedule_with_warmup

class AverageMeter:
    def __init__(self, buffer_size=100):
        self.buffer_size = buffer_size
        self._vals = []

    def avg(self):
        return np.mean(self._vals)

    def update(self, val):
        self._vals.append(val)
        if len(self._vals) >  self.buffer_size:
            self._vals.pop(0)
        

class Trainer:
    def __init__(self, args):
        self.logger = Logger(os.path.join(args.output_dir, 'train.log'))
    
    def _train(self, args, model, parameters, optimizer, scheduler, train_data_loader, dev_data_loader, save_path):
        best_dev_p, best_dev_r, best_dev_f1 = 0, 0, 0
        best_dev_epoch = 0
        iter_step = 0
        train_loss = AverageMeter()
        for epoch in range(args.num_train_epochs):
            self.logger('{:>^70}'.format('epoch ' + str(epoch + 1)))
            for batch in train_data_loader:
                iter_step += 1
                
                model.train()
                optimizer.zero_grad()

                inputs = [batch[field] for field in args.input_fields]

                loss = model(inputs)

                train_loss.update(loss.item())
                
                loss.backward()
                optimizer.step()
                if args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                if scheduler is not None:
                    scheduler.step()

                if iter_step % args.log_interval == 0:
                    dev_loss, dev_p, dev_r, dev_f1 = self._evaluate(args, model, dev_data_loader)
                    self.logger('iter step {}, train loss {:.4f}, dev loss {:.4f}, dev p {:.4f}, dev r {:.4f}, dev f1 {:.4f}'.
                        format(iter_step, train_loss.avg(), dev_loss, dev_p, dev_r, dev_f1))
                    if dev_p > best_dev_p:
                        best_dev_p = dev_p
                    if dev_r >  best_dev_r:
                        best_dev_r = dev_r
                    if dev_f1 > best_dev_f1:
                        self.logger('>> new best')
                        best_dev_epoch = epoch
                        best_dev_f1 = dev_f1
                        torch.save(model.state_dict(), save_path)
            
            if epoch - best_dev_epoch >= args.num_patience_epochs:
                self.logger('>> early stop')
                break

        return best_dev_p, best_dev_r, best_dev_f1

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

    def run(self, args, embedding, train_data, dev_data):
        self.logger('{:*^70}'.format('training on ' + args.train_data_name))

        for i in range(args.num_repeats):
            self.logger('{:#^70}'.format('repeat ' + str(i + 1)))

            data_collator = DataCollator(args.input_fields, args.device)

            train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
            dev_data_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

            model = args.model_class(args, copy.deepcopy(embedding)).to(args.device)
            
            temp_best_path = os.path.join(args.output_dir, 'best_ckpt_{}.pt'.format(i + 1))

            # no_decay = ['bias', 'LayerNorm.weight']
            # grouped_parameters = [
            #     {
            #         'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            #         'lr': args.learning_rate,
            #         'weight_decay': args.weight_decay,
            #     },
            #     {
            #         'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            #         'lr': args.learning_rate,
            #         'weight_decay': 0.0,
            #     },
            # ]
            
            grouped_parameters = [
                {
                    'params': [p for n, p in model.named_parameters() if 'embed.' in n],
                    'lr': args.embed_learning_rate,
                    'weight_decay': args.embed_weight_decay,
                },
                {
                    'params': [p for n, p in model.named_parameters() if 'embed.' not in n], 
                    'lr': args.learning_rate,
                    'weight_decay': args.weight_decay,
                },
            ]
            optimizer = AdamW(grouped_parameters)
            scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * args.num_train_epochs * len(train_data_loader)), args.num_train_epochs * len(train_data_loader))

            self._train(args, model, model.parameters(), optimizer, scheduler, train_data_loader, dev_data_loader, temp_best_path)
