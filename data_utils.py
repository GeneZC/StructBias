# -*- coding: utf-8 -*-

import os

import numpy as np

from transformers import BertTokenizer, BertConfig, BertModel
from modules.modeling_bert_adapter import BertAdapterModel

class Tokenizer:
    def __init__(self, token2idx, pad_token='<pad>', unk_token='<unk>'):
        self.token2idx = token2idx
        self.idx2token = {v: k for k, v in token2idx.items()}
        self.pad_token = pad_token
        self.pad_token_idx = self.token2idx[pad_token]
        self.unk_token = unk_token
        self.unk_token_idx = self.token2idx[unk_token]

    @classmethod
    def from_corpus(cls, corpus, pad_token='<pad>', unk_token='<unk>'):
        token2idx = {pad_token: 0, unk_token: 1}
        for text in corpus:
            for token in cls.tokenize(text):
                if token not in token2idx:
                    token2idx[token] = len(token2idx)

        return cls(token2idx, pad_token=pad_token, unk_token=unk_token)

    @staticmethod
    def tokenize(text):
        return text.split() # nltk.word_tokenize(text.lower(), preserve_line=True)

    def convert_tokens_to_ids(self, tokens):
        return [self.token2idx[t] if t in self.token2idx else 1 for t in tokens]

    def __call__(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

def build_tokenizer(data_dir, cache_dir='caches', use_fast=True):
    print('>> loading cached tokenizer from {}'.format(data_dir))
    tokenizer = BertTokenizer.from_pretrained(data_dir, cache_dir=cache_dir, use_fast=True)

    return tokenizer

def build_embedding(data_dir, cache_dir='cahces', use_adapter=False):
    print('>> loading cached embedding from {}'.format(data_dir))
    config = BertConfig.from_pretrained(data_dir, cache_dir=cache_dir)   
    if use_adapter:
        embedding = BertAdapterModel.from_pretrained(data_dir, config=config, cache_dir=cache_dir)
    else:                   
        embedding = BertModel.from_pretrained(data_dir, config=config, cache_dir=cache_dir)

    return embedding

sentiment2idx = {'NONE': 0, 'NEG': 1, 'NEU': 2, 'POS': 3}
idx2sentiment = {0: 'NONE', 1: 'NEG', 2: 'NEU', 3: 'POS'}
tag2idx = {'O': 0, 'B': 1, 'I': 2}
idx2tag = {0: 'O', 1: 'B', 2: 'I'}

def convert_tags_to_indices(tags):
    return [tag2idx[tag] for tag in tags]

def convert_indices_to_tags(indices):
    return [idx2tag[idx] for idx in indices]

def convert_tags_to_spans(tags):
    spans = []
    start = -1
    for i, tag in enumerate(tags):
        if tag.startswith('B'):
            if start != -1:
                spans.append((start, i - 1))
            start = i
        elif tag.startswith('O'):
            if start != -1:
                spans.append((start, i - 1))
                start = -1
    if start != -1:
        spans.append((start, len(tags) - 1))

    return spans

def pad(indices, max_length=128, pad_idx=0):
    assert len(indices) <= max_length, 'the length exceeds max_length'
    _len = len(indices)
    indices = indices + [pad_idx] * (max_length - _len)
    mask = [1] * _len + [0] * (max_length - _len)
    
    return indices, mask

def build_data(data_dir, tokenizer, max_length=128):
    print('>> building data')

    data_dict = {'train': [], 'dev': [], 'test': []}
    set_types = ['train', 'dev', 'test']
    for set_type in set_types:
        with open(os.path.join(data_dir, '{}.txt'.format(set_type)), 'r', encoding='utf-8') as f:
            for line in f:
                text, triplets = line.strip().split('####')

                # for word piece or byte pair tokenier
                # i am loving you
                # 0 1  2      3
                # [cls] i am love #ing you [sep]
                # 0     1 2  3    4    5   6
                # token_map = {2: (3, 4), ...}
                token_map = {-1: (0, 0)}
                text_tokens = [tokenizer.cls_token]
                for i, token in enumerate(text.split()):
                    token_pieces = tokenizer.tokenize(token)
                    token_map[i] = (len(text_tokens), len(text_tokens) + len(token_pieces) - 1)
                    text_tokens.extend(token_pieces)
                text_tokens.append(tokenizer.sep_token)

                text_len = len(text_tokens)
                text_indices, text_mask = pad(tokenizer.convert_tokens_to_ids(text_tokens), max_length=max_length, pad_idx=tokenizer.pad_token_id)
                
                target_tags = ['O'] * text_len
                opinion_tags = ['O'] * text_len

                sentiment_indices = np.zeros((text_len, text_len), dtype=np.int64)
                
                eval_triplets = []

                for triplet in eval(triplets):
                    t_beg, t_end = triplet[0][0], triplet[0][-1]
                    t_beg, t_end = token_map[t_beg][0], token_map[t_end][1]
                    o_beg, o_end = triplet[1][0], triplet[1][-1]
                    o_beg, o_end = token_map[o_beg][0], token_map[o_end][1]
                    s = sentiment2idx[triplet[2]]

                    target_tags[t_beg: t_end+1] = ['B'] + ['I'] * (t_end - t_beg)
                    opinion_tags[o_beg: o_end+1] = ['B'] + ['I'] * (o_end - o_beg)

                    # bidirectional interplay
                    sentiment_indices[t_beg: t_end+1, o_beg: o_end+1] = s
                    sentiment_indices[o_beg: o_end+1, t_beg: t_end+1] = s

                    eval_triplets.append('-'.join(map(str, (t_beg, t_end, o_beg, o_end, s)))) # string for memory and compute efficiency

                target_indices = convert_tags_to_indices(target_tags)
                opinion_indices = convert_tags_to_indices(opinion_tags)

                target_indices, _ = pad(target_indices, max_length=max_length)
                opinion_indices, _ = pad(opinion_indices, max_length=max_length)

                # default zero paddings
                sentiment_indices = np.pad(sentiment_indices, ((0, max_length - text_len), (0, max_length - text_len)), 'constant')
                
                data = {
                    'text_indices': text_indices,
                    'text_mask': text_mask,
                    'target_indices': target_indices,
                    'opinion_indices': opinion_indices,
                    'sentiment_indices': sentiment_indices,
                    'eval_triplets': eval_triplets,
                }

                data_dict[set_type].append(data)

    return data_dict