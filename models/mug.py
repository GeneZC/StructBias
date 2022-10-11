# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.biaffine import Biaffine, BiaffineEinsum
from data_utils import convert_indices_to_tags, convert_tags_to_spans

class MuG(nn.Module):
    def __init__(self, args, embedding):
        super(MuG, self).__init__()
        self.embed = embedding

        self.target_fc = nn.Linear(args.hidden_size, args.hidden_size)
        self.opinion_fc = nn.Linear(args.hidden_size, args.hidden_size)

        self.target_tagger = nn.Linear(args.hidden_size // 2, args.tag_size)
        self.opinion_tagger = nn.Linear(args.hidden_size // 2, args.tag_size)
        self.sentiment_parser = BiaffineEinsum(args.hidden_size // 2, args.hidden_size // 2, args.sentiment_size, bias=(True, True))

    def routine(self, text_indices, text_mask):
        hid = self.embed(text_indices, attention_mask=text_mask)[0]
        
        target_rep = F.relu(self.target_fc(hid))
        opinion_rep = F.relu(self.opinion_fc(hid))
        target_tag_rep, target_sent_rep = torch.chunk(target_rep, 2, dim=2)
        opinion_tag_rep, opinion_sent_rep = torch.chunk(opinion_rep, 2, dim=2)

        target_out = self.target_tagger(target_tag_rep)
        opinion_out = self.opinion_tagger(opinion_tag_rep)
        sentiment_out = self.sentiment_parser(target_sent_rep, opinion_sent_rep)

        return target_out, opinion_out, sentiment_out

    def forward(self, inputs):
        text_indices, text_mask, target_indices, opinion_indices, sentiment_indices = inputs

        target_out, opinion_out, sentiment_out = self.routine(text_indices, text_mask)        

        loss = self.loss_fn(target_out, target_indices, opinion_out, opinion_indices, sentiment_out, sentiment_indices, mask=text_mask)

        return loss

    def predict(self, inputs):
        text_indices, text_mask, target_indices, opinion_indices, sentiment_indices = inputs

        target_out, opinion_out, sentiment_out = self.routine(text_indices, text_mask)

        loss = self.loss_fn(target_out, target_indices, opinion_out, opinion_indices, sentiment_out, sentiment_indices, mask=text_mask)
        triplets = self.decode(target_out, opinion_out, sentiment_out, mask=text_mask)

        return loss.item(), triplets

    @staticmethod
    def loss_fn(target_out, target_tgt, opinion_out, opinion_tgt, sentiment_out, sentiment_tgt, mask):
        # tag loss
        target_loss = F.cross_entropy(target_out.flatten(0, 1), target_tgt.flatten(0, 1), reduction='none')
        target_loss = target_loss.masked_select(mask.bool().flatten(0, 1)).sum() / mask.sum()
        opinion_loss = F.cross_entropy(opinion_out.flatten(0, 1), opinion_tgt.flatten(0, 1), reduction='none')
        opinion_loss = opinion_loss.masked_select(mask.bool().flatten(0, 1)).sum() / mask.sum()
        tag_loss = target_loss + opinion_loss
        # sentiment loss
        mat_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        sentiment_loss = F.cross_entropy(sentiment_out.reshape(-1, sentiment_out.shape[-1]), sentiment_tgt.reshape(-1), reduction='none')
        sentiment_loss = sentiment_loss.masked_select(mat_mask.bool().reshape(-1)).sum() / mat_mask.sum()

        return tag_loss + sentiment_loss

    @staticmethod
    def decode(target_out, opinion_out, sentiment_out, mask):
        target_out = target_out.argmax(-1)
        opinion_out = opinion_out.argmax(-1)
        sentiment_out = sentiment_out.argmax(-1)

        batch_size = target_out.shape[0]

        target_indices = []
        opinion_indices = []
        for b in range(batch_size):
            target_indices.append(target_out[b].masked_select(mask[b].bool()).cpu().numpy().tolist())
        for b in range(batch_size):
            opinion_indices.append(opinion_out[b].masked_select(mask[b].bool()).cpu().numpy().tolist())

        target_spans = [convert_tags_to_spans(convert_indices_to_tags(indices)) for indices in target_indices]
        opinion_spans = [convert_tags_to_spans(convert_indices_to_tags(indices)) for indices in opinion_indices]
        
        triplets = []
        for b in range(batch_size):
            _triplets = []
            for t_beg, t_end in target_spans[b]:
                for o_beg, o_end in opinion_spans[b]:
                    s_dist = [0] * 4
                    for i in range(t_beg, t_end + 1):
                        for j in range(o_beg, o_end + 1):
                            s_dist[sentiment_out[b, i, j]] += 1
                            s_dist[sentiment_out[b, j, i]] += 1
                    s = s_dist.index(max(s_dist))
                    if s == 0:
                        continue
                    _triplets.append('-'.join(map(str, (t_beg, t_end, o_beg, o_end, s))))
            triplets.append(_triplets)
        
        return triplets
