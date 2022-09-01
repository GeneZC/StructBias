# -*- coding: utf-8 -*-

import os

import numpy as np

import torch

class Sampler:
    def __init__(self, num_instances):
        self.num_instances = num_instances

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return self.num_instances

class BatchSampler:
    def __init__(self, batch_size, sampler, drop_last):
        self.batch_size = batch_size
        self.sampler = sampler
        self.drop_last = drop_last

    def __iter__(self):
        batch_indices = []
        for idx in self.sampler:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if len(batch_indices) > 0 and not self.drop_last:
            yield batch_indices

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore

class SequentialSampler(Sampler):
    def __init__(self, num_instances):
        super(SequentialSampler, self).__init__(num_instances)

    def __iter__(self):
        indices = np.arange(self.num_instances)
        yield from indices

class RandomSampler(Sampler):
    def __init__(self, num_instances):
        super(RandomSampler, self).__init__(num_instances)

    def __iter__(self):
        indices = np.random.permutation(np.arange(self.num_instances))
        yield from indices

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, collate_fn=None):
        self.dataset = dataset
        if shuffle:
            self.batch_sampler = BatchSampler(batch_size, RandomSampler(len(self.dataset)), drop_last)
        else:
            self.batch_sampler = BatchSampler(batch_size, SequentialSampler(len(self.dataset)), drop_last)
        if collate_fn is None:
            self.collate_fn = self.default_collate_fn
        else:
            self.collate_fn = collate_fn

    @staticmethod
    def default_collate_fn(batch):
        new_batch = {k: [] for k in batch[0].keys()}
        for instance in batch:
            for k in instance:
                new_batch[k].append(instance[k])
        
        return {k: torch.tensor(v) for k, v in new_batch.items()}

    def __iter__(self):
        for batch_indices in self.batch_sampler:
            yield self.collate_fn([self.dataset[idx] for idx in batch_indices])

    def __len__(self):
        return len(self.batch_sampler)

class DataCollator:
    def __init__(self, input_fields, device):
        self.input_fields = input_fields
        self.device = device

    def __call__(self, batch):
        new_batch = {k: [] for k in batch[0].keys()}
        for instance in batch:
            for k in instance:
                new_batch[k].append(instance[k])

        for k in new_batch:
            if k in self.input_fields:
                new_batch[k] = torch.tensor(new_batch[k]).to(self.device) # tensorize the data field and move it onto the device, otherwise, keep the data filed as a list

        return new_batch