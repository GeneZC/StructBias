# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Biaffine(nn.Module):
    def __init__(self, from_size, to_size, out_size, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.from_size = from_size
        self.to_size = to_size
        self.out_size = out_size
        self.bias = bias
        self.U = nn.Parameter(torch.empty(from_size + int(bias[0]), out_size * (to_size + int(bias[1]))))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))

    def forward(self, from_tensor, to_tensor):
        if self.bias[0]:
            from_tensor = torch.cat((from_tensor, torch.ones_like(from_tensor[..., :1])), dim=-1)
        if self.bias[1]:
            to_tensor = torch.cat((to_tensor, torch.ones_like(to_tensor[..., :1])), dim=-1)
        
        batch_size, from_len, _ = from_tensor.shape
        to_len = to_tensor.shape[1]

        affine = torch.matmul(from_tensor, self.U) # b, x, o*t
        affine = affine.reshape(batch_size, from_len * self.out_size, self.to_size + int(self.bias[1]))
        biaffine = torch.matmul(affine, to_tensor.transpose(1, 2))
        biaffine = biaffine.reshape(batch_size, from_len, self.out_size, to_len)
        biaffine = biaffine.transpose(2, 3).contiguous()

        return biaffine

class MultiHeadBiaffine(nn.Module):
    def __init__(self, from_size, to_size, out_size, bias=(True, True)):
        super(MultiHeadBiaffine, self).__init__()
        self.from_size = from_size
        self.to_size = to_size
        self.out_size = out_size
        self.bias = bias
        self.U = nn.Parameter(torch.empty(from_size + int(bias[0]), out_size * (to_size + int(bias[1]))))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))

    def forward(self, from_tensor, to_tensor):
        if self.bias[0]:
            from_tensor = torch.cat((from_tensor, torch.ones_like(from_tensor[..., :1])), dim=-1)
        if self.bias[1]:
            to_tensor = torch.cat((to_tensor, torch.ones_like(to_tensor[..., :1])), dim=-1)
        
        batch_size, from_len, _ = from_tensor.shape
        to_len = to_tensor.shape[1]

        affine = torch.matmul(from_tensor, self.U) # b, x, o, t
        affine = affine.reshape(batch_size, from_len * self.out_size, self.to_size + int(self.bias[1]))
        biaffine = torch.matmul(affine, to_tensor.transpose(1, 2))
        biaffine = biaffine.reshape(batch_size, from_len, self.out_size, to_len)
        biaffine = biaffine.transpose(2, 3).contiguous()

        return biaffine

class BiaffineEinsum(nn.Module):
    def __init__(self, from_size, to_size, out_size, bias=(True, True)):
        super(BiaffineEinsum, self).__init__()
        self.from_size = from_size
        self.to_size = to_size
        self.out_size = out_size
        self.bias = bias
        self.U = nn.Parameter(torch.empty(from_size + int(bias[0]), out_size, to_size + int(bias[1])))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        
    def forward(self, from_tensor, to_tensor):
        if self.bias[0]:
            from_tensor = torch.cat((from_tensor, torch.ones_like(from_tensor[..., :1])), dim=-1)
        if self.bias[1]:
            to_tensor = torch.cat((to_tensor, torch.ones_like(to_tensor[..., :1])), dim=-1)
    
        biaffine = torch.einsum('bxf,fot,byt->bxyo', from_tensor, self.U, to_tensor)

        return biaffine