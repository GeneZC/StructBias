# -*- coding: utf-8 -*-

from models.mug import MuG

MODEL_CLASS_MAP = {
    'mug': MuG,
}

INPUT_FIELDS_MAP = {
    'mug': ['text_indices', 'text_mask', 'target_indices', 'opinion_indices', 'sentiment_indices'],
}

def build_model_class(model_name):
    return MODEL_CLASS_MAP[model_name], INPUT_FIELDS_MAP[model_name]
