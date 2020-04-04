# -*- coding: utf-8 -*-

import torch
import torchtext
from torchtext.datasets import text_classification
NGRAMS = 2
import os
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root='./.data', ngrams=NGRAMS, vocab=None)
BATCH_SIZE = 16
