#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : logistic_regression.py
# @Author: Yichun
# @Date  : 2023/11/17 16:38
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, vocab_len, config):
        """
        Model initialization
        """
        super(Model, self).__init__()
        self.num_classes = config.num_classes
        self.embed = nn.Embedding(num_embeddings=vocab_len, embedding_dim=config.embed_dim)
        self.fc = nn.Linear(config.embed_dim, config.num_classes)
        self.ln = nn.LayerNorm(config.num_classes)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, word_ids, label_ids=None):
        """
        Forward propagation
        """
        x = self.embed(word_ids.permute(1, 0))
        x = torch.mean(x, dim=0)  # Average pooling over the sequence dimension
        x = self.fc(x)
        x = self.ln(x)
        label_predict = x

        if label_ids is not None:
            loss = self.loss_fct(label_predict, label_ids)  # loss
        else:
            loss = None

        return loss, label_predict.argmax(dim=-1)
