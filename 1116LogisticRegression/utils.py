#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: Yichun
# @Date  : 2023/11/16 19:51
import torch
import pandas as pd

class ContentLabel(object):
    """
    Data instance, including the content (text) and its associated label
    """
    def __init__(self, content, label):
        self.content = content
        self.label = label
    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


def load_dataset(file_path, config=None, test_file=False):
    """
    Loads data from a file and organizes it into ContentLabel instances
    """
    dataset = []
    if config:
        config.class_list = set()
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                line = line.strip('\n')
                content, label = " ".join(line.split('|-|-|')[1:-1]), line.split('|-|-|')[-1]
                dataset.append(ContentLabel(content, label))
                config.class_list.add(label)
        config.class_list = list(config.class_list)
        config.num_classes = len(config.class_list)
        config.id2class = dict(enumerate(config.class_list))
        config.class2id = {j: i for i, j in config.id2class.items()}
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                line = line.strip('\n')
                content, label = " ".join(line.split("|-|-|")[1:-1]), line.split("|-|-|")[-1]
                dataset.append(ContentLabel(content, label))
    return dataset


class Vocab(object):
    """
    Create a vocabulary from the dataset, mapping each unique word to an integer ID
    """
    def __init__(self):
        self.id2word = None
        self.word2id = {'PAD':0}

    def add(self, dataset, test_file=False):
        id = len(self.word2id)
        for item in dataset:
            for word in item.content.split(" "):
                if word not in self.word2id:
                    self.word2id.update({word: id})
                    id += 1
        self.id2word = {j: i for i, j in self.word2id.items()}
    def __len__(self):
        return len(self.word2id)


class DataLoader(object):
    """
    Data loader for batching the data
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for index in range(len(self.dataset)):
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch):
            yield batch
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def batch_variable(batch_data, vocab, config):
    """
    Convert text to numerical
    """
    batch_size = len(batch_data)
    max_seq_len = config.max_seq
    word_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_ids = torch.zeros((batch_size), dtype=torch.long)

    for index, cl in enumerate(batch_data):
        seq_len = len(cl.content.split(" "))
        if seq_len > max_seq_len:
            cl.content = " ".join(cl.content.split(" ")[:max_seq_len])
            word_ids[index, :max_seq_len] = torch.tensor([vocab.word2id[item] for item in cl.content.split(" ")])
        else:
            word_ids[index, :seq_len] = torch.tensor([vocab.word2id[item] for item in cl.content.split(" ")])
        label_ids[index] = torch.tensor([int(config.class2id[cl.label])])

    return word_ids.to(config.device), label_ids.to(config.device)
