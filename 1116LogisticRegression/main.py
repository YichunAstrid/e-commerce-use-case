#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: Yichun
# @Date  : 2023/11/16 19:51

# Importing required libraries
import numpy as np
import torch
import argparse
from utils import load_dataset, Vocab, DataLoader
from config import Config
from train import train
from importlib import import_module

# Setting up argument parsing for command line options
parser = argparse.ArgumentParser(description='TextClassification')
parser.add_argument('--model', type=str, default='logistic_regression', help='logistic_regression')
args = parser.parse_args()

# Main execution block
if __name__ == '__main__':
    # Dataset directory
    dataset = './data/'
    # Configuration setup
    config = Config(dataset=dataset)

    # Seed for reproducibility
    seed = 4
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures reproducibility in CUDA
    torch.backends.cudnn.deterministic = True

    # Loading datasets
    train_CL = load_dataset(config.train_path, config=config)
    dev_CL = load_dataset(config.dev_path)
    test_CL = load_dataset(config.test_path)

    # Initializing the vocabulary and add datasets
    vocab = Vocab()
    vocab.add(dataset=train_CL)
    vocab.add(dataset=dev_CL)
    vocab.add(dataset=test_CL)

    # Creating data loaders for batching
    train_loader = DataLoader(train_CL, config.batch_size)
    dev_loader = DataLoader(dev_CL, config.batch_size)
    test_loader = DataLoader(test_CL, config.batch_size)

    # Instantiating the model
    model_name = args.model
    lib = import_module('models.'+model_name)
    model = lib.Model(len(vocab), config).to(config.device)

    # Train the model
    train(model=model, train_loader=train_loader, dev_loader=dev_loader, config=config, vocab=vocab)