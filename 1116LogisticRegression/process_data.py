#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : process_data.py
# @Author: Yichun
# @Date  : 2023/11/16 19:51
from random import shuffle


def read_dataset(products_path, reviews_path, id):
    """
    read data
    """
    product_data = []
    with open(products_path.format(id), "r", encoding="utf-8") as product_file:
        for line in product_file.readlines():
            line = line.strip("\n")
            product_data.append(line.split("\t"))

    review_data = []
    with open(reviews_path.format(id), "r", encoding="utf-8") as review_file:
        for line in review_file.readlines():
            line = line.strip("\n")
            review_data.append(line.split("\t"))

    return product_data, review_data

def union_dataset(products_data, reviews_data):
    """
    Merge product and review data
    """
    union_data = []
    for product in products_data:
        for review in reviews_data:
            r_id = review[0]
            if r_id == product[0]:
                union_data.append([product[0], product[2], review[2], review[1], product[1]])
                if 'Sterling Silver Garnet Butterfly Earrings (1.70 CT' in product[2]:
                    print()

    return union_data

def save_dataset(data, save_path):
    """
    Save processed data
    """
    with open(save_path, "w", encoding="utf-8") as file:
        for item in data:
            line = "|-|-|".join(item) + "\n"
            file.writelines(line)



if __name__ == '__main__':
    products_path = "./data/dataset/products-data-{0}.tsv"
    reviews_path = "./data/dataset/reviews-{0}.tsv"
    products_data = []
    reviews_data = []
    for id in range(4):
        product_data, review_data = read_dataset(products_path, reviews_path, id)
        products_data += product_data
        reviews_data += review_data
    union_data = union_dataset(products_data, reviews_data)
    shuffle(union_data)
    # Splitting the data
    train_data = union_data[:int(0.8*len(union_data))]
    dev_data = union_data[int(0.8*len(union_data)):int(0.9*len(union_data))]
    test_data = union_data[int(0.9*len(union_data)):]
    # Saving the split data
    train_save_path = "./data/train.txt"
    dev_save_path = "./data/dev.txt"
    test_save_path = "./data/test.txt"
    save_dataset(train_data, train_save_path)
    save_dataset(dev_data, dev_save_path)
    save_dataset(test_data, test_save_path)
