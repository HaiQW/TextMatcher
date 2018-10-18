# -*- coding: utf-8 -*-
#!/usr/bin/python
#****************************************************************#
# ScriptName: test.py
# Author: @zju.edu.cn
# Create Date: 2018-10-08 17:10
# Modify Author: @zju.edu.cn
# Modify Date: 2018-10-15 11:43
# Function: 
#***************************************************************#
import sys
import numpy
import codecs
import argparse

import numpy as np
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity

from utils.data_loader import load_embedding


def label_similarity(words, labels, embeddings, word2id):
    label2embeddings = [embeddings[word2id[label]] for label in labels]
    labelscores = dict.fromkeys(labels, 0.0)
    embeddingkeys = word2id.keys()
    
    # calculate label scores
    for word in words:
        if word not in embeddingkeys:
            continue
        wid = word2id[word] 
        embedding = embeddings[wid:wid+1]
        d = cosine_similarity(embedding, label2embeddings)[0] 
        d = zip(labels, d) 
        for w in d[0:]:
            labelscores[w[0]] += w[1]

    # softmax label scores
    softmax_values = nn.functional.softmax(
        torch.tensor(labelscores.values())).data.cpu().numpy()
    labelscores = zip(labelscores.keys(), softmax_values)
    labelscores = sorted(labelscores, key=lambda x:x[1], reverse=True) 

    return labelscores
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--embedding_file', 
        type=str, 
        help='Filename containing the pretrained word embeddings.')
    parser.add_argument(
        '--testing_file', 
        type=str, 
        help='File name containing the testing examples.')

    args = parser.parse_args() 
    test_filename = args.testing_file
    embedding_file = args.embedding_file
    embeddings, word2id, id2word = load_embedding(embedding_file)
    
    labels = [
        u'游戏', 
        u'角色扮演', 
        u'moba',
        u'运动', 
        u'三国', 
        u'战争', 
        u'服饰', 
        u'T恤', 
        u'婚恋']
    categories = dict({
        u'游戏': [u'moba', u'角色扮演', u'运动'],
        u'三国': [u'战争'],
        u'婚恋': [u''], 
        u'服饰': [u'T恤']})

    with codecs.open(test_filename, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip().split()
            first_label_scores = label_similarity(sentence, categories.keys(), embeddings, word2id)
            score = first_label_scores[0][1]
            first_label = first_label_scores[0][0]
            second_labels = categories[first_label]
            print(line.strip().encode('utf-8'))
            print(first_label.encode('utf-8')), 
            if len(second_labels) > 1:
                second_label_scores = label_similarity(sentence, second_labels, embeddings, word2id)
                second_label = second_label_scores[0][0]
                second_score = second_label_scores[0][1]
                score = score * second_score
                print(second_label.encode('utf-8')), 
            else:
                print(''.join(second_labels).encode('utf-8')), 
            print(score)
            print('---------------------------- \n')
