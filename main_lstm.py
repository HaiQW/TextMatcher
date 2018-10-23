# -*- coding: utf-8 -*-
#!/usr/bin/python
#****************************************************************#
# ScriptName: main_lstm.py
# Author: hq_weng@zju.edu.cn
# Create Date: 2018-10-10 21:11
# Modify Author: hq_weng@zju.edu.cn
# Modify Date: 2018-10-11 11:56
# Function: 
#***************************************************************#
import os
import sys
import random
import codecs
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

from utils.dataset import TextDataset
from utils.data_loader import load_embedding, encode_setence
from models.lstm import LSTMClassifier, train_lstm, evaluate_lstm

torch.manual_seed(1)
os.environ['PYTHONHASHSEED'] = '0'


def get_padding_codes(sentence_codes):
    lengths = [len(code) for code in sentence_codes]
    padding_codes = np.zeros((len(sentence_codes), max(lengths)))
    for i, code in enumerate(sentence_codes):
        padding_codes[i, 0:len(code)] = code

    return padding_codes, lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--phase', 
        type=str, 
        help='Train or test.')
    parser.add_argument(
        '--embedding_file', 
        type=str, 
        help='Filename to save the trained word embeddings.')
    parser.add_argument(
        '--model_path', 
        type=str, 
        help='The file of the lstm model.')
    parser.add_argument(
        '--test_file', 
        type=str, 
        help='The file of the tesing data.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='The number of training epochs.'
    ) 
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='The batch size of the training phrase.'
    )
    args = parser.parse_args()
    phase = args.phase
    embedding_file = args.embedding_file
    model_path = args.model_path

    embeddings, word2id, id2word = load_embedding(embedding_file)
    id2label = dict({
        0: u'游戏', 
        1: u'角色扮演', 
        2: u'moba',
        3: u'运动', 
        4: u'三国', 
        5: u'战争', 
        6: u'服饰', 
        7: u'T恤', 
        8: u'婚姻'
    })

    EMBEDDING_DIM = 100
    HIDDEN_DIM = 200
    LINEAR_HIDDEN_DIM = 100 
    N_CLASSES = len(id2label)

    # Create the lstm model
    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, LINEAR_HIDDEN_DIM, 
                           len(word2id.keys()), N_CLASSES, embeddings)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    print(model) 

    if phase == 'train':
        print('Load the training data and prepare labels...')
        game_roleplay = 'data/train/1k_std_rollplay.word'
        game_moba = 'data/train/1k_std_moba.word'
        game_sport = 'data/train/1k_std_sport_game.word'
        sanguo_battle = 'data/train/1k_std_sanguo.word'
        cloth_shirt = 'data/train/1k_std_cloth.word'
        marriage = 'data/train/1k_std_marriage.word' 
        sport = 'data/train/1k_std_sport.word'
        
        corpus2label = dict({
            'game_roleplay': (game_roleplay, [1, 1, 0, 0, 0, 0, 0, 0, 0]), 
            'game_moba': (game_moba, [1, 0, 1, 0, 0, 0, 0, 0, 0]),
            'game_sport': (game_sport, [1, 0, 0, 1, 0, 0, 0, 0, 0]),
            'sanguo_battle': (sanguo_battle, [0, 0, 0, 0, 1, 1, 0, 0, 0]),
            'cloth_shirt': (cloth_shirt, [0, 0, 0, 0, 0, 0, 1, 1, 0]),
            'marriage': (marriage, [0, 0, 0, 0, 0, 0, 0, 0, 1]),
            'sport': (sport, [0, 0, 0, 1, 0, 0, 0, 0, 0])
        }) 
 
        corpus_data = []
        labels = []
        for file_name, label in corpus2label.values():
            print(file_name, label)
            tmp_codes, tmp_labels = encode_setence(file_name, word2id, label)
            corpus_data.extend(tmp_codes)
            labels.extend(tmp_labels)
        
        
        corpus_data, lengths = get_padding_codes(corpus_data)
        corpus_data = torch.tensor(np.array(corpus_data), dtype=torch.long)
        lengths = torch.tensor(np.array(lengths), dtype=torch.long)
        labels = torch.tensor(np.array(labels), dtype=torch.float)
    
        # Train and validate
        # labels = np.array(labels)
        train_size = int(corpus_data.shape[0] * 0.8)
        indices = list(range(corpus_data.shape[0]))
        random.shuffle(indices)
        train_indices = indices[0:train_size]
        validate_indices = indices[train_size:]
        
        train_data = corpus_data[train_indices, :]
        train_labels = labels[train_indices, :]
        train_lengths = lengths[train_indices]
        validate_data = corpus_data[validate_indices, :]
        validate_labels = labels[validate_indices, :]
        validate_lengths = lengths[validate_indices]


        # bind variables to cuda
        if torch.cuda.is_available:
            train_data = train_data.cuda()
            train_lengths = train_lengths.cuda()
            train_labels = train_labels.cuda()
            validate_data = validate_data.cuda()
            validate_labels = validate_labels.cuda()
            validate_lengths = validate_lengths.cuda()
            model.cuda()
    
        text_data = TextDataset(train_data, train_labels, train_lengths)
        train_dataloader = data.DataLoader(text_data, batch_size=args.batch_size, shuffle=True) 

        print('Train the LSTM text classifier model...')
        train_lstm(model,
                   model_path, 
                   optimizer, 
                   train_dataloader,  
                   validate_data, validate_labels, validate_lengths, 
                   args.epochs)

    if phase == 'test':
        test_file = args.test_file
        model.load_state_dict(torch.load(model_path)) 
        optimizer.zero_grad()
        test_data, labels = encode_setence(test_file, word2id, 1)
        padding_test_data, lengths = get_padding_codes(test_data)
        padding_test_data = torch.tensor(np.array(padding_test_data), dtype=torch.long)
        lengths = torch.tensor(np.array(lengths), dtype=torch.long)
        scores = evaluate_lstm(model, padding_test_data, lengths) 
        scoers = scores.data.cpu().numpy()

        # for print the result
        for idx, score in enumerate(scores):
            sentence = [id2word[int(code)] for code in test_data[idx]]
            tmp_labels = [id2label[i] for i in np.where(score > 0.5)[0]]
            tmp_score = np.array([float(score[i]) for i in np.where(score > 0.5)[0]]) 
            tmp_score = tmp_score.prod()
            print(idx), 
            print(' '.join(sentence).encode('utf-8').decode('utf-8'))
            print(' '.join(tmp_labels).encode('utf-8').decode('utf-8')),
            print(tmp_score) 


if __name__ == '__main__':
    main()
