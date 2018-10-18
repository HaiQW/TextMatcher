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


def packed_batch(batch_in, lengths):
    # lengths = torch.FloatTensor(lengths)
    _, idx_sort = torch.sort(lengths, dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)
    batch_in = batch_in.index_select(0, idx_sort)
    lengths = list(lengths[idx_sort].data.numpy())
    batch_packed = nn.utils.rnn.pack_padded_sequence(batch_in, lengths, batch_first=True)
    return batch_packed


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--phase', 
        type=str, 
        help='Train or test.')
    parser.add_argument(
        '--embedding_file', 
        type=str, 
        default='trained_models/40w_embedding.txt',
        help='Filename to save the trained word embeddings.')
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='trained_models/model.lstm', 
        help='The file of the lstm model.')

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

    # create the lstm model.
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 200
    LINEAR_HIDDEN_DIM = 100 
    N_CLASSES = len(id2label)
    EPOCHS = 100
    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, LINEAR_HIDDEN_DIM, len(word2id.keys()), 
                           N_CLASSES, embeddings)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    print(model) 

    if phase == 'train':
        print('Train the LSTM text classifier model...')
        # Load training data and prepare labels
        rollplay_data_file = 'data/train/1k_std_rollplay.word'
        moba_data_file = 'data/train/1k_std_moba.word'
        sportgame_data_file = 'data/train/1k_std_sport_game.word'
        sanguo_data_file = 'data/train/1k_std_sanguo.word'
        cloth_data_file = 'data/train/1k_std_cloth.word'
        marriage_data_file = 'data/train/1k_std_marriage.word' 
        sport_data_file = 'data/train/1k_std_sport.word'
        corpus_files = [
            rollplay_data_file, 
            moba_data_file, 
            sportgame_data_file, 
            sanguo_data_file,
            cloth_data_file,
            marriage_data_file,
            sport_data_file
        ]
        text_labels = [
            [1, 1, 0, 0, 0, 0, 0, 0, 0], 
            [1, 0, 1, 0, 0, 0, 0, 0, 0], 
            [1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0]
        ]

        training_data = []
        labels = []
        for file_name, label in zip(corpus_files, text_labels):
            print(file_name, label)
            tmp_codes, tmp_labels = encode_setence(file_name, word2id, label)
            training_data.extend(tmp_codes)
            labels.extend(tmp_labels)

        training_data, lengths = get_padding_codes(training_data)
        training_data = torch.tensor(np.array(training_data), dtype=torch.long)
        lengths = torch.tensor(np.array(lengths), dtype=torch.long)
        labels = torch.tensor(np.array(labels), dtype=torch.float)

        # bind variables to cuda
        if torch.cuda.is_available:
            training_data = training_data.cuda()
            lengths = lengths.cuda()
            labels = labels.cuda()
            model.cuda()
    
        text_data = TextDataset(training_data, labels, lengths)
        dataloader = data.DataLoader(text_data, batch_size=50, shuffle=True) 
        train_lstm(model, model_path, optimizer, dataloader, labels, EPOCHS)

    if phase == 'test':
        test_file = 'test.word'
        model.load_state_dict(torch.load(model_path)) 
        optimizer.zero_grad()
        testing_data, labels = encode_setence(test_file, word2id, 1)
        testing_data, lengths = get_padding_codes(testing_data)
        testing_data = torch.tensor(np.array(testing_data), dtype=torch.long)
        lengths = torch.tensor(np.array(lengths), dtype=torch.long)
        scores = evaluate_lstm(model, testing_data, lengths) 
        scoers = scores.data.cpu().numpy()
        for idx, score in enumerate(scores):
            labels = [id2label[i] for i in np.where(score > 0.5)[0]]
            print(idx), 
            print(' '.join(labels).encode('utf-8'))

if __name__ == '__main__':
    train()
