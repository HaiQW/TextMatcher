#!/usr/bin/python
#****************************************************************#
# ScriptName: data_loader.py
# Author: hq_weng@zju.edu.cn
# Create Date: 2018-10-17 21:57
# Modify Author: @zju.edu.cn
# Modify Date: 2018-10-21 12:53
# Function: 
#***************************************************************#
import codecs

import numpy as np


def load_embedding(filename, dimension=100):
    """
    Load the pre-trained Chinese word embedding from file.

    Parameters
    ----------
    filename: str
        The file name of the pre-trained Chinese word embedding.
	dimension:
		The dimension of each word embedding to be loaded.

    Return
    ------
    np.ndarray: the numpy array containing embedding weight
    dict: word2id dict.
    dict: id2word dict.
    """
    embeddings = []
    words = []
    word2id = dict()
    id2word = dict()
    embeddings.append([0] * dimension)
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip().split(' ')
            word = line[0].strip()
            embedding = [float(x) for x in line[1:] if len(x) > 0]
            if len(embedding) >= dimension:
                embeddings.append(embedding[0:dimension])
                words.append(word)
                word2id[word] = i + 1
                id2word[i + 1] = word

    embeddings = np.array(embeddings)
    return embeddings, word2id, id2word


def encode_setence(file_name, word2id, label):
    """
    Encode the sentence and prepare the training dataset.

    Parameters
    ----------
    file_name: str
        The name of the file that contain texts to be encoded.
    word2id: dict
        The encoding dictionary.
    label: int
        The category label of the encoding texts.  

    Return
    ------
    list
        list of sentence codes.
    list
        list of sentence labels.
    """
    keys = word2id.keys()
    sentence_codes = []
    labels = []
    lengths = []
    with codecs.open(file_name, 'r', 'utf-8') as data_file:
        for line in data_file.readlines():
            codes = [
                word2id[w] 
                for w in line.strip().split(' ') if w in keys]
            if len(codes) > 0:
                sentence_codes.append(codes)
                labels.append(label)
                lengths.append(len(codes))
            else:
                continue
    
    return sentence_codes, labels
