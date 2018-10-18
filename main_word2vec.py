# -*- coding: utf-8 -*-
#!/usr/bin/python
#****************************************************************#
# ScriptName: main_word2vec.py
# Author: hq_weng@zju.edu.cn
# Create Date: 2018-10-17 21:13
# Modify Author: hq_weng@zju.edu.cn
# Modify Date: 2018-10-17 21:13
# Function: Train the Chinese word embeddings. 
#***************************************************************#
from __future__ import print_function 
from __future__ import division

import argparse

from models.word2vec import Word2Vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--corpus_file', 
        type=str, 
        help='The corpus file to train word embeddings.')
    parser.add_argument(
        '--embedding_file', 
        type=str, 
        help='Filename to save the trained word embeddings.')
    parser.add_argument(
        '--vocab_size', 
        type=int, 
        default=200000, 
        help='The maximum vocabulary size to be trained.')

    args = parser.parse_args()
    corpus_file = args.corpus_file
    embedding_file = args.embedding_file
    vocab_size = args.vocab_size
    model = Word2Vec(corpus_file, embedding_file)
    model.make_embedding(vocab_size=vocab_size)


if __name__ == '__main__':
    main()
