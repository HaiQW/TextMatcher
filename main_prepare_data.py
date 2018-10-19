# -*- encoding: utf-8 -*-
#!/usr/bin/python
#****************************************************************#
# ScriptName: main_prepare_data.py
# Author: hq_weng@zju.edu.cn
# Create Date: 2018-10-19 22:24
# Modify Author: @zju.edu.cn
# Modify Date: 2018-10-19 22:24
# Function: prepare the train and test data.
#***************************************************************#
import argparse

from utils.tokenizer import file2wordfile 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw_file', 
        type=str, 
        help='The corpus file containing the training corpus.')
    parser.add_argument(
        '--word_file', 
        type=str, 
        help='Filename to save the word tokens.')
    parser.add_argument(
        '--stopword',
        type=str,
        default='data/stopwords.txt',
        help='The file containing stopwords.'
	)

    args = parser.parse_args()
    raw_file  = args.raw_file
    word_file = args.word_file
    stopword_file = args.stopword
    results = file2wordfile(raw_file, word_file, stopword_file)

if __name__ == '__main__':
    main()

