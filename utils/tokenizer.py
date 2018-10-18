# -*- coding: utf-8 -*-
#!/usr/bin/python
#****************************************************************#
# ScriptName: data_processor.py
# Author: hq_weng@zju.edu.cn
# Create Date: 2018-10-09 14:48
# Modify Author: @zju.edu.cn
# Modify Date: 2018-10-09 14:49
# Function: process the raw text data 
#***************************************************************#
import sys
import codecs

import jieba
import numpy as np

DICT_FILENAME = 'data/dict.txt'
jieba.load_userdict(DICT_FILENAME)


def load_stopwords(file_name):
    """load the stopwords"""
    stopwords = []
    with open(file_name, 'r') as fin:
        for line in fin.readlines():
            stopwords.append(line.decode('utf-8')[:-1])
    return stopwords


def text2words(raw_data, stopwords):
    """Use the jieba fenci package to segment the raw data.
    Then, remove stopwords.
    """
    orig_words = list(jieba.cut(raw_data, cut_all=False))
    words = [
        word for word in orig_words 
        if (not word in stopwords) and (
            not word in [u' ', 'NULL', '/']) and (len(word) >= 2)
    ]
    
    return words


def file2words(filename, stopwords_name):
    # load dict
    stopwords = load_stopwords(stopwords_name) 
    count = 0
    results = []
    with codecs.open(infile, 'r', 'utf-8') as f:
        for line in f: 
            count += 1
            line = line.strip()
            if len(line) <= 0:
                continue
            if count % 10000 == 0:
                print('INFO: process line: %d' % count)
            try:
                words = ' '.join(text2words(line.encode('utf-8'), stopwords))
                if len(result) > 0:
                    results.append(words)
            except:
                print(result.decode('utf-8'))
                continue
    return results


def file2wordfile(infile, outfile, stopwords_name):
    # load dict
    stopwords = load_stopwords(stopwords_name) 
    outfile = codecs.open(outfile, 'w', 'utf-8') 
    count = 0
    with codecs.open(infile, 'r', 'utf-8') as f:
        for line in f: 
            count += 1
            line = line.strip()
            if len(line) <= 0:
                continue
            if count % 10000 == 0:
                print('INFO: process line: %d' % count)
            try:
                result = ' '.join(text2words(line.encode('utf-8'), stopwords))
                outfile.write(result)
                outfile.write('\n')
            except:
                print(result.decode('utf-8'))
                continue
    outfile.close()


def test():
    file_name  = sys.argv[1]
    stopwords_name = 'data/stopwords.txt'
    saved_name = sys.argv[2]
    results = file2wordfile(file_name, saved_name, stopwords_name)

if __name__ == '__main__':
    test()
