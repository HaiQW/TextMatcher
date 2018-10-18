#!/usr/bin/python
#****************************************************************#
# ScriptName: word2vec.py
# Author: hq_weng@zju.edu.cn
# Create Date: 2018-10-12 15:42
# Modify Author: @zju.edu.cn
# Modify Date: 2018-10-17 11:40
# Function: Word2Vec model based on the genism package
#***************************************************************#
import sys
import codecs
from itertools import chain

import nltk
import gensim


class Word2Vec(object):

    def __init__(self, infile, outfile):
       self.infile = infile
       self.outfile = outfile
       self.corpus = self._get_corpus(infile)

    def _get_corpus(self, filename):
        """Get the corpus data directly form file"""
        corpus = []
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f.readlines():
                words = line.split()
                corpus.append(words)

        return corpus
    
    def _get_min_count(self, corpus, vocab_size):
        fdist = nltk.FreqDist(chain.from_iterable(corpus))	
        min_count = fdist.most_common(vocab_size)[-1][1]
        return min_count
    
    def make_embedding(self, 
                       vocab_size=100000, # vocab size
                 	   vector_size=100, # vector embedding dimension
                 	   min_count=5, 
                 	   negative=5, 
                 	   window=5):
		# get min count
        corpus = self.corpus
        min_count = self._get_min_count(corpus, vocab_size)
        self.model = gensim.models.Word2Vec(
            corpus, 
            size=vector_size, 
            min_count=min_count,
            negative=negative,
            window=window) 

        # train model
        self.model.train(
            corpus, 
            epochs=self.model.iter, 
            total_examples=self.model.corpus_count)

        # save model
        self.model.save('%s.bin' % self.outfile)
        # save word2vec
        with codecs.open(self.outfile, 'w', 'utf-8') as f:
            for i, word in enumerate(self.model.wv.index2word):
            	e = self.model[word]
            	e = ' '.join(map(lambda x: str(x), e)) 
            	f.write(u"%s %s \n" % (word.encode('utf8').decode('utf8'), e))


def test():
    """
    infile = sys.argv[1]
    outfile = sys.argv[2] # to save the word embeddings
    vocab_size = 200000
    w = Word2Vec(infile, outfile)
    w.make_embedding(vocab_size=vocab_size)
    """
    pass
