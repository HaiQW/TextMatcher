# TextMatcher
An NLP program used for short text classification.

# Task 
在没有训练样本情况下，借助程序查取搜索，百科等网络资料，实现算法程序来自动计算以下文本标题:

1. 仙剑奇侠传开服啦，现在报名马上得纪念T恤
2. 浪漫情人节，上线QQ飞车，马上送99朵玫瑰
3. 诸葛亮太好用了，简直是排位的大杀器
4. 诸葛亮周瑜火烧赤壁大败曹操

# Environment
  - python 2.7
  - pytorch 0.4.0

# Solution

### 1. Download the repo, the pretrained word embeddings, and the pretraimed LSTM model

  - Repo: git clone https://github.com/HaiQW/TextMatcher
  
  This corpus is a combination of the whole Chinese wikipedia corpus and part of Baidu baike corpus.
  Also, you can prepare your use-specific corpus if you prefer.
 
  - Pretrained Word embeddings: https://www.dropbox.com/s/2sh2c6n4x17avbe/40w_embedding.txt?dl=0
  
  Thoese word embeddings contain a totol of 400,000+ words.  
  
  - Pretrained LSTM model: https://www.dropbox.com/s/vm6fhcuzxjortjd/model.lstm?dl=0


### 2. Train the word embeddings (optional).

  If you prefer to train your own word embeddings, you must prepare your corpus first (the corpus used to train my
  word embeddings is not updolated online due to the limitation of free file sharing space).
  
  - Train word embeddings: python main_word2vec.py --corpus_file=path/to/corpus/file
  --embedding_file=path/to/save/embedding/file 
  
  For your convenience, you can simply use the pretrained word embeddings.

### 3. Train the LSTM multilabel classifier.
  - Train: python main_lstm.py  --phase=train --embedding_file=path/to/file --model_path=path/to/model
  - Test: python main_lstm.py  --phase=test --testing_file=path/to/testing/file

### 4. Predict the label using the naive method.
  - python main_naive.py --embedding_file=trained_models/40w_embedding.txt --testing_file=data/test/test.word 



