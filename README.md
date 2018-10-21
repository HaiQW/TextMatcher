# TextMatcher.
An NLP program used for short text classification.

# Task description.

在没有训练样本情况下，借助程序查取搜索，百科等网络资料，实现算法程序来自动计算以下文本标题:

1. 仙剑奇侠传开服啦，现在报名马上得纪念T恤
2. 浪漫情人节，上线QQ飞车，马上送99朵玫瑰
3. 诸葛亮太好用了，简直是排位的大杀器
4. 诸葛亮周瑜火烧赤壁大败曹操

# Environment.
  - python 2.7, python 3.5
  - pytorch 0.4.0 
  

# Solution.

### 1. Download the repo, the pretrained word embeddings, and the pretraimed LSTM model.

  - Repo: git clone https://github.com/HaiQW/TextMatcher
  
  This corpus is a combination of the whole Chinese wikipedia corpus and part of Baidu baike corpus.
  Also, you can prepare your use-specific corpus if you prefer.
 
  - To download the pretrained Word embeddings, run:
  ```bash
  # to download from Dropbox
  wget https://www.dropbox.com/s/2sh2c6n4x17avbe/40w_embedding.txt?dl=0 
  # to download from Google Drive
  (Google Drive): https://drive.google.com/file/d/1JZ21lqcBU3_9nmDu9WTFpMBp9AdLa_x3/view?usp=sharing
  ```
  The above word embeddings contain a totol of 400,000+ words.
  
  - To download the pretrained the LSTM model, run:
  ```base
  # to download from Dropbox
  wget  https://www.dropbox.com/s/vm6fhcuzxjortjd/model.lstm?dl=0
  # to download from Google Drive
  wget https://drive.google.com/file/d/1E5ky_TXmci7nG4H59AG4UY2ZUhHIBShd/view?usp=sharing
  ```


### 2. Train the word embeddings (optional).

  If you prefer to train your own word embeddings, you must prepare your corpus first (the corpus used to train my
  word embeddings is not updolated online due to the limitation of free file sharing space).
  
  - To train word embeddings, run:
  ```bash
  python main_word2vec.py --corpus_file=path/to/corpus/file --embedding_file=path/to/save/embedding/file 
  ```
  For your convenience, you can simply use the pretrained word embeddings.
  

### 3. Train the LSTM multilabel classifier.
  - To train the LSTM classifier, run: 
  ```bash 
  python main_lstm.py --phase=train --embedding_file=path/to/file --model_path=path/to/model
  ```
  
### 4. Match the title label.

  - To predict using the LSTM classifier, run: 
  ```
  python main_lstm.py --phase=test --embedding_file=path/to/file --model_path=path/to/model \
  --test_file=path/to/testing/file
  ```
  - To predict using the naive method (purely base on the word embeddings), run:
  ```bash
  python main_naive.py --embedding_file=trained_models/40w_embedding.txt --test_file=data/test/test.word 
  ```



