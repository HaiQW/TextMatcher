# -*- encoding: utf-8 -*-
#!/usr/bin/python
#****************************************************************#
# ScriptName: models/lstm.py
# Author: hq_weng@zju.edu.cn
# Create Date: 2018-10-10 22:29
# Modify Author: hq_weng@zju.edu.cn
# Modify Date: 2018-10-16 11:58
# Function: Simple lstm based multi-label classifier 
#***************************************************************#
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class LSTMClassifier(nn.Module):
    """
	Implementation based on 
    'https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html'
	"""
    def __init__(self, 
                 embedding_dim, 
                 hidden_dim, 
                 linear_hidden_dim, 
                 vocab_size, 
                 label_size, 
                 pretained_word_embeddings=None):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.label_size = label_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear_hidden_dim = linear_hidden_dim
        self.topk = 2

        if pretained_word_embeddings is not None:
            self.word_embeddings.weight = nn.Parameter(
                torch.FloatTensor(pretained_word_embeddings))
            # self.word_embeddings.weight.requires_grad = False 
 
        # The LSTM takes word embeddings as inputs, 
        # and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=2,
            bias=True, 
            # batch_first=False,
            bidirectional=True
        )

        # fc layer that maps from hidden state space to tag space	
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 * self.topk, self.linear_hidden_dim),
            nn.BatchNorm1d(linear_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_dim, label_size)
        )

        # self.hidden2label = nn.Linear(hidden_dim * 2, label_size)
        # self.hidden = self.init_hidden(10)

    def init_hidden(self, batch_size):
        """Not used more"""
        return (torch.zeros(4, batch_size, self.hidden_dim),
                torch.zeros(4, batch_size, self.hidden_dim))

    def forward(self, sentence, lengths):
        # packed inputs
        lengths, perm_idx = lengths.sort(0, descending=True)
        sentence = sentence[perm_idx]
        embeds = self.word_embeddings(sentence)
        sentence = sentence.transpose(0, 1) 

        # embeds = self.word_embeddings(sentence)
        embeds = embeds.transpose(0, 1)
        packed_input = pack_padded_sequence(embeds, lengths.cpu().numpy())    
        lstm_out, self.hidden = self.lstm(packed_input) #:, self.hidden) #self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)

        lstm_out = lstm_out.transpose(0, 1)
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = kmax_pooling((lstm_out), 2,  self.topk)
        x = lstm_out
        #x = lstm_out[:, 0, :]
        x = x.view(-1, 2 * self.topk * self.hidden_dim)
        x = self.fc(x)
        return x
   

def lstm_loss(output, label):
    # loss_function = nn.NLLLoss()
    loss_function = nn.MultiLabelSoftMarginLoss()
    loss = loss_function(output, label)
    return loss


def lstm_accuracy(output, label):
    output = (nn.functional.sigmoid(output) > 0.5)
    output = output.data.cpu().numpy() 
    label = label.data.cpu().numpy()
    correct = 0
    size = len(label)
    for idx in range(size):
        pred = output[idx]
        true = label[idx]
        tmp = (pred == true).sum()
        if tmp == len(pred):
            correct += 1
    accuracy = float(correct) / size
    return accuracy


def train_lstm(model, model_path, optimizer, dataloader, validate_data, validate_labels, validate_lengths, epochs):
    for epoch in range(epochs):
        model.train() 
        t = time.time()
        losses = []
        accuracies = []
        for i_batch, (sentence, label, lengths) in enumerate(dataloader):
            model.zero_grad()

            # Step 2. Clear out the hidden state of the LSTM.
            # model.hidden = model.init_hidden(len(label))
            # Step 3. Run our forward pass.
            output = model(sentence, lengths[0]).view(len(label), -1) 
    
            loss = lstm_loss(output, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            accuracies.append(lstm_accuracy(output, label))
            
        # validate 
        model.eval()
        print(validate_lengths)
        validate_output = model(validate_data, validate_lengths)
        validate_loss = lstm_loss(validate_output, validate_labels)
        validate_accuracy = lstm_accuracy(validate_output, validate_labels) 

        print('Epoch: %04d, loss_train: %.4f, acc_train: %4.f,' 
              'loss_validate: %.4f, acc_train: %4.f, time: %.4f' 
                % (epoch + 1, sum(losses) / float(len(losses)), 
                   sum(accuracies) / float(len(accuracies)), 
                   validate_loss, validate_accuracy, time.time() - t))

    torch.save(model.state_dict(), model_path)  
    return model


def evaluate_lstm(model, dataset, lengths):
    model.eval()
    # model.hidden = model.init_hidden(1)
    output = model(dataset, lengths)
    output = nn.functional.sigmoid(output)
    return output


if __name__ == '__main__':
    pass
