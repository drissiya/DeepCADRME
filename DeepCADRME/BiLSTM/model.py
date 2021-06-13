import torch
import torch.nn as nn
import copy
import math
import numpy as np


class BiLSTMModel(nn.Module):
    def __init__(self, args):
        
        super().__init__()
        self.args = args

        self.in_dim = self.args.word_dim
        if self.args.emb_type == 'word+pos':
            self.in_dim += self.args.pos_dim
        if self.args.emb_type == 'word+char':
            self.in_dim += self.args.char_dim
        if self.args.emb_type == 'word+pos+char':
            self.in_dim += self.args.pos_dim + self.args.char_dim

        self.word_emb = nn.Embedding(args.size_vocab_word, args.word_dim, padding_idx = args.pad_idx)
        self.word_emb.weight.data.copy_(args.word_matrix)
        self.word_emb.weight.data[args.pad_idx] = torch.zeros(args.word_dim)

        self.pos_emb = nn.Embedding(args.size_vocab_pos, args.pos_dim, padding_idx = args.pad_idx)
        self.pos_emb.weight.data.copy_(args.pos_matrix)
        self.pos_emb.weight.data[args.pad_idx] = torch.zeros(args.pos_dim)

        self.char_emb = nn.Embedding(args.size_vocab_word, args.char_dim, padding_idx = args.pad_idx)
        self.char_emb.weight.data.copy_(args.char_matrix)
        self.char_emb.weight.data[args.pad_idx] = torch.zeros(args.char_dim)
        self.char_emb.weight.requires_grad = True
        
        self.lstm = nn.LSTM(self.in_dim, args.hidden_lstm, bidirectional=True)
        
        self.fc = nn.Linear(args.hidden_lstm * 2, args.num_labels)
        
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, word_ids, pos_ids, char_ids):
        word_input = self.word_emb(word_ids)
        embs = [word_input]
        
        if self.args.emb_type == 'word+pos':
            embs += [self.pos_emb(pos_ids)]
        if self.args.emb_type == 'word+char':
            embs += [self.char_emb(char_ids)]
        if self.args.emb_type == 'word+pos+char':
            embs += [self.pos_emb(pos_ids)] + [self.char_emb(char_ids)]
 
            
        embs = torch.cat(embs, dim=2)
        embs = self.dropout(embs)        
        outputs, _ = self.lstm(embs)   
        currentLevel = self.fc(self.dropout(outputs))        
        return currentLevel